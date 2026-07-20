# Reachability floor — glm52 / dsa_attn_decode >40% HBM

All numbers under the **gate protocol** (harness `timing.py` CUPTI device-span
`max(kernel.end)-min(kernel.start)`, cold-L2 253 MB flush) on the pinned idle
B200 (GPU 2, boost 1965 MHz). Scripts: `profiling/{cupti_floor,gather_ceiling}.py`.

## Constants that govern reachability
- Ceilings (strict >40%): **M16 < 12.533760 us, M32 < 25.067520 us** — both == 3.2 TB/s of the model bytes.
- Model bytes credited: M16 40.11 MB (32768 KV rows, `min(2048*M, S)`, zero-collision), M32 80.22 MB (65536 rows).
- Physical distinct KV rows (indices = per-query `randperm(65536)[:2048]`): **M16 26210 (30.2 MB), M32 41739 (48.1 MB)** — both fit B200 L2 (**126.5 MB**) 2.6–4.2× over.
- Stock kernel is `grid == s_q == M`: 16/32 CTAs on 148 SMs → 0.11/0.22 waves. NCU M16: **DRAM 15.2 % = 1.17 TB/s → 51.6 MB actually read** (1.7× the 30 MB distinct set; L2 hit 31 %). Occupancy wall, not BW wall.

## Cold-L2 contiguous BW-vs-size (torch.sum, near-peak; gate protocol)
| MB | 10 | 20 | 30 | 40 | 48 | 64 | 80 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| TB/s | 1.79 | 2.62 | 3.04 | 3.21 | 3.34 | 3.34 | 3.43 | 3.64 | 4.30 |
| %peak | 22 | 33 | 38 | 40 | 42 | 42 | 43 | 45 | 54 |

Marginal BW ~4.7 TB/s (59 %); small transfers are fill/drain-limited. 40 MB contiguous = 40.2 %.

## Gather-BW ceiling (pure read of the required rows, MAX memory parallelism)
The irreducible read of any correct sparse-MLA kernel — no compute, no combine, no O write.
| shape | distinct-once (perfect dedup) | all-slots (per-query topk, natural L2 dedup) | ceiling |
|---|---|---|---|
| **M16** | 10.88 us / **46.1 %** | **12.42 us / 40.4 %** | 12.534 us |
| **M32** | 15.58 us / **64.3 %** | **20.94 us / 47.9 %** | 25.068 us |

## Verdict
- **M32 is reachable in principle**: gather floor 20.9 us vs 25 us ceiling → ~4 us headroom for compute+reduce.
- **M16 is at/over the physical edge**: the *realistic* gather floor (12.42 us, reading the per-query topk that a correct kernel must) is essentially the 12.53 us ceiling. Any real kernel adds (a) QK/softmax/AV compute and (b) a cross-split reduction — the reduction cannot use a DRAM partial round-trip (FP32 partials for split S are `M*S*64*512*4*2` bytes = 4.2 MB·S for M16; even S=4 adds 17 MB, busting the budget). The only sub-ceiling read (10.88 us, distinct-once) needs a pre-deduplicated row set a per-query kernel cannot form without materializing a gather. So **M16 >40 % has no physical slack**.
- Target requires **both** shapes → **M16 is the blocking no-go**, pending the strongest-attempt confirmation (single-launch split-KV kernel measured on the gate) and independent review.

## Why not the mechanisms already considered
- **Split-heads reshape (free occupancy)**: BLOCKED — stock head64 kernel dispatches only `h_q ∈ {64,128}` (tcgen05 MMA atom `SM100_MMA_...<B_H=64,...>` needs M=64/128). Cannot make smaller head-groups.
- **2-kernel split-KV (Python/Triton/simple fork)**: FP32 partial round-trip busts the byte budget on both shapes (M16 +17–33 MB, M32 +67–134 MB). This is why the plan mandated *in-kernel* reduction.
- **In-kernel cluster/DSM reduction fork (P2)**: the only design that avoids partial traffic; can plausibly clear M32 but still cannot beat M16's gather floor (a pure-read lower bound already ≈ ceiling).
