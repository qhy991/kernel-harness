# Results — glm52/index_score_decode HBM >=82%

## Verdict: physical NO-GO on both shapes (independently confirmed)

Independently confirmed by adversarial Codex review (gpt-5.5:xhigh, transcript
`.humanize/skill/2026-07-20_01-25-48-*/output.md`), which inspected the fork
source and ruled out every in-bound mechanism.

The frozen `deep_gemm.fp8_paged_mqa_logits(clean_logits=False)` path already reads
essentially exactly the bytes it needs (no coalescing waste to recover), and the
82% span-inclusive target sits **above the physically achievable HBM ceiling** for
both shapes:

- **M=32 is the strict absolute-roofline proof**: with the measured span it needs
  8.12 TB/s = **101.5% of the 8.0 TB/s spec peak** — impossible. Even at zero span
  it needs 6.56 TB/s vs the kernel's current ~5.75 TB/s.
- **M=16 is the weakest current-efficiency shape**: its isolated NCU kernel time
  (28.5 us) already exceeds the 82% limit (21.749 us); passing needs ~98% of spec
  peak span-inclusive vs a 64% in-kernel ceiling.

Both must pass; neither can.

## Baselines (3 idle GPU-3 runs, seed hash verified)

Seed SHA-256 `a1969f1148fb273886026f990366583ea50304db373d4089f04870feb86b6995`
matches `prompt.md`. Candidate = stock reference call. `calc_diff = 0.0` all runs.

| M | trial1 | trial2 | trial3 | median us | HBM% | 82% limit |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 54.37% | 54.21% | 54.24% | 32.80 us | ~54.3% | <=21.749 us |
| 32 | 61.33% | 61.37% | 61.25% | 58.16 us | ~61.3% | <=43.498 us |

Stable (timing_spread ~1.01, timing_unstable=false). Gate denominator 8.0 TB/s.

**Reproduced on idle GPU-1** (fresh gate, `work/gpu1_gate_confirm.json`): M16
33.08us/53.9%, M32 58.30us/61.2%, both PASS correctness, calc_diff 0. Baseline is
device-independent.

### Sharpest single kill point (M=16)

The **isolated NCU kernel time (28.5–29.25 us) already exceeds the 82% latency
limit of 21.749 us by ~31%** — measured at the most favorable conditions (boosted
clocks, no device span, autotuned). A conforming kernel would have to be ~24%
faster than the current best in-kernel measurement *and* incur zero device span,
while the pure-bytes floor at realistic peak is 18.61 us, leaving only ~3.1 us for
the entire device span (measured span is ~4 us). M=16 cannot pass.

## AC-1 — reconciling user prior 73% vs local 54/61%

The two numbers measure **different quantities**; they are not in conflict:

- Harness gate = **cold-L2 device-kernel span**, /8.0 TB/s → 54.3% / 61.3%.
- NCU in-kernel `DRAM Throughput` (isolated, boosted, no span) → **64.3% / 75.1%**.
- Warm-L2 CUDA-event timing (`work/time_probe.py`, GPU-1 idle) → 56.2% / 68.2%.

The user's "~73%, close to roofline" is consistent with an **in-kernel / warm
measurement** (NCU M=32 = 75%, warm M=32 = 68%), not the frozen cold-L2 gate.
The authoritative gate is 54/61%; the true roofline the prior was "close to" is
the 64–75% in-kernel ceiling — itself below 82%.

## AC-3 — NCU / roofline / span-floor evidence (raw in `work/`)

Kernel: `sm100_paged_mqa_logits<0,1,32,128,64,1,0,3,5,256,16,128,256,float,float,2>`
(kNumHeads=32, headDim=128, PAGE_KV=64, kNumQStages=3, **kNumKVStages=5**,
SPLIT_KV=256, kSplitsPerChunk=16, 128 producer + 256 math threads = block 384).

| metric | M=16 | M=32 |
|---|---:|---:|
| NCU kernel duration | 28.5–29.25 us | 49.79–49.89 us |
| NCU DRAM Throughput | 64.3% | 75.1% |
| actual DRAM bytes | ~144.2 MB | ~286.3 MB |
| requested (`bytes_hbm`) | 142.67 MB | 285.35 MB |
| **actual/requested** | **1.011** | **1.003** |
| L1/TEX hit | 3.29% | ~3.3% |
| L2 hit | 24.44% | ~24% |
| Grid / #SMs | 148 / 148 | 148 / 148 |
| Waves per SM | 1 | 1 |
| Regs/thread | 168 | 168 |
| Dyn smem/block | 220.67 KB | 220.67 KB |
| Theoretical occupancy | 18.75% | 18.75% |
| Achieved occupancy | ~16.0% | ~16.0% |
| Block limit (regs) | 1 | 1 |
| Block limit (smem) | 1 | 1 |
| Scheduler "No Eligible" | 84.1% | ~84% |
| Dominant stall | L1TEX scoreboard (mem latency) 71–73% | same |

**Finding 1 — bytes are optimal.** actual ≈ requested within ~1%. The 132B FP8
paged rows are read efficiently; there is no wasted-byte / coalescing headroom to
reclaim. Any win must come from *raising sustained DRAM throughput*, not from
moving fewer bytes.

**Finding 2 — the throughput ceiling is occupancy-bound and hard-capped.**
Occupancy is pinned at 18.75% (1 block/SM) by the *simultaneous* register (168)
and shared-memory (220.67 KB ≈ 97% of the 228 KB SM budget) limits, and by the
explicit `__launch_bounds__(384, 1)`. The KV software pipeline is already 5 stages
— adding a stage does not fit in smem. This is why M=16 (fewer Q rows → less
memory-level parallelism per block) sustains only 64% DRAM while M=32 reaches 75%.
Lifting DRAM efficiency to ~100% would require materially higher MLP, which the
occupancy cap precludes without a full redesign that DeepGEMM's own SM100 tuning
(5 stages / 1 block) already reflects.

## Roofline / span-floor accounting (the go/no-go pivot)

NCU-derived realistic HBM peak = 4.93 / 0.6431 = **7.666 TB/s (95.8% of the 8.0
TB/s spec denominator)**. Span = gate − NCU kernel time.

| | M=16 | M=32 |
|---|---:|---:|
| 82% latency limit | 21.749 us | 43.498 us |
| pure floor @100% **spec** peak (8.0) | 17.83 us | 35.67 us |
| pure floor @100% **realistic** peak (7.666) | 18.61 us | 37.22 us |
| measured span (gate − NCU) | 3.55 us | 8.36 us |
| **best case: kernel@100% realistic peak + span** | **22.16 us → 80.5%** | **45.58 us → 78.3%** |
| in-kernel rate required to hit 82% (span-incl) | **7.84 TB/s = 98.0% of spec** | **8.12 TB/s = 101.5% of spec** |

**Both shapes fall below 82% even in the physically-unreachable best case where the
kernel sustains 100% of the realistic HBM peak.** To actually pass, the kernel would
have to sustain 98% (M=16) / >100% (M=32) of *spec* peak span-inclusive — i.e. read
HBM at or beyond its physical maximum with zero span cost. This is impossible.

The current kernel is at 64% / 75% of realistic peak; there is no source mechanism
(row coalescing — already optimal; page arithmetic; split-KV/chunks; persistent
scheduling — already 1 wave over all 148 SMs; deeper pipeline — smem-capped;
epilogue) that closes a gap this large, because the gap is against the memory
system's own ceiling, not against a software inefficiency.

## Fastest correct candidate preserved

`candidate/candidate.py` = stock frozen call, `calc_diff = 0.0`, HBM 54.3/61.3%.
This is the fastest *correct* variant; no faster correct variant exists (the target
is above the physical ceiling).

## Independent review notes (Codex, verbatim points incorporated)

- Confirmed no `kNumKVStages > 5` viable: FP8 paged fixes `num_kv_stages = 5` in
  `csrc/jit_kernels/impls/sm100_mqa_logits.hpp`; shared storage scales with
  `kNumKVStages * SPLIT_KV * head_dim` in `layout/mqa_logits.cuh`, already ~220 KB
  of the 228 KB budget.
- A 2-CTA/SM variant (`SPLIT_KV=128`, `kNumMathThreads=128`, `kNumKVStages=3`,
  `__launch_bounds__(256,2)`) is implementable but has **no defensible expected
  number**: it trades pipeline depth (5→3) and adds split/TMA overhead to chase
  MLP, and would still need to lift M16 from ~5.0 to ≥6.56 TB/s (zero span) or
  ~7.84 TB/s (current span). Not a credible pass path.
- TMA multicast / cluster KV reuse: not applicable — `tokens_per_request=1`, so no
  sibling Q-blocks share a KV tile; cross-request reuse would be cached-KV cheating
  and is not in the ABI.
- Span caveat: `gate − NCU` is not a cleanly-isolated reducible host gap (it mixes
  measurement mode, cache state, clocks). This does not rescue the task: even at
  **zero span** both shapes still require 6.56 TB/s vs ~5.0/5.75 TB/s achieved.
- Byte accounting and peak choice judged sound; occupancy/BW conclusion coherent
  (one-CTA/SM streaming TMA kernel starved of memory-level parallelism, not a
  coalescing bug).

**Both reviewers agree: no in-bound source mechanism has a credible path to ≥82%
span-inclusive on both M=16 and M=32. Reviewed NO-GO.**
