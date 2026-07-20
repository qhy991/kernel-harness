# Run Log — glm52 / dsa_attn_decode >40% HBM

Idle B200 GPU 0 (`CUDA_VISIBLE_DEVICES=0`). Harness `KDA_HARNESS_ROOT=/home/qinhaiyan/Kernel-Harness`.
Seed SHA-256 `9960b32616270535c916a2e110f5f57c0ee1dd05f0f5aa4e4be986be031533c4` (matches frozen seed).

## Round 0 — baseline + floor + NCU evidence

### task1 — baseline reproduction (AC-1)
- Copied frozen seed to `candidate/candidate.py`; sha256 matches.
- Gate: `run.sh --candidate $PWD/candidate` → CORRECT (exit 1), calc_diff 0 both shapes.

| M | candidate_us | reference_us | HBM util | correctness |
|---:|---:|---:|---:|---|
| 16 | 45.927 | 45.944 | 10.92% | PASS, calc_diff 0 |
| 32 | 47.151 | 47.264 | 21.27% | PASS, calc_diff 0 |

Log: `docs/baseline_repro.log`.

### task2 — single-kernel span floor + NCU (AC-3)
Standalone profiler `profiling/profile_dsa.py` builds the exact frozen inputs
(`glm52_ops.build_inputs('dsa_attn','decode',M,65536,seed=0)`) and calls stock
`flash_mla_sparse_fwd`. It only reads the reference; it never edits the harness.

- Main kernel: `void sm100::fwd::head64::sparse_attn_fwd_kernel<...>` — **exactly one
  launch per `run()` call** (nsys `docs/ncu/nsys_m16.nsys-rep`). The heavy radix-sort
  kernels in the trace are `torch.randperm` input construction, outside the timed window.
- CUDA-event span (warm-L2, my probe): M16 48.96 us, M32 50.29 us. The cold-L2
  CUPTI gate number (45.9/47.2 us) is lower because the gate protocol matters here:
  B200 L2 (~60 MB) caches most of the ~40 MB KV when warm.

NCU (cold-L2, `--set` sections; raw: `docs/ncu/ncu_m16.ncu-rep`, `ncu_m32.ncu-rep`,
text dumps `*_details.txt`):

| Metric | M16 | M32 |
|---|---:|---:|
| Duration (us) | 44.06 | 42.78 |
| **Grid Size** | **16** | **32** |
| Block Size | 384 | 384 |
| **Waves Per SM** | **0.11** | **0.22** |
| DRAM Throughput % | 15.20 | 15.52 |
| Compute (SM) % | 6.26 | 13.18 |
| Achieved Occupancy % | 17.67 | 17.78 |
| Registers/thread | 128 | 128 |
| Block Limit (registers / smem) | 1 / 1 | 1 / 1 |

### Diagnosis
`grid == M`: the kernel launches **one CTA per query token**, processing all 64 heads
in that block (MLA shares the single KV head, so KV rows are read once and reused
across heads — good bandwidth reuse, but only M CTAs). On 148 SMs that is
0.11–0.22 waves → **~89% of the GPU sits idle**, so DRAM throughput is pinned at ~15%.
Both shapes take ~the same wall time; M32 only looks 2× better because the byte model
credits it 2× the bytes at ~equal latency.

### Timing-metric fact governing the fix
`testbench/harness/timing.py:132` measures the **device-timeline span**
`max(kernel.end) − min(kernel.start)` over every kernel in one `run()` call — NOT a
sum of kernel durations. So any scheme that overlaps more concurrent work across the
idle SMs (split-KV in one launch, or multiple concurrent-stream launches) shrinks the
measured span.

### Conclusion (feeds task3)
Confirmed occupancy wall (consistent with `BL-20260716-mla-occupancy-wall`, and the
floor-first mandate of `BL-20260719-single-kernel-span-floor`). Mechanism =
**split-KV / flash-decoding** to raise resident CTAs from M to M×num_splits and lift
DRAM throughput toward the 40% bar. Implementation path chosen in task3.

### task3 — mechanism selection (analyze → Codex, AC-3/AC-4)
Routed to Codex (`gpt-5.5:xhigh`) with the full evidence
(`.humanize/skill/2026-07-19_16-32-36-*/output.md`). Ranking: **P1** (multi-stream
Python split, cheap probe) → **P2** (isolated FlashMLA fork, single-launch split-KV,
fused FP32 reduction; highest ceiling) → **P3** (task-local CuTe from scratch).
Recommended: run P1 as a one-round ceiling probe; if M16 doesn't land ≲13.5us, jump
to P2. Suggested splits S≈9 (M16), S≈5 (M32) targeting ~148 resident CTAs. Codex
flagged the exact risk that later killed P1: BF16 partial precision + host-gap span.

### task4 — first source attempt A1/P1 (AC-4 probe)
Implemented multi-stream split-KV wrapper (`attempts/A1_P1_multistream_split.py`):
split topk on 128-boundaries (kernel asserts topk % B_TOPK==0, B_TOPK=64 for
sm100/head64), S = round(148/M) → S=9 (M16), S=5 (M32); S concurrent
`flash_mla_sparse_fwd` on S streams; exact base-2 lse merge in FP32.

Result — **REJECTED on correctness**:
- `run.sh --candidate ... --M 16` → VERDICT INCORRECT (exit 2).
- Aggregate calc_diff 3.773e-6 PASSES (< 5e-6), but elementwise layer FAILS:
  elementwise_failed = 35,105; max_abs_err 1.95e-3 (abs_tol 2.85e-5);
  max_rel_err 105x (rel_tol 0.0157).
- Cause: stock kernel emits BF16 partials; merging pre-rounded partials perturbs
  small outputs ~1 bf16-ulp, which the single-pass FP32 reference avoids. Inherent
  to any Python-level split of this kernel (see
  `BL-20260719-bf16-partial-split-fails-elementwise`).
- Secondary: eager multi-stream device-timeline span is host-gap-dominated
  (probe ~1.9 ms), confirming a single-launch design is needed for timing too.

Decision: abandon Python-level split; the viable path is **A1/P2** — an isolated
FlashMLA fork doing single-launch split-KV with an in-kernel FP32 reduction before
one BF16 write (fixes precision AND host-gap). Active candidate restored to the
pristine seed (sha256 9960b326…, calc_diff 0). P2 build/feasibility is next round.

### task5 — split-heads probe A2 (AC-4 free-occupancy attempt)
Before committing to the P2 fork, probed the one mechanism that raises occupancy
without any KV split or partial merge: fold the 64-head axis into the query axis so
grid becomes M*G (`attempts/A2_splitheads.py`). Result — **BLOCKED**: the stock
sm100 head64 kernel dispatches only `h_q ∈ {64,128}` (tcgen05 MMA atom
`SM100_MMA_...<B_H=64>` requires M=64/128), so smaller head-groups cannot be formed.
No free-occupancy path exists.

### Round 1 — reachability floor + FINAL NO-GO
Built the pure-read reachability floor before spending a fork (floor-first mandate,
`BL-20260719-single-kernel-span-floor`). Scripts `profiling/{cupti_floor,gather_ceiling}.py`,
gate protocol (CUPTI device-span, cold-L2 253 MB flush), pinned idle B200. Full
derivation: `docs/floor_evidence.md`.

- Strict-40% ceilings: M16 < 12.534 us, M32 < 25.068 us (== 3.2 TB/s of model bytes).
- **Realistic gather floor** (pure read of per-query topk rows, max mem parallelism,
  no compute/combine/write — a lower bound on ANY correct kernel):
  **M16 12.42 us / 40.4%**, M32 20.94 us / 47.9%.
- M16: gather floor (12.42) ≈ ceiling (12.53) → **≈0 slack** for QK/softmax/AV +
  cross-split reduction. The only sub-ceiling read (10.88 us distinct-once) needs a
  pre-deduplicated row set a per-query kernel cannot form without materializing a
  gather. FP32 partial round-trip busts the byte budget (M16 +17 MB at S=4).
- M32: ~4 us headroom → reachable in principle, but the target requires BOTH shapes.

**USER DECISION (option 1): finalize evidence-backed reviewed NO-GO-CONFIRMED now.**
Do NOT build Triton. Do NOT build the FlashMLA/CUTLASS fork (A1/P2, A1/P3 pruned as
below-floor). M16 is the decisive blocking no-go. Kept candidate = pristine stock seed
(sha256 9960b32616270535c916a2e110f5f57c0ee1dd05f0f5aa4e4be986be031533c4, calc_diff 0).
Artifacts (results.md, attempt_dag.md, floor_evidence.md, profiling/, docs/ncu/)
committed. STOP.
