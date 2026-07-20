# Results — GLM-5.2 MoE Gate Projection Prefill B200 MFU Campaign

Verdict basis: per-shape medians over authoritative `./run.sh` gate runs on idle
GPU 3. Harness exit 0 is necessary but not sufficient — the plan's bar is the hard
MFU thresholds (60% / 67% / 67%). See `run_log.md` for environment and raw baseline.

## Current status: exploration complete for the DeepGEMM direction; targets not yet met

| M | best correct MFU (this campaign) | target MFU | max latency | gap | status |
|---:|---:|---:|---:|---:|---|
| 1024 | 46.51% (seed/baseline) | ≥60% | 76.355 us | −13.5 pp | NOT MET |
| 2048 | 57.42% (seed/baseline) | ≥67% | 136.755 us | −9.6 pp | NOT MET |
| 4096 | 62.13% (seed/baseline) | ≥67% | 273.510 us | −5.0 pp | NOT MET |

No candidate variant found so far beats the seed by more than measurement noise, so
the best correct candidate remains the byte-identical seed (`sha256 3f96f1f1…73dd`).

## Direction 1 — DeepGEMM knob specialization (Milestone 3): exhausted, insufficient

Correctness-gated in-process sweep (`bench/sweep.py`), all variants calc_diff = 0
unless noted. MFU at each shape (best per knob):

| variant | M=1024 MFU | M=2048 MFU | M=4096 MFU | note |
|---|---:|---:|---:|---|
| baseline (default `compiled_dims='nk'`) | 46.66% | 57.31% | 62.03% | reference config |
| `compiled_dims='mnk'` | 46.70% | 57.38% | 62.05% | +0.0–0.1% (noise) |
| `compiled_dims='mn'/'m'/''` | 42.3% | 50.9% | 54.0% | **worse** (forces bad M path) |
| `set_pdl(True)` | 47.21% | 57.83% | 62.29% | +0.4–1.2%, best knob, still far short; median-only, stays "neutral" at the gate |
| `set_num_sms(<148)` | ≤46.2% | ≤57.0% | ≤61.2% | monotonically **worse** (compute-bound wants all SMs) |
| `set_tc_util` | already 100 (max) | — | — | no headroom |
| `recipe=(1,128,128)` and split `recipe_a=(1,128)`,`recipe_b=(128,128)` | 46.65% | 57.36% | 61.93% | correct (calc_diff 0); **== default** (auto-detected), within noise (0.999x) |
| `recipe=(1,1,128)` / `recipe=(128,128,128)` (mismatched) | **error** | error | error | recipe is fixed by the non-re-quantizable `(1,128,128)` scale layout — a different granularity misreads the frozen scales |
| `disable_ue8m0_cast=True` | **error** (gemm.hpp:295) | error | error | f32 scales are not in the packed layout it asserts |

**Finding:** the DeepGEMM masked path is already near-optimally configured. The best
knob (`set_pdl`) yields ≤1.2% and does not clear the harness win margin, let alone the
MFU thresholds. The `recipe` knob offers **no tuning freedom**: the only correct recipe is
the one matching the frozen `(1,128,128)` scale granularity (identical to the default
auto-detection, within noise), and every mismatched recipe errors because the scales cannot
be re-quantized. Every threshold needs 1.08–1.29x; knobs deliver ~1.01x.

## Direction 2 — reduce per-call preprocessing overhead: not reachable via the API

The masked call spends ~8–20% of its latency in preprocessing (scale-pack + grouped
index/schedule prep; see `run_log.md` breakdown). Attempts to remove it:

- `disable_ue8m0_cast=True` with per-expert `get_mn_major_tma_aligned_packed_ue8m0_tensor`
  pre-packing → **error** (masked API rejects the per-expert-stacked packed layout).
- Pre-transform scales with `get_mn_major_tma_aligned_tensor` then default cast →
  **correct but 3–5x slower** (Python per-expert loop overhead + DeepGEMM still runs
  its internal pack). Measured: M=1024 425 us, M=4096 544 us vs 98 / 295 us baseline.

The preprocessing is internal to `deep_gemm`'s `_C` masked call and is not separable
via the Python API without re-implementing the kernel. NOTE: the faster
**packed-ue8m0 production path** is explicitly forbidden by the plan as a "win"
(it changes the frozen f32-block-scale problem, not the kernel), and is used here for
evidence only.

## Direction 3 — contiguous grouped-GEMM entry point (Codex's flagged lead): correct but far slower

`m_grouped_fp8_gemm_nt_contiguous` is DeepGEMM's natural prefill layout (vs the masked
static layout). Probe (`bench/contiguous_probe.py`): gather each expert's valid rows
into a 128-aligned contiguous buffer + build `m_indices` + contiguous GEMM + scatter
back, **all inside the timed window**, f32-scale numerics preserved (calc_diff = 0):

| M | reference (f32 masked) | contiguous (gather+GEMM+scatter) | speedup | MFU |
|---:|---:|---:|---:|---:|
| 1024 | 98.37 us / 46.57% | 580.58 us | 0.17x | 7.9% |
| 2048 | 159.78 us / 57.35% | 672.21 us | 0.24x | 13.6% |
| 4096 | 295.41 us / 62.03% | 844.55 us | 0.35x | 21.7% |

**Finding:** correct but a 3–6x regression. The timed compaction dominates, and the
contiguous kernel STILL runs its own internal f32→ue8m0 scale-pack — so it does not even
remove that tax. Even an optimal fused gather (≈15 us) + scatter (≈15 us) would add ≥30 us
on top of the same scale-pack, exceeding the masked path's entire ~8–27 us preprocessing.
This is exactly the "expensive timed compaction" Codex flagged as disqualifying. The
contiguous entry point is closed on both empirical and analytical grounds.

## Independent analysis (Codex gpt-5.5:xhigh, task3/task6 `analyze` routing)

Codex, given the same evidence, independently concluded:
- **Binding bounds**: M=1024 & M=2048 → *tensor-core pipe rate of the main GEMM* (below
  target even with zero preprocessing); M=4096 → *fixed preprocessing overhead* (main-GEMM
  -only ~67.7% ≥ 67%).
- **No specific fixable DeepGEMM limitation** is named by the evidence — valid-tile
  scheduling, tail-wave, occupancy, and memory bandwidth are all ruled out; the main GEMM
  is "at or near the practical ceiling for this problem class." A bespoke SM100 kernel is
  **not justified** under DEC-1.
- **TARGET MET (all three) is not physically supported** on the frozen f32-block-scale
  path; a per-shape evidence-backed NO-GO with the best correct seed preserved is the
  correct terminal deliverable per DEC-2/AC-6.
- Flagged the contiguous entry point as the one thing to check → tested above, closed.

## Binding bound per shape (AC-6) — differential NCU + timeline evidence

The MFU deficit decomposes into two named, evidence-backed bounds:

1. **Fixed per-call preprocessing overhead** (scale-pack `transpose_and_pack_fp32_into_ue8m0`
   ×2 + torch `scatter_gather`/`div_floor`/`arange` schedule prep): ~19.5 us at M=1024
   (~20%), amortizing to ~8% at M=4096. Internal to DeepGEMM; not removable via the API.

2. **Tensor-core-pipe-bound main GEMM.** The core `sm100_fp8_fp4_gemm_1d1d_impl` runs at
   72.3% TC-pipe utilization (top pipeline), 65% memory throughput, 12.5% occupancy
   (shared-memory limited by design), persistent grid=148, BLOCK 128³, 2-CTA cluster —
   a textbook-optimal SM100 schedule. Its **standalone FLOP MFU is ~57.8% / 65.4% / 67.7%**
   (M=1024/2048/4096), i.e. below the 60/67/67% targets on the two smaller shapes **even
   with zero preprocessing overhead**. The ~28% gap from TC-pipe-busy to FLOP-peak is the
   intrinsic cost of blockwise-f32-scale application + mainloop TMA/MMA bubbles, which any
   FP8 blockwise-scaled kernel on this data pays.

### Physical-reachability read (evidence-based)

| M | main-GEMM-only latency | main-GEMM-only MFU | target latency | reachable by removing preprocessing alone? |
|---:|---:|---:|---:|---|
| 1024 | ~79 us | ~57.8% | ≤76.355 us | **No** — even zero-overhead main GEMM (79 us, 57.8%) misses both the latency ceiling and the 60% floor |
| 2048 | ~140 us | ~65.4% | ≤136.755 us | **No** — 140 us > 136.755 us; 65.4% < 67% |
| 4096 | ~271 us | ~67.7% | ≤273.510 us | **Borderline yes** — but only if a bespoke fused kernel both matches DeepGEMM's main GEMM AND removes the ~8% overhead |

Consequence: on the two smaller shapes the target sits **above** DeepGEMM's demonstrated
core-GEMM ceiling, so TARGET MET (all three) would require a bespoke kernel that
*outperforms* DeepGEMM's SOTA tcgen05 masked GEMM by 1.05–1.14x, not merely removes
overhead. The differential NCU does **not** name a specific fixable DeepGEMM limitation
(no padded-tile waste — it walks valid tiles; no bad schedule — it is persistent/2-CTA/
optimal blocks), which is the escalation trigger the plan requires.

## Candidate DAG / attempt ledger

| id | candidate | correct | M=1024 | M=2048 | M=4096 | disposition |
|---|---|---|---:|---:|---:|---|
| c0 | seed (verbatim reference) `3f96f1f1…73dd` | yes | 46.51% | 57.42% | 62.13% | **best correct candidate (preserved)** |
| e1 | `compiled_dims='mnk'` | yes | 46.70% | 57.38% | 62.05% | rejected (≡ baseline within noise) |
| e2 | `set_pdl(True)` | yes | 47.21% | 57.83% | 62.29% | rejected (≤1.2%, stays neutral at gate) |
| e2b | `recipe=(1,128,128)` / split `recipe_a`,`recipe_b` | yes | 46.65% | 57.36% | 61.93% | rejected (≡ default within noise; recipe fixed by scale layout) |
| e2c | `recipe=(1,1,128)` / `(128,128,128)` (mismatched) | error | — | — | — | rejected (recipe must match non-re-quantizable scales) |
| e3 | `set_num_sms<148`, `compiled_dims∈{mn,m,''}` | yes | worse | worse | worse | rejected (regressions) |
| e4 | `disable_ue8m0_cast=True` (+packed scales) | error | — | — | — | rejected (API assertion) |
| e5 | pre-`get_mn_major_tma_aligned_tensor` + default cast | yes | 0.23x | 0.35x | 0.54x | rejected (huge slowdown) |
| e6 | contiguous grouped GEMM + timed gather/scatter | yes | 0.17x | 0.24x | 0.35x | rejected (3–6x regression; timed compaction) |

## Correctness + statelessness (AC-2, AC-3)

`bench/verify.py --candidate candidate/candidate.py` → ALL PASS on M∈{1024,2048,4096}:
pre-timing correctness on poisoned buffer, input immutability (data_ptr + **byte-exact**
per-tensor equality, dtype-agnostic uint8 view — not a collision-prone scalar sum), no lazy
output, post-timing recheck on fresh inputs, and cross-workload re-entrancy across two
different M. calc_diff 0. `bench/verify.py --selftest` confirms a sum-preserving padding-row
mutation is CAUGHT (correctness on valid rows unaffected), proving the immutability check.

## Final confirmation (AC-5) — three authoritative gate runs, per-shape medians

`./run.sh --candidate candidate/candidate.py` ×3 on idle GPU 3 (raw JSON preserved:
`docs/profiles/gate_run_{1,2,3}.json`). Dispatch path on every shape = **DeepGEMM masked
(direct `fp8_m_grouped_gemm_nt_masked`)** — the judged candidate is the byte-identical seed,
so no `glm52_ops.reference` fallback branch is used.

| M | run1 us | run2 us | run3 us | median us | run1/2/3 MFU | median MFU | target | threshold | classification | dispatch |
|---:|---:|---:|---:|---:|---|---:|---|---|---|---|
| 1024 | 98.62 | 98.61 | 98.47 | 98.61 | 46.46/46.46/46.52% | **46.46%** | ≥60% / ≤76.355us | **MISS** | neutral (all 3) | DeepGEMM masked (direct) |
| 2048 | 159.75 | 159.90 | 159.85 | 159.85 | 57.35/57.30/57.32% | **57.32%** | ≥67% / ≤136.755us | **MISS** | neutral (all 3) | DeepGEMM masked (direct) |
| 4096 | 295.71 | 296.01 | 295.92 | 295.92 | 61.97/61.91/61.93% | **61.93%** | ≥67% / ≤273.510us | **MISS** | neutral (all 3) | DeepGEMM masked (direct) |

Harness exit code 1 on every run (correct on all shapes; performance gate not met — the
seed only matches the reference, so no shape WINS and none REGRESSES). `TARGET MET` = **false**
(no shape clears its threshold; the requirement is all three). No shape is a WIN or a PARTIAL
WIN. Medians are stable to <0.1% across the three runs.

## Verdict: evidence-backed NO-GO — targets physically unreachable on the frozen f32-block-scale path

`TARGET MET` requires all three of AC-4.1/4.2/4.3. The evidence shows this is not
physically reachable on the frozen `deep_gemm.fp8_m_grouped_gemm_nt_masked` f32-block-scale
path, and no NCU-named fixable DeepGEMM limitation exists to authorize a bespoke SM100
kernel (DEC-1 trigger unmet). Per-shape named bounds:

- **M=1024 — bound: tensor-core pipe rate of the main GEMM.** Its own NCU profile
  (`docs/profiles/ncu_maingemm_m1024`): TC top pipeline **72.3%** (lowest of the three —
  small-per-expert-M wave/tail effect), memory 64.2%, DRAM 30.6%, occupancy 12.5%, dominant
  stall MMA/smem scoreboard 56.1%. The main GEMM alone (nsys median at boost clocks, 60 iters)
  is **81.5 us / 56.4% MFU** — above the 76.355 us ceiling AND below the 60% floor. So deleting
  100% of the ~20% preprocessing tax still misses on both. (An NCU window showed a
  non-reproducible 75.5 us / 60.7% at a transiently higher clock; the gate runs at the boost
  clocks where the main GEMM is 81.5 us.) Reaching 60% needs DeepGEMM's own SOTA tcgen05 main
  GEMM to get ~1.07x faster — a mainloop win with no evidence of a defect to exploit.
- **M=2048 — bound: tensor-core pipe rate of the main GEMM.** Its own NCU profile
  (`docs/profiles/ncu_maingemm_m2048`): TC top pipeline **81.0%**, memory 74.1%, DRAM 24.7%,
  occupancy 12.5%, dominant stall MMA/smem scoreboard 55.9% — well-pipelined, compute-bound,
  no defect. Main-GEMM-only ~140 us > the 136.755 us ceiling and ~65.4% < 67%; overhead
  removal alone cannot reach it.
- **M=4096 — bound: fixed preprocessing overhead.** Its own NCU profile
  (`docs/profiles/ncu_maingemm_m4096`): TC top pipeline **87.7%** (near ceiling), memory 73.9%,
  DRAM 20.5%, occupancy 12.5%. Main-GEMM-only ~271 us / ~67.7% is *marginally* at target, but
  the ~8% scale-pack + schedule-prep tax (internal to DeepGEMM, not removable via the API;
  contiguous re-layout is 3–6x slower) keeps the delivered total at 295 us / 62.1%. Reaching
  target here requires a bespoke fused kernel that both matches DeepGEMM's main GEMM AND
  removes the overhead — and would still leave M=1024/M=2048 short, so it cannot deliver
  `TARGET MET`.

**Best correct candidate preserved:** the byte-identical seed
(`candidate/candidate.py`, sha256 `3f96f1f1cfc3ae1211d1850bce935eb6751ad4282aff4da679af1bf30faa73dd`),
correct + stateless + non-gaming on all shapes (calc_diff 0), three-run median MFU
46.5% / 57.4% / 62.1%.

This satisfies the plan Lower Bound and AC-6 (evidence-backed, per-shape named-bound no-go,
DeepGEMM direction attempted and profiled, best correct candidate preserved). Escalation to
a bespoke kernel is **not triggered by condition**: DEC-1 gates it on an NCU-named fixable
DeepGEMM limitation, and the per-shape profiles name none (TC-bound, well-pipelined,
compute-bound, optimally scheduled on every shape). Per the reachability analysis it could at
best win the single borderline shape M=4096 while M=1024/M=2048 remain below target, so it
cannot achieve `TARGET MET`. Codex (`analyze`) independently concurs on both the per-shape
bounds and the not-triggered decision.

**Scope of this conclusion (deliberately not overclaimed):** this is not an absolute proof
that no future hand-written kernel could ever beat DeepGEMM's mainloop. The defensible and
sufficient conclusion is narrower: on this frozen f32-block-scale contract, the DeepGEMM
masked path and every API-accessible variant cannot reach target, the differential profiles
name no fixable DeepGEMM limitation, and therefore a bespoke SM100 escalation is not
authorized by the plan (DEC-1). If a future profiling pass ever names a concrete mainloop
defect worth ≥1.05x, the goal-tracker records that as the reconsideration trigger.
