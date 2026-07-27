# Attempt ledger

All promotion decisions use the immutable campaign at
`hardened_runs/20260722T174049Z-immutable/`. It ran every alternating series and
all three profiler captures in one `with_flexible_gpu.sh` allocation on physical
GPU 0 (`GPU-30b619de-87f2-1862-0d07-a595da8fe417`). The validator recomputed raw
sample summaries, paired ratios, the 1.03x gate, dual-poison correctness replays,
source/JIT hashes, module origins, and the final repository allowlist. Its status
is `PASS`, with no stable region candidate.

`candidate.speedup` is the runner-recorded median of 60 interleaved per-pair
`reference_ms / candidate_ms` ratios. The p10/p90 values below are distributions
of those paired ratios. Marginal latency medians are descriptive and are never
divided to manufacture a speedup.

The original root-level fused-region campaign used a non-production FP8 `wq_b`
and generic RoPE. It is preserved and scoped in `SUPERSEDED_CAMPAIGN.md`; none of
its region, K-before-Q, or post-revert numbers drives the final decision. Its
isolated M4096/N160/K6144 BF16 projection rows remain valid because that workload
does not construct `wq_b` or RoPE. The later exact-BF16 campaign is useful
historical corroboration but remains provisional; the immutable campaign is the
authority below.

## Noise control and stock baselines

- Stock-vs-stock identity is 0.990699x isolated (paired p10/p90
  0.923719x/1.162011x) and 0.997625x for the fused prepare/store subregion
  (0.888315x/1.166066x), 60 pairs each.
- Three reference-only isolated medians are 0.031296, 0.026320, and 0.025888 ms;
  three region medians are 0.144048, 0.143616, and 0.175168 ms. These are baseline
  stability context, not candidate ratios.
- Promotion requires at least 1.03x in every repeated paired run. One favorable
  repetition or individual-pair tail inside the identity spread is noise.
- The immutable stock Nsys event is 1.085568 ms, 6.20x-7.56x the unprofiled
  region baselines. Operation identity, order, grids, streams, and matched
  instrumented ratios remain usable; absolute gaps and idle fraction do not.

## Attempt 1: direct SGLang CuTe-DSL TGV BF16 GEMM

- Hypothesis and baseline: the SM100 TGV tactic might improve the 128-CTA,
  0.865-wave M4096/N160/K6144 BF16 projection, whose stock NCU report shows
  255 registers/thread, 224,760 B shared memory/CTA, 8.84% achieved occupancy,
  and 499/648 attributed samples at a cooperative synchronization wait.
- Exact delta: external `indexer_wk_cutedsl_tgv.py` replaces only the
  `wk_weights_proj` BF16 backend. Stock BF16 `wq_b`, RoPE, Q/K kernels, cache
  layout, linear wrappers, and dual-stream schedule are unchanged.
- Expected effect: reduce the narrow projection duration without an adapter
  kernel, packing step, allocation, or stream change.
- Correctness: three isolated runs pass BF16 output comparison before and after
  timing on shared deterministic inputs. Three fused-region runs pass Q/gate
  comparison and byte-exact full-cache dual-poison replay before timing and on a
  fresh deterministic generation after timing.
- Isolated paired p50s: 0.329284x, 0.322368x, 0.361148x. The corresponding
  paired p10-p90 intervals are [0.248642, 0.434878], [0.253571, 0.453990], and
  [0.247364, 0.459946].
- Region paired p50s: 0.564382x, 0.562612x, 0.553539x. The paired p10-p90
  intervals are [0.454660, 0.631760], [0.441872, 0.700061], and
  [0.468212, 0.705410].
- Profiler delta: no TGV Nsys rerun was needed after decisive isolated and region
  regressions; the prior exact-BF16 trace already showed this is the same reached
  projection slot. This absence is not used to infer a bottleneck.
- Risk, decision, rollback: overriding SGLang's normal N>=1024 TGV guard would
  create a severe regression. Reject the external candidate; stock dispatch was
  never changed.

## Attempt 2: FlashInfer BF16 library sweep

- Hypothesis and baseline: an existing library tactic might beat stock for the
  same narrow BF16 GEMM before custom device code was justified.
- Exact delta: external `indexer_wk_flashinfer.py` selects FlashInfer auto,
  cuBLASLt, cuDNN, CUTLASS, or TGV on the isolated production-exact projection;
  there is no caller-ABI or weight-layout change.
- Expected effect: expose a better low-N tactic with no source integration.
- Correctness and paired distribution: the valid isolated campaign passes BF16
  output checks. Paired p50 [p10, p90] is auto 0.217030x [0.091555, 0.319777],
  cuBLASLt 0.109375x [0.065319, 0.149019], cuDNN 0.300118x
  [0.147770, 0.498953], CUTLASS 0.315262x [0.186480, 0.531981], and TGV
  0.194219x [0.134252, 0.276027].
- Profiler delta: not collected because every backend is already an isolated
  regression and the stronger direct SGLang TGV path also loses in the full
  corrected region.
- Risk, decision, rollback: adapter/tactic overhead cannot be recovered in the
  target subregion. Reject all variants; no production source changed.

## Attempt 3: direct ATen `torch.mm`

- Hypothesis and baseline: spelling the projection as direct MM might remove
  functional-linear dispatch overhead while reaching equivalent optimized BF16
  device math.
- Exact delta: external `indexer_wk_torch_mm.py` calls
  `torch.mm(x, weight.t())` through the same production `ReplicatedLinear`
  wrapper; all other region code and streams remain stock.
- Expected effect: lower host dispatch time without an extra CUDA operation.
- Correctness: three isolated runs pass BF16 output comparison before and after
  timing on shared deterministic inputs. Three fused-region runs pass pre-timing
  and fresh-seed post-timing Q/gate comparison plus full-cache write coverage.
- Isolated paired p50s: 0.984024x, 0.995630x, 0.996183x; paired p10-p90 intervals
  are [0.720803, 1.055990], [0.928251, 1.294621], and [0.879050, 1.073236].
- Region paired p50s: 1.003540x, 1.032630x, 1.002945x; paired p10-p90 intervals
  are [0.899364, 1.106195], [0.819939, 1.173850], and [0.796409, 1.229313].
- Matched profiler delta: the same four operation classes and dual-stream mapping
  remain. Candidate/stock ratios are 0.967773x projected span, 0.979158x host
  range, 0.992634x narrow-BF16 kernel duration, and 0.995549x CUDA-event time.
  These near-unity instrumented values do not override the unprofiled repeats.
- Risk, decision, rollback: the only >=1.03x repetition is contradicted by two
  neutral repetitions and broad identity-like tails. Reject; external candidate
  only, so rollback is the unchanged stock method.

## Attempt 4: exact stock-linear single-stream schedule

- Hypothesis and baseline: because the perturbed stock capture does not show
  kernel-on-kernel overlap, the method's existing `enable_dual_stream=False`
  branch might avoid coordination overhead.
- Exact delta: external `indexer_single_stream.py` selects only that branch while
  retaining the exact stock `ReplicatedLinear -> UnquantizedLinearMethod`
  sentinels. Unlike the preliminary adapter-contaminated experiment, this is a
  schedule-only control.
- Expected effect: remove reciprocal-stream coordination without changing any
  GEMM, Q/K kernel, layout, allocation, or output contract.
- Correctness: all three candidate series pass pre/post correctness and complete
  dual-poison cache replay.
- Region paired p50s: 1.012753x, 0.985414x, 0.978400x; paired p10-p90 intervals
  are [0.892789, 1.192308], [0.843897, 1.074260], and [0.825093, 1.170142].
- Matched profiler delta: all four kernels serialize on stream 7. The perturbed
  capture reports 0.979712 ms CUDA-event time, 501.918 us projected span, and
  1.349585 ms host range, versus stock 1.085568 ms, 532.223 us, and 1.445249 ms.
  Only the stream assignment is production-structural; profiler timing is not the
  promotion metric because both captures are over sixfold perturbed.
- Risk, decision, rollback: eliminating overlap can regress under different launch
  timing, and two unprofiled repeats already regress. Reject; keep the production
  default `enable_dual_stream=True`.

## Historical attempt 4a: preliminary single stream with adapter contamination

- Hypothesis and baseline: the same no-observed-overlap hypothesis motivated an
  early single-stream trial against corrected BF16-wq stock.
- Exact delta: `enable_dual_stream=False` was combined unintentionally with
  `_BackendLinearMethod -> functional.linear` in place of the stock
  `UnquantizedLinearMethod`; it was therefore not a schedule-only control.
- Expected effect: remove stream coordination, though the adapter change made host
  attribution ambiguous.
- Correctness: all three fused-region runs pass the campaign's pre-timing
  Q/gate/full-cache check; this historical runner did not yet perform the
  immutable fresh-seed post-check.
- Paired p50s are 0.974558x, 1.201739x, and 0.971418x; paired p10-p90 intervals
  are [0.901915, 1.183941], [0.899827, 1.733630], and [0.820538, 1.156332].
- Profiler delta: projected span grows 402.942 -> 757.310 us (1.879x) and host
  range grows 1072.645 -> 1581.976 us (1.475x). Those values are instrumented
  and adapter-contaminated, not production schedule attribution.
- Risk, decision, rollback: the lone favorable p50 has very wide tails and two
  repetitions regress. Reject as exploratory evidence; the exact immutable
  stock-linear control above supersedes it, and production never changed.

## Historical attempt 5: K-before-Q source schedule trial

- Hypothesis and baseline: in the original captured schedule, launching the short
  K/cache-store kernel before Q might expose more alternate/current-stream
  concurrency without changing device math.
- Exact delta: SGLang commit `a75a772a2` reordered K before Q inside the reached
  eager method. It did not change tensor layouts, kernels, cache semantics, or
  the final current-stream wait.
- Expected effect: reduce the captured stage-2 gap and projected span; there was
  no PTX/SASS change because only host launch order changed.
- Correctness: all three historical candidate rows passed that runner's
  pre-timing output/cache check, but the reconstruction used wrong FP8 `wq_b`
  and generic RoPE. This is not fixed-model correctness evidence.
- Historical paired p50s are 0.991256x, 1.114721x, and 1.007711x; paired p10-p90
  intervals are [0.757702, 1.289845], [0.811366, 1.473018], and
  [0.880246, 1.113967]. They are recorded for experiment honesty and excluded
  from the final disposition.
- Historical profiler delta: versus its matched wrong-ABI stock capture, kernel
  sum is 0.995668x while projected span is 1.137083x and host range 1.171039x;
  neither capture transfers to the fixed model.
- Risk, decision, rollback: launch reordering could destroy overlap, and the
  measurement is both unstable and wrong-ABI. Commit `2fbd443a1` reverts the
  trial; final `dsa_indexer.py` is byte-identical to stock.

## Containing and topology gates

No candidate reaches the necessary rank-local region gate, so none is enabled
for score/top-k, selected DSA attention, graph replay, or end-to-end acceptance.
The first all-GPU run was invalid TP4/DP1/EP1. A corrected TP4/DP4/EP4 allocation
then failed closed on package-origin provenance before CUDA server launch; commit
`95060f3` fixed the logical/canonical path comparison. A fresh corrected request
made 180 locked wrapper attempts, all returned 75 under shared-host contention,
and the diagnostic never executed. The controller record is
`tp4_live/20260722T181018Z-canonical_scheduler_blocker.json`. This is neither TP4
reachability nor TP8 acceptance evidence.

The required TP8/DP8/EP8 production gate needs eight B200s and is unavailable on
this four-GPU host. It is not weakened or relabeled. Stock fallback stays active,
so this external blocker cannot turn a rejected local candidate into a promotion.

## Final decision

No immutable candidate produces a repeat-stable >=1.03x improvement in the fused
prepare/store subregion. This supports no replacement at the validated rank-local
inner gate; it is not TP8 production acceptance. No registry entry, environment
switch, shape guard, or SGLang source change enables a candidate. The fixed source dependency graph is
two max-gated stages, `max(wq_b, wk)` then `max(Q,K)`; the immutable capture places
the targeted wk+K operations on the shorter overlap branches, while profiler
perturbation prevents an honest absolute launch-gap diagnosis. The historical
independent K GEMM does not transfer because production instead reaches the fused
BF16 N160 projection plus K normalize/RoPE/quant/cache store. Stock dual-stream
SGLang remains the production implementation and fallback for every bucket.
