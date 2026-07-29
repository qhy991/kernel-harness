# GLM-5.2 W13 decode grouped-GEMM result

## Disposition

**No replacement.** Stock DeepGEMM remains the production default.

The exact same-source BM32 two-SM variant is correct, graph-safe and measurably
faster in the eager leaf, graph leaf and eager containing region. It fails the
mandatory graph-containing region contract: series 2's BA estimator is
1.028125x, below the required 1.03x. The plan makes any required per-series
estimator below 1.03x terminal, so pooled positive summaries cannot promote
the candidate and the remaining matrix was not run.

This is not an `external-acceptance-candidate`; unavailable TP8 hardware is not
the blocker. The local containing-region gate failed first.

## P0 fairness closure

Goal 19 timings were not inherited. They used PDL=false and compared an
installed stock package with a candidate overlay. Goal 24 instead built both
modules from DeepGEMM commit
`731e7c7a97d269e4b9f482ea18d0e709a948f293`, with one tracked candidate patch
and identical compiler/build plans.

The runtime deterministically binds stock before SGLang `compile_utils`, warms
only expected-M 4/5/8/9, freezes distinct caches, and restores the caller's
environment. On the leased B200, each of installed stock, same-source stock and
candidate was explicitly set and read back at PDL=true, `num_sms=148`,
`tc_util=100`. Mutate/readback/restore probes in both directions prove that the
side modules do not share runtime state.

The production integration and tracked materializer are committed locally in
SGLang revision
`1c671bf3a30360100e7947c87e0c873a387ad0be`.

The exact build manifest SHA256 is
`afa7063a860cd045138e51abcb5b8b44c226db13c08e212ae30da077f5655621`.
Its source, compiler, dependency, DSO, package, JIT and cache identities are in
[`build_manifest.json`](build_manifest.json) and the tracked reconstruction
inputs live in SGLang `third_party/deepgemm_w13/`.

## Correctness and production semantics

Both bounded variants pass the complete correctness suite:

- all independent expected-M points 4, 5, 8 and 9;
- zero/minimum/maximum/skewed expert counts;
- empty experts and 31/32/33 plus 127/128/129 boundaries;
- random, deterministic ramp, extreme finite, changed and poisoned data;
- eager and independently captured/replayed CUDA graphs;
- exact dtype, shape, stride, offset, packed-scale bytes and output ownership;
- exact stock `None` return, non-default stream and untouched masked regions.

Maximum absolute and relative error are both exactly zero for both candidates.
The graph observer mutates activation and device `masked_m`, re-poisons output,
proves pointer stability and deterministic replay, and contains no capture-time
D2H mask read.

The production wrapper trace selects each expected-M point exactly once,
returns exactly `None`, resets the private forward marker after every call,
selects stock without the marker, and propagates a candidate error without a
second stock launch. See
[`correctness.json`](correctness.json),
[`production_trace_bm32_1sm.json`](production_trace_bm32_1sm.json) and
[`production_trace_bm32_2sm.json`](production_trace_bm32_2sm.json).

The source and graph callsite proof is in
[`reachability_and_contract.md`](reachability_and_contract.md).

## Fair timing results

Every result contains three independent same-process series, ten alternating
AB/BA pairs per series, all raw samples and order, clock/UUID evidence, fresh
post-timing correctness and an auditor-recomputed four-estimator gate.

| Candidate and lane | Pooled | AB median | BA median | Order-balanced | Minimum required series estimate | Gate |
|---|---:|---:|---:|---:|---:|:---:|
| genuine one-SM, leaf eager | 1.031906 | 1.034595 | 1.029046 | 1.031817 | 1.026873 | fail |
| two-SM, leaf eager | 1.045878 | 1.043573 | 1.047735 | 1.045652 | 1.041952 | pass |
| two-SM, leaf graph | 1.042713 | 1.043134 | 1.044024 | 1.043579 | 1.042015 | pass |
| two-SM, region eager | 1.036626 | 1.040881 | 1.035545 | 1.038210 | 1.030817 | pass |
| two-SM, region graph | 1.036620 | 1.036411 | 1.035545 | 1.035978 | **1.028125** | **fail** |
| stock identity, region graph | 1.000000 | 1.000277 | 1.000000 | 1.000138 | 0.999447 | forced non-win |

The authoritative files are under [`results/`](results/). Each passes the
standalone serving-native result auditor. The first one-SM failure caused its
route to stop. The later two-SM graph-region failure is the campaign terminal
condition; expected-M 5/8/9 lanes were intentionally not used to average away
that failure.

## Code-generation and profiler result

The candidate names correspond to genuine generated implementations:

- BM32 one-SM uses `tcgen05.mma.cta_group::1`, plain `UTCQMMA`, 33
  registers/thread and no cooperative `UCGABAR`;
- stock and BM32 two-SM use `tcgen05.mma.cta_group::2`,
  `UTCQMMA.2CTA`, 36 registers/thread and cooperative cluster barriers;
- all have zero stack/local memory and no spills.

For expected-M4, full NCU replay reports 136.58 us stock versus 128.32 us
two-SM candidate, DRAM reads 840.8→818.7 MB and writes 31.45→10.08 MB. Both
launch 148 CTAs over 148 SMs in one wave. The dominant PCs remain the
persistent long-scoreboard sleep and `UCGABAR_WAIT`. This supports reduced
padded output/epilogue work and does not support a CLC rewrite.

The complete six-dimension analysis, exact PTX/SASS/cubin hashes, raw metrics,
collection logs and re-openable NCU reports are in
[`../../../profile/w13-bm32-stock-vs-2sm-em4-20260728/REPORT.md`](../../../profile/w13-bm32-stock-vs-2sm-em4-20260728/REPORT.md).

## Graph-containing attribution

The containing region is exactly:

```text
stock-or-candidate W13
  -> stock SwiGLU + packed quant
  -> stock W2
```

Stock and candidate are captured separately. The result requires equal event
and node counts and the same ordered non-W13 kernel sequence after substituting
only W13. Candidate hit/fallback/delegation counters are bound to capture and
warmup phases, and the selected hot path contains no allocation, adapter,
precompile, lock, file write or D2H synchronization.

The graph region's pooled 1.036620x is therefore a valid diagnostic but not a
win: the per-series BA failure is preserved in the raw pairs.

## Terminal matrix

| Requirement | Status |
|---|---|
| CPU fairness/lifecycle/graph/reproducibility contracts | pass |
| Same-source stock/candidate build and manifest | pass |
| PDL/SM/TC-util equality and state independence | pass |
| Exact ABI, correctness, return, stream and fallback | pass |
| Genuine one-SM code generation | pass, performance non-win |
| Genuine two-SM code generation | pass |
| Leaf eager and graph promotion | pass for two-SM |
| Containing-region eager | pass for two-SM |
| Containing-region graph, every estimator in every series >=1.03x | **fail** |
| Remaining M5/M8/M9 local lanes | stopped by terminal rule |
| TP4 diagnostic | not required after terminal local failure |
| TP8/DP8/EP8 checkpoint acceptance | not run; cannot rescue local failure |
| Production default | stock / candidate off |

The full hypothesis history is in
[`attempt_ledger.md`](attempt_ledger.md). Exact future TP8 commands are retained
in [`external_tp8_commands.md`](external_tp8_commands.md), with an explicit
warning that the current candidate is ineligible.

## Rollback

Leave `SGLANG_GLM52_W13_DECODE_VARIANT` unset. Startup then performs no W13 DSO
load, CUDA query or cache mutation and reports `default_off`. No artifact was
installed over stock, no remote state was modified, and no candidate is
enabled by the production-safe default profile.
