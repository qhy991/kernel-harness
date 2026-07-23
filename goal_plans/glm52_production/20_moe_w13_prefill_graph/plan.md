# Goal: optimize W13 prefill without graph or overlap regression

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Port only the useful parts of prior scale-pack/PDL ideas into the fused production
W13 prefill path, and prove that they improve the real graph/eager and DeepEP
overlap region.  Do not promote a graph-blind microbenchmark win.

## Production target

Start from local M4096, eight-rank normal DeepEP, 32 local experts, K6144, fused
N4096, packed production scales.  Add an exact
`moe_w13_grouped_prefill_m4096` workload if it does not yet exist.

## Work

- [ ] Trace the full prefill MoE path and determine whether W13 is captured,
  replayed, eager, overlapped, recipe-aware, or SM-partitioned in the target launch.
- [ ] Prove whether scale packing exists in production.  If inputs already satisfy
  the packed ABI, discard the old pack port and focus on the measured PDL/launch,
  scheduling, or handoff bottleneck.
- [ ] Baseline W13, normal DeepEP dispatch/combine, full MoE region, graph replay
  where applicable, and SGLang prefill.
- [ ] Profile scale/layout work, launch gaps, W13 kernel, SwiGLU handoff, signals,
  and overlap.  Separate candidate kernel time from critical-path time.
- [ ] Implement measured changes in SGLang, DeepGEMM, or DeepEP.  Preserve every
  recipe and overlap argument, and ensure graph replay reads current tensors rather
  than captured stale data.
- [ ] Require repeated paired compute gain plus non-regressing graph replay.  Then
  require full eight-rank MoE-region and end-to-end improvement.

## Completion

Complete with a production win or an evidence-backed no-replacement result.  If
eager improves but graph or the full region regresses, leave stock enabled and
report the failed promotion.

## Deliverables

Provide graph/overlap reachability, exact workload/tests, scale-path decision,
nsys/NCU evidence, source/build diff, paired compute and graph tables, full-region
and e2e results, and policy.

