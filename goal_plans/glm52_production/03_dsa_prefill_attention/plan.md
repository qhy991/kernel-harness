# Goal: optimize the DSA prefill backend actually selected by SGLang

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Identify and optimize the current production DSA prefill/extend attention path.
Do not assume the decode TRT-LLM backend or the historical sparse-attention task is
the production prefill implementation.

## Reachability gate

This goal has no valid optimization target until a real prefill request proves:

- the SGLang backend class and source symbol;
- whether the call is FlashInfer, FlashMLA, sgl-kernel, or another backend;
- local token count, sequence/context distribution, sparse top-k, page layout,
  dtype, workspace, graph/eager mode, CP/DP/TP topology, and stream behavior;
- the dominant GPU kernel names and their contribution to the prefill region.

Use local M4096 as the initial balanced DP8 test point.  If the traced request uses
a different important bucket, add a new name instead of changing that test point.

## Work

- [ ] Run a short production-config prefill trace and record the complete call
  chain.  Decide explicitly whether the frozen `dsa_prefill_attn` task is
  mathematically and operationally representative.
- [ ] Add an exact, named `serving_native` workload for the reached backend if one
  is missing; add structural tests and document every fixed parameter.
- [ ] Establish three paired production baselines plus full DSA-region and SGLang
  prefill baselines.
- [ ] Use nsys and NCU to classify compute, KV traffic, sparse gather, tail,
  reduction, launch, and overlap limits.
- [ ] Modify the reached open-source backend in an isolated source checkout.  A
  source attempt is required before declaring no replacement, but it must target a
  measured bottleneck rather than blindly porting a decode kernel.
- [ ] Validate correctness across representative sequence distributions, the fixed
  local M4096 bucket, any newly justified bucket, and the actual graph/eager mode.
- [ ] Promote only after the complete prefill DSA region and end-to-end prefill
  improve; otherwise retain the stock backend.

## Completion

Completion requires either a deployable production win or an evidence-backed
no-replacement result after reachability, an exact serving workload, profiling, and
at least one justified source attempt.  A 1.15x result against a mismatched
synthetic backend does not complete this goal.

## Deliverables

Provide the reachability decision, new workload and tests if needed, source/build
provenance, paired results, profiler evidence, correctness matrix, region and
end-to-end results, and final enable/fallback policy.

