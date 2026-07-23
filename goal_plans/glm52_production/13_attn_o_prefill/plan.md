# Goal: optimize production packed attention O projection in prefill

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Optimize the attention O-projection prefill GEMM using the exact packed scale ABI,
local token bucket, and SGLang backend reached in production.  Do not treat the
f32-scale synthetic task as the production denominator.

## Reachability and workload

Trace the current SGLang call first.  The initial balanced DP8 test point is
M4096, N6144, K16384, but dtype/layout, TP/DP behavior, graph/eager mode, and actual
kernel must be recorded from a request.  Add a named
`linear_attn_o_prefill_m4096` serving-native workload if it is absent.

## Work

- [ ] Prove the live prefill call chain from `Fp8LinearMethod.apply` through the
  selected packed DeepGEMM symbol and record all tensor layouts.
- [ ] Add and structurally test the exact serving-native workload without changing
  the frozen `o_proj_prefill` task.
- [ ] Establish three paired production baselines, attention-layer baseline, and
  SGLang prefill baseline.
- [ ] Profile the compute-bound GEMM for tensor activity, issue stalls, scale/TMA
  traffic, tail waves, register pressure, spills, and epilogue stores.
- [ ] Sweep supported production configuration, then make isolated DeepGEMM/CuTe
  source changes only for profiler-backed headroom.  Keep JIT caches and import
  identities explicit.
- [ ] Preserve packed scales and output semantics; validate any additional prefill
  buckets only under separately named workloads.
- [ ] Promote only when paired microbenchmark, layer, and end-to-end prefill all
  improve with no enabled regression.

## Completion

Complete with a production win or a profiler-backed no-replacement result after an
exact workload and justified source attempt.  Old MFU or latency ceilings may be
reported as diagnostics but are not the primary gate.

## Deliverables

Include reachability evidence, new workload/tests, fork/build provenance, profiler
and variant ledger, paired results, correctness, layer/e2e validation, and policy.

