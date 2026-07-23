# Goal: optimize production MoE W2 prefill after DeepEP dispatch

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Optimize the W2 grouped GEMM reached during GLM-5.2 prefill with normal DeepEP,
using the actual EP8 local-expert distribution and packed production ABI.  Do not
reuse the frozen E=8 synthetic capacity model as the deployment contract.

## Production target

- Initial prefill point: local M4096 on every DP8 rank.
- Communication: `deepep_normal_dispatch_prefill` and
  `deepep_normal_combine_prefill` on eight ranks.
- Compute entry: production `grouped_gemm_nt_f8f8bf16_masked` for W2 after
  `_varlen_deep_gemm_silu_mul_quant`.
- Geometry to capture live: 32 local experts, `masked_m`, `expected_m`, compact or
  padded receive layout, recipes, SM budget, and overlap mode.

## Work

- [ ] Trace a real prefill request from normal DeepEP dispatch through W13,
  SwiGLU+quant, W2, and combine.  Record all W2 tensors and overlap arguments.
- [ ] Add an exact `moe_w2_grouped_prefill_m4096` serving-native workload if one is
  missing.  It must construct inputs from the production dispatch contract rather
  than the frozen synthetic builder.
- [ ] Establish three paired W2 baselines, eight-rank dispatch/combine baselines,
  a full MoE-region baseline, and SGLang prefill baseline.
- [ ] Profile tail distribution, mask scheduling, tensor utilization, TMA/scale
  traffic, SwiGLU handoff, SM partitioning, and communication/compute overlap.
- [ ] First sweep production-visible DeepGEMM controls.  Escalate to an isolated
  DeepGEMM/CuTe/CUDA source change only for a measured device-kernel limit.
- [ ] Preserve packed scales, recipes, masked rows, signals, streams, and graph or
  eager semantics.  Do not insert any f32 scale conversion to reproduce a synthetic
  score.
- [ ] Promote only when the paired compute gain survives the full eight-rank MoE
  region and SGLang prefill.  Otherwise keep stock W2.

## Completion

Complete with a production prefill win or with an evidence-backed no-replacement
result after the exact workload and at least one justified optimization attempt
exist.  Historical 65% MFU is diagnostic context, not the deployment gate.

## Deliverables

Provide the live shape/ABI capture, new workload and tests, profiler files, attempt
ledger, source/build record, paired W2 table, DeepEP/full-region table, end-to-end
table, and final policy.

