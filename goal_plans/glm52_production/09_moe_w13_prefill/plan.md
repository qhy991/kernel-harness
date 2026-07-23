# Goal: optimize fused production W13 prefill

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Optimize the fused gate+up W13 grouped GEMM in the current GLM-5.2 MoE prefill
pipeline.  Never deploy separate gate and up projection kernels in place of W13.

## Production target

- Normal DeepEP prefill with local M4096 and eight ranks.
- W13 shape family: 32 local experts, K6144, N4096, production packed scales.
- Containing region: DeepEP dispatch -> W13 -> SwiGLU+quant -> W2 -> combine.
- Current entry: `grouped_gemm_nt_f8f8bf16_masked`; live `masked_m`,
  `expected_m`, recipes, and overlap state must be captured before tuning.

## Work

- [ ] Prove that W13 is fused and record the real prefill input distribution after
  normal DeepEP dispatch.  Map any old gate/up tag to this one production call.
- [ ] Add `moe_w13_grouped_prefill_m4096` to `serving_native` if absent, using
  production packed tensors and the exact masked grouped ABI.
- [ ] Baseline W13, normal DeepEP communication, full MoE region, and end-to-end
  prefill with all replacement dispatch disabled.
- [ ] Profile tensor-pipe activity, expert/tail imbalance, tile waves, scale loads,
  TMA stalls, epilogue stores, SM reservation, and W13-to-SwiGLU handoff.
- [ ] Test low-risk configuration changes, then source-level DeepGEMM/CuTe changes
  justified by NCU.  Optimize the fused N4096 output, not one N2048 half.
- [ ] Preserve recipes, masked semantics, production scale ABI, output layout, and
  all stream/graph contracts.  Validate W13 output and the following SwiGLU+quant
  result.
- [ ] Accept only if the full eight-rank MoE region and SGLang prefill improve.

## Completion

Finish with a shared-definition production win or no-replacement disposition.  A
win in `moe_gate_proj_prefill` or `moe_up_proj_prefill` alone is insufficient
because those tasks do not model fused W13.

## Deliverables

Include fusion/reachability evidence, production workload and tests, source/build
provenance, NCU evidence, attempt ledger, paired W13 results, full-region and
end-to-end results, and final enable/fallback policy.

