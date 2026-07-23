# PR-history decisions for the serving-native GLM-5.2 suite

Queried on 2026-07-22:

- `sglang/glm5-glm51/README.en.md`
- `sglang/deepseek-v3-r1/README.en.md`

Decisions influenced by prior work:

- PR #22850: indexer projection/cache-store fusion means the new suite uses
  `wk_weights_proj` rather than standalone `index_k_proj` and
  `index_weights_proj` tasks.
- PR #20438: prior work overlaps indexer all-gather with query computation, so
  communication tasks report the maximum rank latency and keep overlap as an
  integration-level follow-up rather than treating rank-local kernel time as
  the serving result.
- PRs #14162/#21719/#22316: DeepEP communication dtype and low-latency dispatch
  are model/runtime-sensitive. The new decode task uses the actual FP8/UE8M0
  low-latency Buffer ABI; the prefill task uses normal dispatch/combine. The
  adjacent W13/W2 tasks use the EP8 packed layout instead of legacy separate
  gate/up projections.
- PRs #19329 and #27510: group registration and DP-attention/TBO affect the
  communication path. The AllGather reference therefore calls SGLang's
  `GroupCoordinator.all_gather_into_tensor`, not a synthetic copy kernel.
- GLM-5.2 deployment PRs #28437/#28448/#28460: fixed shapes are bound to the
  verified B200 TP8/DP8/DeepEP balanced lane.
