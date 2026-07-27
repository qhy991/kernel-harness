# Enable and fallback policy

No replacement is enabled.

- `SGLANG_GLM52_OPT=0` remains the reference and production-safe fallback.
- `--dsa-paged-mqa-logits-backend auto` continues to resolve to DeepGEMM on
  CUDA for M16 and M32.
- Normal decode retains the `next_n=1` split wrapper,
  `deep_gemm.fp8_paged_mqa_logits(clean_logits=False)`, and
  `topk_transform_512_v2`.
- The external CuTe-DSL candidates are benchmark artifacts only. SGLang does
  not import or dispatch to them.
- No M, ABI, graph/eager, world-size, PP, or request-class predicate enables a
  candidate.
- Unsupported and unvalidated cases therefore fail closed trivially: every
  case stays on stock.

A future enablement must be exact to operator × M × ABI × graph mode × topology
and must pass all of the following in one pinned deployment:

1. score/logit and exact top-k-set correctness;
2. graph replay with no timed adapter, allocation, copy, or host sync;
3. at least three same-GPU paired series, each at least `1.03x`;
4. non-regressing complete-indexer and selected-DSA regions;
5. a real checkpoint-backed SGLang decode improvement; and
6. TP8/DP8/EP8 rank-max correctness and latency.

The TP4 diagnostic cannot substitute for item 6.
