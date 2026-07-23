# Goal: optimize the fused DSA decode score path

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Optimize the QK/score side of the production `flashmla_kv` DSA decode kernels.
The former standalone FP8 `W_UK` batched matmul is a diagnostic decomposition, not
a SGLang replacement boundary.

## Production target

- add `dsa_flashmla_kv_decode_m16`
- add `dsa_flashmla_kv_decode_m32`
- `DSAAttentionBackend._forward_flashmla_kv` ->
  `sgl_kernel.flash_mla.flash_mla_with_kvcache`
- fixed context 8192 and sparse top-k 2048 in the current serving-native contract

The candidate must preserve score scaling, sparse page selection, softmax behavior,
KV cache addressing, output accuracy, and graph replay.

## Work

- [ ] Prove the explicit `flashmla_kv` backend selection and identify the
  split-KV/combination source sections that implement query/key score
  accumulation and reduction.
- [ ] Measure stock per-bucket latency, score-path contribution where separable,
  and complete DSA decode latency.
- [ ] Profile Q/K loads, tcgen05 use, reductions, register lifetime, occupancy,
  divergence, sparse index traffic, and tail effects.
- [ ] Try source changes only against the reached fused kernel.  Consider score
  tile geometry, query reuse, index prefetch, reduction scheduling, precision-safe
  fusion, and per-bucket generated variants when supported by evidence.
- [ ] Treat any isolated `absorbed_W_UK_decode` result as a lower-bound experiment;
  include its launch and dataflow cost before drawing a production conclusion.
- [ ] Validate M16 and M32 independently, then the full DSA region and SGLang
  decode.  Keep stock behavior for any losing bucket.

## Completion

Meet the shared production-win definition for at least one bucket, or document a
no-replacement result proving that the standalone opportunity disappears inside
the fused production kernel.

## Deliverables

Include fused-source mapping, profiler evidence, attempt ledger, source and build
diff, correctness, paired bucket results, static oracle, graph validation, DSA
region result, and end-to-end result.
