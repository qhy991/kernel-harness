# Goal: optimize production indexer wq_b decode

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Optimize the DSA indexer's own `wq_b` projection at local M16 and/or M32.  Keep it
separate from attention `q_b_proj`: the output width, prefix, consuming kernels,
stream placement, and overlap region differ.

## Production target

| Bucket | Workload | Shape |
|---|---|---|
| M16 | `linear_indexer_wq_b_decode_m16` | M16, N4096, K2048 |
| M32 | `linear_indexer_wq_b_decode_m32` | M32, N4096, K2048 |

The projection runs in `Indexer._fused_q_prepare_and_store` and may overlap on an
alternate stream with BF16 `wk_weights_proj` and subsequent fused Q/K preparation.

## Work

- [ ] Trace both buckets, verify the `indexer.wq_b` prefix, production packed scale
  ABI, exact DeepGEMM config, alternate stream, and graph behavior.
- [ ] Baseline the serving-native GEMM and the full indexer preparation region.
  Capture an nsys stream timeline so overlap loss is part of the comparison.
- [ ] Evaluate the strongest existing skinny-M candidate only after adapting it to
  native packed scales and the N4096 production shape.  Do not copy the attention
  Q-B N16384 oracle.
- [ ] Profile launch cost, weight traffic, CTA waves, split/reduction traffic,
  TMA/scale handling, registers, and the wait position on the main stream.
- [ ] Implement source changes in SGLang, DeepGEMM, Triton, or CuTe as justified.
  A split-K design must include its reduction and scratch traffic and remain graph
  compatible.
- [ ] Select M16/M32 independently, then validate query layout, RoPE/quant consumer
  correctness, dual-stream overlap, full indexer latency, and end-to-end decode.

## Completion

Complete with a shared-definition production win or no-replacement disposition.
A fast standalone GEMM that lengthens the alternate-stream critical path or changes
query layout is rejected.

## Deliverables

Include prefix/reachability proof, stream timeline, native ABI source diff, profiler
evidence, paired bucket table, correctness, full-indexer/e2e results, and fallback
policy.

