# Goal: optimize fused indexer K and weights preparation in prefill

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Optimize the current CUDA indexer fusion around BF16 `wk_weights_proj` and K-cache
preparation during prefill.  Do not optimize the historical independent FP8 K
projection with a fake fixed physical row count.

## Production target

The current SGLang indexer constructs `wk_weights_proj: [6144 -> 160]`, splits the
result into key `[128]` and head weights `[32]`, and overlaps it with `wq_b` plus
fused Q/K preparation and cache store.  Decode sanity workloads already exist as
`indexer_wk_weights_decode_m16/m32`; the exact prefill region must be traced and a
new M4096 serving-native workload added if absent.

## Work

- [ ] Trace `Indexer._fused_q_prepare_and_store` on a real prefill request.  Record
  whether dual-stream execution is active and the exact scopes of
  `wk_weights_proj`, normalization/rotation, quantization, and cache store.
- [ ] Add a named production prefill workload for the exact callable or region,
  including BF16 weights and the actual output/cache contract.  Do not dispatch on
  old nominal M labels that generate identical tensors.
- [ ] Baseline both the isolated BF16 linear and the full overlapped indexer region.
  Capture stream timelines so a faster kernel that destroys overlap is visible.
- [ ] Profile GEMM shape, launch overhead, memory traffic, epilogue/split cost,
  cache-store kernels, waits, and alternate-stream critical path.
- [ ] Optimize the strongest measured target.  Allowed directions include
  SGLang fusion, a tuned BF16 linear backend, split/epilogue fusion, direct cache
  layout stores, or source edits in the reached kernel library.
- [ ] Verify the fused weight-loading contract, LoRA behavior if enabled in scope,
  numerical correctness, cache contents, graph/eager behavior, and no added stream
  synchronization.
- [ ] Accept only a full indexer-region and SGLang prefill improvement.

## Completion

Complete with a production win for the fixed prefill bucket, or a no-replacement
result that identifies the critical path and explains why the independent K GEMM
does not transfer to the fused implementation.

## Deliverables

Provide the live call/stream diagram, exact workload and tests, paired isolated and
region results, profiler artifacts, source diff/build record, correctness evidence,
and final integration or fallback decision.

