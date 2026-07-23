# Goal: optimize the production GLM-5.2 FlashMLA KV decode path

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Deliver a production-compatible optimization of the exact DSA decode attention
backend observed in the B300 trace: `flashmla_kv`.  Optimize the complete measured
region containing `flash_fwd_splitkv_mla*` and its combine kernel.  Do not replace
this goal with the no-flag Blackwell TRT-LLM backend, `flashmla_sparse_fwd`, or the
old standalone W-UK/W-UV synthetic operators.

The old score and value operations are conceptual portions of this fused backend,
not independent SGLang dispatch points.  Changes to one portion are allowed, but
acceptance is based on the total FlashMLA region and complete SGLang decode path.

## Fixed production contract

- Force `--dsa-decode-backend flashmla_kv` for this lane and prove the branch at
  runtime from kernel names plus code mapping.
- Test local decode `M=16` and `M=32` as independent fixed buckets.  Do not divide
  either value by DP world size.
- Preserve the caller's paged FP8 KV cache, page size 64, sparse top-k indices,
  cache sequence lengths, Q-head padding, softmax scale, scheduler metadata,
  output shape/dtype, stream semantics, and CUDA Graph replay.
- Add separately named `serving_native` workloads
  `dsa_flashmla_kv_decode_m16` and `dsa_flashmla_kv_decode_m32`; do not rename or
  overwrite the existing TRT-LLM workloads.
- The production call chain is
  `DSAAttentionBackend._forward_flashmla_kv` ->
  `sgl_kernel.flash_mla.flash_mla_with_kvcache` ->
  `torch.ops.sgl_kernel.fwd_kvcache_mla`.

## Authorized source optimization

Source changes are explicitly encouraged when evidence supports them.  They may
touch SGLang integration, `sgl-kernel`, the pinned sgl-project/FlashMLA source,
CUTLASS/CuTe, build configuration, scheduler selection, or a narrowly scoped
custom CUDA kernel.  Because FlashMLA is currently fetched by CMake, make any
dependency change reproducible with a pinned fork, vendored patch/overlay, or
recorded source checkout; never patch an installed wheel or transient build tree
without also preserving the source delta and build provenance.

Keep the stock FlashMLA implementation available as the reference and fallback.
An optimization may be enabled for only M16 or only M32 when the other bucket does
not win.

## Required work

- [ ] Record the exact SGLang, sgl-kernel, FlashMLA, CUTLASS, CUDA, PyTorch, and
  built-extension identities.  Confirm which SM100 sparse-decode instantiation is
  reached for both fixed buckets and capture `num_splits` and scheduler metadata.
- [ ] Implement both exact serving-native workloads using tensors and metadata
  produced in the same representation as `_forward_flashmla_kv`.  Add structural
  tests and prove that the reference calls the production symbol.
- [ ] Capture repeated paired stock baselines for M16 and M32, the combined
  split-KV-plus-combine time, the containing DSA region, and complete SGLang decode.
- [ ] Use Nsight Systems to measure kernel ordering, launch gaps, split count, and
  main/combine balance.  Use Nsight Compute on the dominant kernel to quantify
  sparse gather and page access efficiency, L2/HBM traffic, tensor-core activity,
  register pressure, spills, occupancy, warp stalls, and tail waves.  Inspect
  ptxas output and SASS when the hypothesis depends on generated code.
- [ ] Rank and test one change at a time.  Candidate areas include M16/M32-specific
  tiles or persistent scheduling, fewer or better-balanced splits, sparse index
  loading, FP8 cache dequantization, Q-head padding overhead, softmax/reduction,
  main/combine fusion or specialization, and graph-safe temporary storage reuse.
- [ ] Validate exact numerical correctness, invalid-index handling, graph capture
  and replay, stream behavior, repeated alternating timing, the containing DSA
  region, and end-to-end throughput/latency.
- [ ] Maintain a per-bucket static oracle.  Dispatch must use existing shape or
  graph-bucket knowledge and may not introduce a host synchronization or timed
  adapter kernel.  Every unsupported or losing case must fail closed to stock.

## Completion

Finish with exactly one disposition from the shared rules.  A production win
requires at least one of M16 or M32 to improve paired p50 by at least 3%, with no
enabled-bucket regression and with a measurable containing-region and end-to-end
benefit.  Otherwise preserve profiler evidence and attempted source changes as a
no-replacement result and leave stock FlashMLA active.

## Deliverables

Commit the workload, source/build changes, tests, per-bucket dispatch policy,
raw paired benchmark outputs, profiler reports, experiment ledger, graph results,
containing-region and end-to-end comparisons, dependency/build provenance, and a
concise final report to this goal's isolated branches.  Do not push remote state.
