# Goal: optimize the production DSA decode value path

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Improve the value/output side of GLM-5.2 DSA decode as it is fused inside the
production `flashmla_kv` implementation.  The former standalone batched FP8
`W_UV` operation is not a deployment replacement point; it may be used only to
understand math or establish a component floor.

## Production target

- Add exact workloads `dsa_flashmla_kv_decode_m16` and
  `dsa_flashmla_kv_decode_m32`; do not relabel the existing TRT-LLM workloads.
- SGLang entry: `DSAAttentionBackend._forward_flashmla_kv` in
  `sglang.srt.layers.attention.dsa_backend`.
- Reached symbol: `sgl_kernel.flash_mla.flash_mla_with_kvcache`, including
  `flash_fwd_splitkv_mla*` and its combine kernel.
- Fixed decode context: 64 heads, head dimension 576, context 8192, sparse top-k
  2048, page size 64, and local batch 16 or 32.
- Reference: stock SGLang/FlashInfer with all GLM-5.2 replacement dispatch off.

## Work

- [ ] Prove that the selected deployment resolves `dsa_decode_backend` to
  `flashmla_kv` for both M buckets and record the exact sgl-kernel/FlashMLA
  package, source commit, and kernel variant.  Also record the explicit launch
  flag because the no-flag SM100 default can differ.
- [ ] Capture paired production-ABI baselines and an nsys timeline.  Attribute
  time to query preparation, KV loads, score/softmax work, value accumulation,
  output conversion, and launch/synchronization gaps as far as the fused kernel
  permits.
- [ ] Use NCU on the dominant kernel to determine whether the value side is bound
  by sparse KV traffic, tensor-core issue, reductions, occupancy, or output stores.
- [ ] Implement a source-level change in an isolated sgl-kernel/FlashMLA kernel
  checkout only when the evidence identifies value-path headroom.  Candidate
  directions include tile/layout changes, vectorized KV loads, reduced conversion
  work, pipeline overlap, or removing redundant stores.  Preserve numerics, sparse
  indexing, page semantics, and graph capture.
- [ ] Compare M16 and M32 independently.  Add only a winning variant to a static
  host-side oracle; retain the stock variant for the other bucket.
- [ ] Validate the complete DSA decode region and SGLang end-to-end decode, not the
  former standalone BMM.

## Completion

Complete only with a production win under the shared gate, or with an evidenced
no-replacement result showing why the fused kernel offers no deployable value-path
gain.  A speedup on `absorbed_W_UV_decode` alone is never sufficient.

## Deliverables

Provide the reachability trace, exact FlashMLA kernel identity, source diff and
build provenance, paired M16/M32 table, NCU/nsys evidence, correctness results,
bucket dispatch policy, DSA-region result, and end-to-end result.
