# CPU source reachability audit

This is source-level evidence only. Runtime kernel-name proof is collected
separately under the GPU lock.

## Forced backend route

- The active class is `DeepseekSparseAttnBackend` in
  `python/sglang/srt/layers/attention/dsa_backend.py` (the plan's generic
  `DSAAttentionBackend` spelling is not the current class name).
- The explicit `--dsa-decode-backend flashmla_kv` value survives argument
  resolution. On Blackwell with FP8 KV, the no-flag default may resolve to
  TRT-LLM, so explicit selection is mandatory for this lane.
- `_forward_flashmla_kv` reshapes Q to `[M,1,local_heads,576]`, pads the head
  axis to 64 when needed, views KV as `[-1,64,1,656]`, turns the indexer's
  physical token slots into `[M,1,2048]`, and passes the precomputed metadata,
  split prefix, cache lengths, scale, and empty block table to
  `sgl_kernel.flash_mla.flash_mla_with_kvcache`.
- The Python wrapper selects
  `torch.ops.sgl_kernel.fwd_kvcache_mla.default` for BF16 Q plus sparse
  indices. `sgl-kernel/csrc/flashmla_extension.cc` registers that op to the
  CUDA implementation.

## Exact SM100 mapping

- QK head dimension 576 maps to FlashMLA model type V32.
- 64 Q heads map to
  `csrc/sm100/decode/head64/instantiations/v32.cu`.
- The complete region launches
  `flash_fwd_splitkv_mla_fp8_sparse_kernel*` followed by the generic decode
  combine kernel. Both launches are required in timing and profiling.
- The stock SM100 metadata path creates one scheduler part per physical SM for
  one-token decode. On a 148-SM B200 this means metadata shape `[148,8]`.
  With uniform top-k 2048, source arithmetic predicts eight splits/request at
  M16 and four splits/request at M32; runtime `num_splits` is the authority and
  is persisted in every result JSON.

## Workload representation

- KV uses page size 64 and SGLang's own `quantize_k_cache`, producing the packed
  656-byte FP8/scales/RoPE representation.
- Physical page zero is retained as the reserved dummy page. Sparse indices
  start at the next page and use a deterministic affine permutation within
  each request's 8192-token allocation.
- M is 16 or 32 locally and is never divided by DP. TP8/DP8 makes attention
  TP=1, so the production symbol sees all 64 Q heads on each DP rank.
- The absorbed Q tensor is 576 wide, but the caller preserves GLM-5.2's model
  attention scale `1/sqrt(192 + 64) = 0.0625`; using `1/sqrt(576)` would be an
  ABI/correctness error even though it does not change launch geometry.
- Setup-only quantization and scheduler metadata construction occur before the
  timed region. The timed reference is exactly the wrapper call, including its
  output/LSE allocation, main kernel, and combine kernel.
