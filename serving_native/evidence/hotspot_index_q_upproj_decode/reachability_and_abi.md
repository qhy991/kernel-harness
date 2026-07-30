# Reachability & frozen ABI — decode `index_q_upproj` (CPU audit)

## Production symbol and call path

- Module: `C4Indexer.wq_b`, defined in
  `sglang/python/sglang/srt/layers/attention/dsv4/indexer.py`:
  `self.wq_b = ReplicatedLinear(q_lora_rank, n_heads*head_dim, bias=False,
  quant_config=quant_config, params_dtype=torch.bfloat16, prefix="…wq_b")`.
- **Replicated**, not tensor-parallel-sharded — deliberately distinct from the
  attention `q_b_proj` (different shape and sharding). `prefix_to_op_name`
  (`glm52_opt/context.py`) maps `wq_b`, `q_up_proj`, `index_q_upproj →
  index_q_upproj`.
- Decode path: `C4Indexer.forward` → `attn_backend.forward_c4_indexer` →
  `_forward_prepare_normal`/`_multi_stream` → `C4Indexer.compute_q(q_lora, …)`
  → `q, _ = self.wq_b(q_lora)` → `Fp8LinearMethod.apply` (block_quant) →
  `op_context("index_q_upproj")` → `w8a8_block_fp8_linear` →
  `apply_w8a8_block_fp8_linear` (`fp8_utils.py`) →
  `try_dispatch_fp8_gemm(...)`.
- The candidate and stock share the same callsite, the same dynamic activation
  quantizer (`sglang_per_token_group_quant_fp8`, `column_major_scales=True`,
  `scale_tma_aligned=True`, `scale_ue8m0=DEEPGEMM_SCALE_UE8M0`), FP8 tensors,
  packed int32 UE8M0 scales, BF16 output, and CUDA-graph path. The only
  candidate-side change is `compiled_dims="nk"` in `deep_gemm.fp8_gemm_nt`.

## Frozen ABI (fail-closed selection)

| Field | Value |
|---|---|
| Phase / mode | decode, `ForwardMode.DECODE` only |
| Local M buckets | 16 and 32 (never divided by DP degree) |
| N (weight rows) | 4096 |
| K (contraction) | 2048 |
| Block size | 128 × 128 |
| Activation | contiguous CUDA `float8_e4m3fn`, `[M, 2048]` |
| Weight | contiguous CUDA `float8_e4m3fn`, `[4096, 2048]` |
| Activation scale | int32 packed UE8M0, shape `[M, 4]`, stride `[1, M]` |
| Weight scale | int32 packed UE8M0, shape `[4096, 4]`, stride `[1, 4096]` |
| Output | `bfloat16`, no bias |
| Device | one common CUDA device |

Verified on device: `DEEPGEMM_SCALE_UE8M0 = True`; activation scale observed as
`dtype=torch.int32 shape=[16,4] stride=[1,16]` (M16) and `[32,4] stride=[1,32]`
(M32). `_fixed_nk_abi_matches` rejects any deviation (wrong M/N/K, non-packed
scales, non-contiguous, bias, non-bf16 out, speculative/mixed/split-prefill/
target-verify modes) and returns the stock SGLang path before any launch.

## Registry entry (this worktree)

`registry._E2E_DECODE["index_q_upproj"]`: `implementation="fixed_nk"`,
`profiler_name="infini_kernel_glm52_index_q_upproj_decode_nk"`,
`m_values=(16,32)`, `n=4096`, `k=2048`, `graph_only=True`. Explicit-only
(`_E2E_EXPLICIT_OPS`): never enabled by `serving_safe` or an empty `OPT_OPS`.
