# Production reachability

## Callable chain

The current SGLang source and the production-shaped runtime trace agree on this
prefill path:

1. `DeepseekV2AttentionMLA` creates the attention output projection as a
   `RowParallelLinear`; `forward_mla.py` calls `self.o_proj(attn_bmm_output)`.
2. The FP8-loaded linear invokes `Fp8LinearMethod.apply`.
3. Its block-quantized branch calls
   `deepgemm_w8a8_block_fp8_linear_with_fallback`.
4. SGLang quantizes the BF16 activation per token group and emits column-major,
   TMA-aligned packed `int32` UE8M0 scales.
5. The default-safe `glm52_opt` lookup misses, so
   `w8a8_block_fp8_matmul_deepgemm` allocates the BF16 output and invokes the
   registered `deep_gemm_fp8_fp8_bf16_nt` custom op.
6. `deep_gemm_wrapper.gemm_nt_f8f8bf16` calls the installed
   `deep_gemm.fp8_gemm_nt` implementation.

The runtime trace records the native custom-op boundary as
`torch.ops.sglang.deep_gemm_fp8_fp8_bf16_nt`; Nsight Systems and Nsight Compute
then identify the reached device kernel as
`deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl`.

## Rank-local production ABI

| Tensor | Shape | Dtype | Stride | Notes |
|---|---|---|---|---|
| Caller activation | `[4096, 16384]` | BF16 | `[16384, 1]` | contiguous input to `Fp8LinearMethod.apply` |
| Quantized activation | `[4096, 16384]` | FP8 E4M3FN | `[16384, 1]` | produced by SGLang per-token quantization |
| Activation scale | `[4096, 32]` | `int32` | `[1, 4096]` | packed UE8M0, column-major/TMA-aligned |
| Checkpoint weight | `[6144, 16384]` | FP8 E4M3FN | `[16384, 1]` | contiguous |
| Checkpoint weight scale | `[6144, 32]` | `int32` | `[1, 6144]` | packed UE8M0, column-major/TMA-aligned |
| Output | `[4096, 6144]` | BF16 | `[6144, 1]` | newly allocated, contiguous |

The group size is `[128, 128]`. No float32 scale conversion, repack adapter, or
host synchronization occurs in the measured production path.

## Runtime state

The trace was captured with `SGLANG_GLM52_OPT=0`, profile `serving_safe`,
DeepGEMM PDL enabled, 148 visible SMs, current stream 0, eager execution, and no
active CUDA graph capture. It is a world-size-one reproduction of the exact
rank-local callable and ABI. The fixed production deployment remains TP8/DP8;
the trace does not claim to replace the missing eight-rank request.

The physical GPU for the baseline/profile bundle was GPU 1,
UUID `GPU-5b9be10b-5bfc-b658-9b31-f7ae8516dc54`, exposed as logical GPU 0 by
the flexible-GPU wrapper. The source-experiment bundle used GPU 2,
UUID `GPU-df8b1d78-b06c-39a2-54f0-66b9fabf3a99`, with every stock/candidate
comparison kept inside that one wrapper allocation. No performance comparison
mixes those two devices.

## Primary evidence

- `../production_20260723d/reachability_runtime.json`: callable identities,
  layouts, strides, dtype, graph state, stream, world size, and staged call
  observations.
- `../production_20260723d/reachability_runtime.log`: import/JIT diagnostics
  separated from the machine-readable trace.
- `../production_20260723d/dispatcher_hits.json`: exact explicit-dispatch hit
  key used only for the opt-in dispatcher experiment.
- `../production_20260723d/check_env.txt`, `../production_20260723d/topology.txt`,
  and `../production_20260723d/gpu_identity.csv`: environment and device
  provenance.
- `../../../workloads.py` and its self-test: the named
  `linear_attn_o_prefill_m4096` workload.
