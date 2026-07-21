# GLM-5.2 算子 CUDA(B200) → AMD(MI300X) Baseline 后端对照表

本表把 `llm_flops/bench_glm5_prefill.py`（NVIDIA Hopper/B200，DeepGEMM + FlashMLA +
sgl_kernel）里 GLM-5.2 的 13 个 baseline 算子，逐个映射到 **AMD MI300X (CDNA3 / gfx942,
ROCm 7.0)** 上 sglang-ROCm 实际使用的后端（aiter / hipBLASLt）。这是 `amd_bench_glm5_prefill.py`
与 A 卡 rewardbench 的算子契约来源。

> **重要结论**：`llm_flops` 仓库**没有 A 卡分支**（纯 NVIDIA Hopper，全 git 历史零 ROCm/HIP/aiter）。
> 「A 卡分支」指的是 **sglang 在 ROCm 上的算子实现**（走 aiter），已由 qhy 在 MI300X 上跑通
> （PyTorch 2.10/2.11+rocm7.0，AITER git HEAD）。本表据此重建。

## 硬件与数值基线（MI300X / CDNA3 / gfx942）

| 项 | 值 |
|---|---|
| 计算单元 | 304 CU，wavefront=64（对比 NVIDIA warp=32） |
| 显存 / 带宽 | 192 GB HBM3 / **5300 GB/s (5.3 TB/s)** |
| FP8(e4m3) 峰值 | **2614.9 TFLOPS**（dense；≈2× BF16） |
| BF16 峰值 | **1307.4 TFLOPS**（dense） |
| ridge point | FP8 = 2614.9e12/5.3e12 ≈ **493 FLOP/byte**；BF16 ≈ **247** |
| FP8 dtype | **`torch.float8_e4m3fnuz`**（gfx942 硬件格式），量化 `FP8_MAX = 224.0`（非 448/240） |
| 矩阵核 | MFMA（`__builtin_amdgcn_mfma_*`），非 NVIDIA tensor core / tcgen05 |
| 无 | TMA、TMEM、UE8M0 scale、PDL/GDC（这些是 Hopper/Blackwell 专有，A 卡无对应） |

> **FP8 ckpt 陷阱**：GLM-5.2-FP8 权重以 NVIDIA `e4m3fn` 存储，加载到 MI300X 需
> ① 把 `int8==-128`(fn 的 NaN) 置 0 后按 fnuz 位重解释；② 所有 `weight_scale_inv ×2`
> （fnuz 同位模式数值是 fn 的一半，指数偏置差异）。本 bench 用随机张量直接量化，不涉及此转换，
> 但复现真实权重路径时必须处理。

## 13 算子映射（prefill；GLM-5.2 shape：hidden=6144, q_lora=2048, kv_lora=512, heads=64, moe_inter=2048, topk=2048）

| # | 算子 | GLM-5.2 shape (M=prefill token) | CUDA/B200 后端 (llm_flops) | **AMD/MI300X 后端 (sglang-ROCm)** | dtype | roofline bound |
|---|---|---|---|---|---|---|
| 1 | `fused_qkv_a_proj` | [M,6144]×[6144,2624] | `deep_gemm.fp8_gemm_nt`（TMA/UE8M0） | **`aiter.gemm_a8w8_blockscale`**（CK blockscale）；fallback `torch._scaled_mm`(hipBLASLt) | FP8→bf16 | compute (大M) |
| 2 | `q_b_proj` | [M,2048]×[2048,16384] | `deep_gemm.fp8_gemm_nt` | `aiter.gemm_a8w8_blockscale` / hipBLASLt | FP8→bf16 | compute |
| 3 | `absorbed_W_UK` | bmm b=64 [M,192]×[192,512] | `sgl_kernel.bmm_fp8`（cuBLAS per-tensor） | **per-head `torch._scaled_mm` 循环**（hipBLASLt）；小 BMM `torch.bmm` 亦优 | FP8 per-tensor→bf16 | memory |
| 4 | `absorbed_W_UV` | bmm b=64 [M,512]×[512,256] | `sgl_kernel.bmm_fp8` | per-head `torch._scaled_mm` 循环 | FP8 per-tensor→bf16 | memory |
| 5 | `o_proj` | [M,16384]×[16384,6144] | `deep_gemm.fp8_gemm_nt` | `aiter.gemm_a8w8_blockscale` / hipBLASLt | FP8→bf16 | compute |
| 6 | `dsa_prefill_attn` | flash_mla_sparse s_q=M, s_kv=65536, topk=2048 | `sgl_kernel.flash_mla_sparse_fwd`（sm90 warp-spec sparse MLA） | **aiter MLA / DSA aiter backend**（`--dsa-prefill-backend aiter`）；fallback gather+SDPA(bf16) | bf16 | compute |
| 7 | `index_k_proj` | [S,6144]×[6144,128] | `deep_gemm.fp8_gemm_nt` | `aiter.gemm_a8w8_blockscale` / hipBLASLt | FP8→bf16 | memory (N=128 小) |
| 8 | `index_q_upproj` | [M,2048]×[2048,4096] | `deep_gemm.fp8_gemm_nt` | `aiter.gemm_a8w8_blockscale` / hipBLASLt | FP8→bf16 | compute |
| 9 | `index_weights_proj` | [M,6144]×[6144,32] | `deep_gemm.bf16_gemm_nt`（bf16→f32） | **`torch.mm`(bf16→f32)**（rocBLAS） | bf16→f32 | memory (N=32 极小, launch-bound) |
| 10 | `index_score` | mqa q[M,32,128]·k[S,128] | `deep_gemm.fp8_mqa_logits`（融合 fp8 MQA） | **per-head `torch._scaled_mm` + 加权和**（无专用 aiter kernel；qhy 报告此为 2 处已知 fail 之一，需 Triton/自实现） | FP8 | compute |
| 11 | `moe_gate_proj` | grouped 8e [·,6144]×[6144,2048] | `deep_gemm.fp8_m_grouped_gemm_nt_masked` | **`aiter.fused_moe`**（QuantType.per_1x128, ActivationType.Silu）；per-op 粒度 fallback per-expert `torch._scaled_mm` 循环 | FP8→bf16 | compute (prefill) |
| 12 | `moe_up_proj` | grouped 8e [·,6144]×[6144,2048] | 同 gate（运行时与 gate 融合 w13 N=4096） | `aiter.fused_moe` / per-expert 循环 | FP8→bf16 | compute |
| 13 | `moe_down_proj` | grouped 8e [·,2048]×[2048,6144] | `deep_gemm.fp8_m_grouped_gemm_nt_masked` | `aiter.fused_moe` / per-expert 循环 | FP8→bf16 | compute |

**decode 差异**：M=batch∈{1,4,8,16,32,64}；`dsa_prefill_attn`→`dsa_decode_attn`（`--dsa-decode-backend aiter`）；
`index_score` 用 paged 变体（`fp8_paged_mqa_logits` → paged fp8 KV 读主导，memory-bound）。decode 下
所有 dense GEMM 变 skinny-M weight-memory-bound（reward 自动切 HBM 带宽利用率）。

## aiter 精确 API（sglang-ROCm 生产路径，来自 qhy 已跑通脚本）

| 用途 | import | 调用签名 |
|---|---|---|
| dense FP8 GEMM (CK) | `from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale` | `gemm_a8w8_blockscale(x_fp8, w_fp8, x_scale, w_scale, dtype=torch.bfloat16)` |
| dense FP8 GEMM (ASM, 大M 2.64×) | `from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle_asm` + `shuffle_weight(w,(16,16))` | 需 bpreshuffle 权重 + x_scale 转置；guard `n%128==0 and k>=1024`；sglang 阈值 `_GFX942_M_THRESHOLD=256` |
| FP8 MoE grouped | `from aiter.fused_moe import fused_moe` + `from aiter import ActivationType, QuantType` | `fused_moe(x, w1, w2, topk_weights, topk_ids, activation=ActivationType.Silu, quant_type=QuantType.per_1x128, w1_scale=s1, w2_scale=s2)` |
| SiLU+Mul 融合 | `import aiter` | `aiter.silu_and_mul(o, x)`（o 预分配 [M,inter]，x=[M,2*inter]） |
| RMSNorm | `import aiter` | `aiter.rmsnorm(o, x, w, 1e-6)` |
| sglang dispatcher | `sglang/srt/layers/quantization/fp8_utils.py::aiter_w8a8_block_fp8_linear` | ASM/CK 按 M 阈值分派（大 M ASM，小 M CK） |

## Baseline 后端选择说明（本 bench 的双路策略）

- **默认（active）= torch-native ROCm 路径**：`torch._scaled_mm`（hipBLASLt FP8）、`torch.mm`、
  gather+SDPA。**一定可跑**，给出合法的 MI300X baseline 延迟，与算子 FLOP/字节模型一致。
- **aiter 路径（若 `import aiter` 成功自动启用）**：调 sglang-ROCm 生产 kernel（`gemm_a8w8_blockscale` 等），
  更贴近真实服务路径；CSV 的 `backend` 列记录每算子实际走了哪条路。
- **计时**：默认 hipGraph capture+replay（对齐 llm_flops/rewardbench 的 CUDA-graph 口径），
  对不可 capture 的 aiter kernel 自动回退到 HIP-event 计时（`AMD_BENCH_NO_GRAPH=1` 可强制 event）。

## 已知 A 卡实测锚点（qhy MI300X，单卡，torch.cuda.Event）

| 算子/形态 | CK (aiter) | ASM (aiter) | PyTorch bf16 | 备注 |
|---|---|---|---|---|
| dense GEMM M=4096 | ~330 TFLOPS | **869 TFLOPS (2.64×)** | — | ASM 仅大 M 赢；M=16 时 CK(26TF) > ASM(12TF) |
| dense GEMM 峰值(M=16384) | ~340 TFLOPS | ~869 TFLOPS | — | 距 FP8 峰值 2615 尚有大 headroom |
| SiLU+Mul (dim=12288, M=4096) | fused 0.065ms | — | 0.161ms | fused **2.46×** |
| Prefill MoE (bf16) | — | asm_moe 11.8 TF | torch_moe 1.4 TF | **8.22×** |
| MLA tiny BMM (decode) | deepgemm 0.4 TF | — | **torch.bmm 8.3 TF** | 小 BMM 用 torch.bmm 最优 |

> 这些是**优化目标锚点**，不是本 baseline bench 的输出。本 bench 的 baseline = 默认后端（CK/hipBLASLt）
> 在 GLM-5.2 精确 shape 上的 reward，写入 `amd_glm5_ops_{prefill,decode}_reward.csv`。
