# GLM-5.2 算子 CUDA(B200) → AMD(MI300X) Baseline 后端对照表

本表把 `llm_flops/bench_glm5_prefill.py`（NVIDIA Hopper/B200，DeepGEMM + FlashMLA +
sgl_kernel）里 GLM-5.2 的 baseline 算子，逐个映射到 **AMD MI300X (CDNA3 / gfx942,
ROCm 7.0)** 上 sglang-ROCm 实际使用的后端（aiter / hipBLASLt）。本地评测保留原 split
MoE 诊断项，并额外加入 fused SGLang MoE total 作为 production ABI rollup。这是
`amd_bench_glm5_prefill.py` 与 A 卡 rewardbench 的算子契约来源。

> **重要结论**：`llm_flops` 仓库**没有 A 卡分支**（纯 NVIDIA Hopper，全 git 历史零 ROCm/HIP/aiter）。
> 「A 卡分支」指的是 **sglang 在 ROCm 上的算子实现**（走 aiter），已由 qhy 在 MI300X 上跑通
> （PyTorch 2.10/2.11+rocm7.0，AITER git HEAD）。本表据此重建。

> **E2E 对照（2026-07-22）**：TP=8 真实 prefill profile 显示 **AllReduce > MoE fused > dense MLA stage1 > FP8 GEMM**；
> 与本目录 TP1 campaign（`o_proj`/`index_k`/`dsa_attn` sparse）**优先级与 baseline 不对齐**。
> 详见 **`e2e_profile_20260722/README.md`** 与 `e2e_vs_harness_baseline_mismatch.csv`。

## harness baseline ↔ sglang 生产 dispatch 对照（2026-07-22 源码审计）

> **一句话**：`amd-glm-object.csv` 里的 `baseline(aiter us)` 与 `speedup vs aiter` 的分母，都是
> **aiter 的 Triton fallback 路径**，不是 sglang 在 gfx942 上默认 dispatch 的 kernel。数字本身没错，
> 只是"分母对象"需要注明清楚。

### GEMM (o_proj / index_k)

sglang 生产 dispatch — `sglang/srt/layers/quantization/fp8_utils.py::aiter_w8a8_block_fp8_linear`
（第 788 行）在 `_use_aiter_bpreshuffle_gfx942`（MI300X 上默认 True）时：

```
if M >= _GFX942_M_THRESHOLD (=4096) and (K,N) not in {(2048,4096)}:
    gemm_a8w8_blockscale_bpreshuffle_asm     ← ASM 路径（2.64× vs CK）
else:
    ck_gemm_a8w8_blockscale                  ← CK 路径
# aiter.ops.triton.gemm_a8w8_blockscale 只在 bpreshuffle 关闭时作为 fallback
```

对应到 CSV 三条 o_proj 行（`K=16384, N=6144`；均不在 exception set）：

| M | sglang 生产实际 dispatch | harness baseline | 差别 |
|---|---|---|---|
| 1024 | `ck_gemm_a8w8_blockscale` | `aiter.ops.triton.gemm_a8w8_blockscale` | 不同 kernel |
| 2048 | `ck_gemm_a8w8_blockscale` | 同上 (Triton) | 不同 kernel |
| 4096 | `gemm_a8w8_blockscale_bpreshuffle_asm` | 同上 (Triton) | 不同 kernel；ASM 快 2.64× |

`index_k` 三条：`_build_gemm` 让 prefill 的 `rows = S = 65536`，所以三个 CSV row 都是同一个
`[65536, 6144]×[6144, 128]` GEMM（这也是 CSV 里 index_k 三行延迟几乎相同的原因）。M=65536 ≥ 4096，
sglang 生产实际全部走 **ASM 路径**；harness baseline 仍是 Triton。

### MLA Decode (dsa_attn)

sglang 生产 dispatch — `sglang/srt/layers/attention/dsa_backend.py::_run_aiter_mla_decode_fwd`
（第 2051-2121 行）：

```python
from aiter.mla import mla_decode_fwd
mla_decode_fwd(q_kernel, kv_cache, o_kernel, cu_seqlens_q,
               kv_indptr, kv_indices, kv_last_page_lens, max_seqlen_q,
               sm_scale=layer.scaling, logit_cap=layer.logit_cap)
# 内部落到 aiter.mla_decode_stage1_asm_fwd（ASM），非 Triton
```

harness baseline — `aiter_baseline.py:85`：
```python
from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse
```

两者签名、算法、KV cache layout 都不一样：`mla_decode_fwd` 是 paged-KV + ASM stage1+reduce，
`pa_decode_sparse` 是 flat-KV + Triton split-K。**`grep -r pa_decode_sparse sglang/` 结果为空** ——
sglang 里没有任何代码路径会调用 `pa_decode_sparse`。

### 正确性 (`OK(calc_diff<5e-6)`)

`aiter_baseline.py:126` 的 oracle 是 `rocm_mi300x.py` 里的自包含参考：

| 算子族 | oracle | 缺失的 sglang 语义 |
|---|---|---|
| GEMM | `_blockwise_fp8_gemm_torch` (dequant→f32→matmul→bf16) | — (纯数学) |
| MLA | `_ref_mla` (gather + softmax + matmul) | `logit_cap`、sink、FP8 KV cache 量化、paged block table、head padding、多请求 batching |

**通过 calc_diff<5e-6 = "kernel 匹配 blockwise-FP8 GEMM / sparse MQA 的数学定义"**；
**≠ "kernel 与 sglang 生产 kernel 在同一输入上产生 bit-close 输出"**。这是刻意设计
（独立 oracle 能同时抓 baseline 和 candidate 各自的 bug），但读 CSV 时不能把它当作
"能直接替换进 sglang 服务"的通行证。

### 后续要复现"vs 生产 dispatch"分母，需要

1. **GEMM**：把 baseline 换成 `gemm_a8w8_blockscale_bpreshuffle_asm` (M≥4096) + `ck_gemm_a8w8_blockscale` (M<4096) 的分档测量；ASM 需要 aiter C++ JIT 构建到 aiter 包目录（本机只读，未跑通）。
2. **MLA**：把 baseline 换成 `aiter.mla.mla_decode_fwd`（配合 sglang `_run_aiter_mla_decode_fwd` 的 KV cache layout 与 `kv_indptr`/`kv_indices` 语义）。`pa_decode_sparse` 可保留为独立的 Triton 参考，但不再叫 "aiter baseline"。
3. **端到端 bit-close 正确性**：把 `run_glm52_no_offload.py` shim 起 sglang，在生产 kernel 的入口/出口 tap 出 Tensor，与 candidate 逐 shape 比对。
4. **配置 caveat**：现在 shim 里 `SGLANG_DISABLE_GFX942_BPRESHUFFLE=1`（省 5 GiB `weight_original`）会让 sglang 改走 Triton fallback；这条配置下 harness 的 Triton baseline 就 = 生产 dispatch。但这是"内存换速度"的降级路径，不是默认生产配置。

---

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

## 算子映射（prefill；GLM-5.2 shape：hidden=6144, q_lora=2048, kv_lora=512, heads=64, moe_inter=2048, topk=2048）

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
| 10 | `index_score` | mqa q[M,32,128]·k[S,128] | `deep_gemm.fp8_mqa_logits`（融合 fp8 MQA） | **`aiter.ops.triton.fp8_mqa_logits`**（SGLang HIP DSA indexer production kernel；weights 已折入 q_scale 和 softmax scale） | FP8 | compute |
| 11 | `moe_gate_proj` | grouped 8e [·,6144]×[6144,2048] | `deep_gemm.fp8_m_grouped_gemm_nt_masked` | **`aiter.fused_moe`**（QuantType.per_1x128, ActivationType.Silu）；per-op 粒度 fallback per-expert `torch._scaled_mm` 循环 | FP8→bf16 | compute (prefill) |
| 12 | `moe_up_proj` | grouped 8e [·,6144]×[6144,2048] | 同 gate（运行时与 gate 融合 w13 N=4096） | `aiter.fused_moe` / per-expert 循环 | FP8→bf16 | compute |
| 13 | `moe_down_proj` | grouped 8e [·,2048]×[2048,6144] | `deep_gemm.fp8_m_grouped_gemm_nt_masked` | `aiter.fused_moe` / per-expert 循环 | FP8→bf16 | compute |
| 14 | `moe_total` | fused gate/up + SiLU/gate + routed weight + down | runtime fused MoE path | **`sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe`** | FP8→bf16 | compute (prefill) / memory (decode) |

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
