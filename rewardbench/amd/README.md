# AMD MI300X GLM-5.2 算子 Reward Bench

GLM-5.2 prefill / decode 各 13 个算子的**性能 reward 评测**，目标硬件 **AMD Instinct
MI300X (CDNA3 / gfx942, ROCm 7.0)**。是 kernel-harness `rewardbench/`（B200/SM100 + DeepGEMM）
的 A 卡移植：算子清单、reward 公式、CSV schema 与 B200 版一一对应，只把 roofline 峰值换成
MI300X、后端 kernel 换成 sglang-ROCm（aiter / hipBLASLt）。

reward = **bound-aware roofline 利用率 ∈ [0,1]** = 实测吞吐 ÷ roofline 天花板
（compute-bound → 矩阵核利用率；memory-bound → HBM 带宽利用率，按 arithmetic intensity 自动判定）。
指标设计见 `operator_mapping.md`（算子对照表）与 kernel-harness 的 `../GLM5_ops_reward_design.md`。

## 文件

| 文件 | 说明 |
|---|---|
| `amd_glm5_ops_common.py` | 共享引擎（cost 模型 / AMD kernel builders / roofline reward / 计时 / driver） |
| `amd_bench_glm5_prefill.py` / `amd_bench_glm5_decode.py` | 逐算子延时+reward benchmark（13 算子，含 backend/TFLOPS/GB/s/AI/reward） |
| `bench_AMD_GLM5_ops_prefill.py` / `bench_AMD_GLM5_ops_decode.py` | **reward-bench 入口**（=baseline 模式，产 reward 分母 CSV） |
| `amd_glm5_ops_prefill_reward.csv` / `amd_glm5_ops_decode_reward.csv` | **baseline 测试结果**：13 个参考算子在 MI300X 上的 reward（reward 基准） |
| `operator_mapping.md` | CUDA(B200)→AMD(MI300X) 算子后端对照表 + aiter 精确 API |
| `rebuild_env.sh` | 一键重建 venv（torch2.11+rocm7.0 / py3.12 / sglang0.5.15；路径按本节点，改用前调整） |

## 硬件峰值（reward 分母，MI300X / CDNA3）

| 项 | MI300X | (B200 原版) |
|---|---|---|
| HBM 带宽 | **5.3 TB/s** | 8 TB/s |
| FP8(e4m3) 峰值 | **2614.9 TFLOP/s** | 4.5 PFLOP/s |
| BF16 峰值 | **1307.4 TFLOP/s** | 2.25 PFLOP/s |
| FP8 ridge | ≈493 FLOP/byte | 562 |
| CU / wavefront | 304 / 64 | 148 SM / warp 32 |
| FP8 dtype | `e4m3fnuz`, scale by 224.0 | `e4m3fn` UE8M0, 448 |

## 两种模式（与 B200 rewardbench 完全一致）

```bash
source <venv>/bin/activate            # your torch2.11+rocm7.0 venv (see rebuild_env.sh)
export HIP_VISIBLE_DEVICES=0

# 1) baseline —— 测 13 个参考算子（reward 分母），写 reward CSV
python bench_AMD_GLM5_ops_prefill.py            # M sweep 1024/2048/4096 -> amd_glm5_ops_prefill_reward.csv
python bench_AMD_GLM5_ops_decode.py             # batch sweep 1/4/8/16/32/64 -> amd_glm5_ops_decode_reward.csv
python bench_AMD_GLM5_ops_prefill.py --m 4096   # 单 shape

# 逐算子详细 benchmark（含每算子实际 backend 列）
python amd_bench_glm5_prefill.py                # -> amd_glm5_prefill_perf.csv
python amd_bench_glm5_decode.py

# 计时：默认 hipGraph capture+replay；对不可 capture 的 kernel 自动回退 HIP-event
AMD_BENCH_NO_GRAPH=1 python bench_AMD_GLM5_ops_prefill.py   # 强制 event 计时
```

> **candidate 模式（PR 前的构建准备）**：优化后的算子按 kernel-harness 的 `--kernels-dir <dir>`
> 契约测分——`<dir>/<candidate>/solution.py`（`def run(...)` + 可选 `get_inputs`）+ `META.md`
> (`reward operator: <op>`) + `task.json`。本 baseline 建立了 reward 分母；candidate 引擎与 B200
> 版 `run_candidate_folder` 接口对齐，优化产物直接可测 speedup + roofline reward。

## Baseline 后端（双路策略）

- **默认 = torch-native ROCm**：`torch._scaled_mm`（hipBLASLt FP8 per-tensor）、`torch.mm`、
  chunked attention/mqa-logits。**必跑通**，给合法 MI300X baseline。
- **aiter 路径（`import aiter` 成功则自动启用）**：`aiter.ops.gemm_op_a8w8.gemm_a8w8_blockscale`、
  `aiter.fused_moe.fused_moe` 等 sglang-ROCm 生产 kernel。CSV `backend` 列记录每算子实际路径。

## 实测 baseline 摘录（MI300X 单卡，prefill M=4096）

| 算子 | backend | lat(ms) | AI | bound | reward |
|---|---|---|---|---|---|
| fused_qkv_a_proj | hipBLASLt | 0.139 | 2077 | compute | 0.362 |
| q_b_proj | hipBLASLt | 0.268 | 1558 | compute | 0.392 |
| o_proj | hipBLASLt | 0.685 | 3744 | compute | 0.461 |
| index_k_proj | hipBLASLt | 0.150 | 238 | memory | 0.546 |
| moe_gate/up/down | per-expert | ~0.82 | 1863 | compute | ~0.38 |
| **dsa_prefill_attn** | gather+chunked | 14.7 | 1719 | compute | **0.061** ← 最大 headroom |
| **index_score** | batched _scaled_mm | 46.0 | 2000 | compute | **0.018** ← 最大 headroom |

低 reward 的 `dsa_prefill_attn` / `index_score` 精确标注了 A 卡上最大的优化空间——它们在 B200 上
也依赖专用融合 kernel（flash_mla_sparse_fwd / fp8_mqa_logits），在 MI300X 上需要 aiter MLA 或
Triton-ROCm 融合实现。AI 值与 B200 rewardbench 完全一致，证明 cost 模型移植无误。

## 环境

```bash
# 参考节点：py3.12 / torch2.11.0.dev+rocm7.0 / sglang0.5.15 / 8×MI300X (gfx942, ROCm 7.0)
# rebuild_env.sh 顶部的 VENV / PYBASE / mirror 路径按本节点写死，改用前请调整。
bash rebuild_env.sh
```
