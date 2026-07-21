# AMD MI300X GLM-5.2 算子 Reward Bench

GLM-5.2 prefill / decode 算子的**性能 reward 评测**，目标硬件 **AMD Instinct
MI300X (CDNA3 / gfx942, ROCm 7.0)**。是 kernel-harness `rewardbench/`（B200/SM100 + DeepGEMM）
的 A 卡移植：算子清单、reward 公式、CSV schema 与 B200 版一一对应，只把 roofline 峰值换成
MI300X、后端 kernel 换成 sglang-ROCm（aiter / hipBLASLt）。在原 split MoE 诊断项之外，
这里还加入 fused SGLang MoE total，用于客观复现生产 ABI。

reward = **bound-aware roofline 利用率 ∈ [0,1]** = 实测吞吐 ÷ roofline 天花板
（compute-bound → 矩阵核利用率；memory-bound → HBM 带宽利用率，按 arithmetic intensity 自动判定）。
指标设计见 `operator_mapping.md`（算子对照表）与 kernel-harness 的 `../GLM5_ops_reward_design.md`。

## 文件

| 文件 | 说明 |
|---|---|
| `amd_glm5_ops_common.py` | 共享引擎（cost 模型 / AMD kernel builders / roofline reward / 计时 / driver） |
| `amd_bench_glm5_prefill.py` / `amd_bench_glm5_decode.py` | 逐算子延时+reward benchmark（含 split MoE diagnostics 与 fused MoE total，输出 backend/TFLOPS/GB/s/AI/reward） |
| `bench_AMD_GLM5_ops_prefill.py` / `bench_AMD_GLM5_ops_decode.py` | **reward-bench 入口**（=baseline 模式，产 reward 分母 CSV） |
| `amd_glm5_ops_prefill_reward.csv` / `amd_glm5_ops_decode_reward.csv` | **baseline 测试结果**：历史参考算子在 MI300X 上的 reward（重新跑会包含 fused MoE total） |
| `operator_mapping.md` | CUDA(B200)→AMD(MI300X) 算子后端对照表 + aiter 精确 API |
| `sglang_moe_configs/` | GLM-5.2 `E=8,N=2048,dtype=fp8_w8a8` fused MoE 本地 Triton config；AMD backend 会默认设置 `SGLANG_MOE_CONFIG_DIR` 使用它，避免 SGLang 因缺文件打印 warning。当前内容与 SGLang default config 等价，正式 tuned config 需要 correctness-gated tuning 后替换 |

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
source <venv>/bin/activate            # ROCm PyTorch + sglang/AITER environment
export HIP_VISIBLE_DEVICES=0

# 1) baseline —— 测参考算子（reward 分母），写 reward CSV
python bench_AMD_GLM5_ops_prefill.py            # M sweep 1024/2048/4096 -> amd_glm5_ops_prefill_reward.csv
python bench_AMD_GLM5_ops_decode.py             # batch sweep 1/4/8/16/32/64 -> amd_glm5_ops_decode_reward.csv
python bench_AMD_GLM5_ops_prefill.py --m 4096   # 单 shape

# 逐算子详细 benchmark（含每算子实际 backend 列；perf CSV 是本地运行产物，默认被 .gitignore 忽略）
python amd_bench_glm5_prefill.py                # -> amd_glm5_prefill_perf.csv
python amd_bench_glm5_decode.py

# 计时：默认 hipGraph capture+replay；对不可 capture 的 kernel 自动回退 HIP-event
AMD_BENCH_NO_GRAPH=1 python bench_AMD_GLM5_ops_prefill.py   # 强制 event 计时
```

## Taskset: selected 11 target tasks

本地统一评测默认使用 `../../tasksets/glm52_rocm_local.json`。它只覆盖目标范围内
的 11 个 GLM-5.2 ROCm 任务，并额外加入 `moe_total_prefill` /
`moe_total_decode` 作为 fused production total rollup，用于本地校准和
official total 汇总。它不是 “所有 GLM 任务” 的全量 taskset。

prefill 的 `M` 是 token 数/序列长度 `1024/2048/4096`；decode 的 `M` 是
单步 batch/active query 数，完整 sweep 与原 decode rewardbench 一致为
`1/4/8/16/32/64`，KV context `S=65536` 固定。

MoE 本地指标保留 4 个口径：生产等价的 fused
`Routed Expert Gate+Up/Down` total 作为本地 official/leaderboard 汇总指标，
`gate` / `up` / `down` 三个 split 指标作为目标任务和诊断指标继续单独输出。
taskset 里的 `score_scope` / `metric_group` / `metric_component` /
`production_equivalent` 会写入 rewardbench CSV，避免把 split proxy 误当成
fused production total。

本地统一 rewardbench:

```bash
python bench_AMD_GLM5_ops_taskset.py --smoke
python bench_AMD_GLM5_ops_taskset.py --phase prefill
python bench_AMD_GLM5_ops_taskset.py --phase decode
```

单任务 rewardbench:

```bash
python bench_AMD_GLM5_ops_taskset.py \
  --task index_score_prefill --smoke
```

## SGLang 官方 AMD 指标入口

SGLang 上游没有为本仓库的 GLM-5.2 11-task operator set 提供一个现成的官方
leaderboard。能直接查看的官方/上游指标分三类：

- Serving-level nightly 指标：`/opt/devmachine/lichangye/repos/sglang/docs/performance_dashboard/README.md`。
  在该目录运行 `python server.py --fetch-on-start` 后访问 `http://localhost:8000`；
  需要 GitHub token 才能拉取完整 GitHub Actions artifacts。这里看的是 throughput、
  latency、TTFT 等端到端服务指标。
- AMD 平台与 CI perf：`/opt/devmachine/lichangye/repos/sglang/docs/platforms/amd_gpu.md`
  和 `/opt/devmachine/lichangye/repos/sglang/test/registered/amd/perf/`。
  GLM 相关用例在 `mi30x/test_glm5_perf_amd.py`、`mi35x/test_glm5_perf_mi35x.py`
  等文件里。
- AMD MoE/kernel tuning：`/opt/devmachine/lichangye/repos/sglang/3rdparty/amd/tuning/TUNING.md`、
  `3rdparty/amd/tuning/benchmark_moe_rocm.py`、以及
  `benchmark/kernels/fused_moe_triton/`。

本目录的 CSV 是 local operator-level calibration：它把这些生产路径拆成可重复的
算子级 benchmark，用来驱动 KDA/agent 优化；正式声称 “SGLang baseline reproduced”
时仍需通过 `testbench/bin/baseline_calibration.py` 验证对应 production call site。

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
| **index_score** | aiter.fp8_mqa_logits | 46.0 | 2000 | compute | **0.018** ← 最大 headroom |

低 reward 的 `dsa_prefill_attn` / `index_score` 精确标注了 A 卡上最大的优化空间——它们在 B200 上
也依赖专用融合 kernel（flash_mla_sparse_fwd / fp8_mqa_logits），在 MI300X 上分别走 SGLang
tilelang/aiter production 路径或对应 fallback。AI 值与 B200 rewardbench 完全一致，证明 cost 模型移植无误。

## 环境

参考节点：py3.12 / torch2.11.0.dev+rocm7.0 / sglang0.5.15 / 8×MI300X
(gfx942, ROCm 7.0)。环境创建由机器管理员或 CI 提供；本目录只提交可移植的
benchmark、taskset、baseline reference 和必要的 SGLang MoE config，不提交节点
专属 venv 重建脚本。
