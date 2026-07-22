# AMD MI300X GLM-5.2 算子 Reward Bench

GLM-5.2 prefill / decode 各 13 个算子的**性能 reward 评测**，目标硬件 **AMD Instinct
MI300X (CDNA3 / gfx942, ROCm 7.0)**。是 kernel-harness `rewardbench/`（B200/SM100 + DeepGEMM）
的 A 卡移植：算子清单、reward 公式、CSV schema 与 B200 版一一对应，只把 roofline 峰值换成
MI300X、后端 kernel 换成 sglang-ROCm（aiter / hipBLASLt）。

> **正确性检验 bench (opbench) 也已移植 + 完整优化 flow —— 见下方「MI300X 正确性 bench + 优化 flow」。**
> 现在两个 mark 都在 A 卡上：**opbench**（正确性 gate + 加速比，`testbench/`）与
> **rewardbench**（roofline reward tracker，本目录）。二者共享同一 cost model，且都经
> `validate_marks.py` 严格自检。

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
| **`e2e_profile_20260722/`** | **TP=8 真实 e2e prefill profile 份额 + 与 harness baseline 不对齐台账（共享给同伴）** |
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

---

## MI300X 正确性 bench (opbench) + 优化 flow

把 B200 的**正确性检验 bench**（`testbench/`：`run.sh` → 正确性 + 加速比 + roofline
reward，CUPTI 计时）也移植到了 MI300X，**零改动 oracle** `testbench/harness/glm52_ops.py`
——harness 早已 provider 抽象化，移植只是加一个后端 bundle。

### 组件（本目录，`rewardbench/amd/`）

| 文件 | 作用 |
|---|---|
| `../../testbench/harness/backends/rocm_mi300x.py` | ROCm 后端：`DeviceProfile`（MI300X 峰值 / e4m3fnuz）+ 自带 torch+triton provider（**triton blockwise-fp8 GEMM** 参考核 + chunked sparse-MLA + torch 参考）+ HIP-event 计时器。仅依赖 torch+triton（本机无 aiter/deep_gemm/sgl_kernel） |
| `validate_marks.py` | **Req-0 mark 自检（24 项全绿）**：opbench↔rewardbench cost model 逐字节一致、triton-GEMM vs bf16 dequant oracle（calc_diff ~5e-9）、chunked-MLA vs full-attn oracle、反作弊有效（no-op 失败、2× 幅度错误即便 cosine 1.0 也失败、poison 生效） |
| `emit_targets.py` | 测 opbench 参考基线，产出 **target = min(hardware roofline, 1.5×baseline)**；写 `amd_glm5_targets.csv`（baseline 右边并排 target 列），`--augment` 给两张 reward CSV 在 `reward` 右侧加 `target_reward`/`target_desc` |
| `run_flow.py` | **一条命令跑完整 flow**：先正确性 gate（错误候选 exit 2，不进性能比较），再 roofline reward vs target，产出 flotilla 兼容 `status.json` |
| `amd_glm5_targets.csv` | 3 个优化目标算子（o_proj/index_k prefill、dsa_attn decode）的 baseline + target 台账 |

### 用法

```bash
cd ~/repos/kernel-harness && source /opt/mizar/huyan/amd_env.sh && export HIP_VISIBLE_DEVICES=0
# 硬盘全满：amd_env.sh 把 triton/LLVM 临时目录 + 缓存重定向到 /opt/mizar tmpfs（否则编译报 No space）

# 0) 先验两个 mark 本身正确（防 flow 出问题）
.venv/bin/python rewardbench/amd/validate_marks.py            # 24 checks

# 1) 产出 baseline + target 台账，并给 reward CSV 加 target 列
.venv/bin/python rewardbench/amd/emit_targets.py --augment

# 2) 跑 flow：候选 → 正确性 gate → 性能 vs target
.venv/bin/python rewardbench/amd/run_flow.py o_proj_prefill --candidate <kernel.py> --round 1
#   state: INCORRECT (exit 2) | CORRECT_BELOW_TARGET (exit 1) | TARGET_MET (exit 0)

# 也可直接跑 opbench gate（正确性 + 加速比）
testbench/tasks/glm52/o_proj_prefill/run.sh --candidate <kernel.py>
```

后端选择靠 gitignore 的 `testbench/harness.env`（本机已设 rocm/rocm-mi300x/torch-triton-rocm/event）
+ `.venv` 软链到 `/opt/mizar/huyan/venvs/amd-glm52`。`get_backend()` 也会读 `harness.env` 兜底。

### 已跑通结果（单卡 MI300X）

| 算子 / phase | family/bound | 参考 reward | target(1.5×) | 最佳候选 | 结果 |
|---|---|---|---|---|---|
| `o_proj` prefill | gemm / compute | 0.045–0.052 | 0.067–0.078 | autotuned triton blockwise-fp8 GEMM | **TARGET_MET**，geomean **1.62×**，3/3 shape win，116% of target |
| `index_k` prefill | gemm / memory | 0.096 | 0.145 | 同上（shape-generic） | **TARGET_MET**，geomean **1.72×**，3/3 win，116% of target |
| `dsa_attn` decode | mla / memory | 0.030–0.039 | 0.045–0.059 | 融合 flash-DECODING（tk-split + 融合 combine 核） | **TARGET_MET**，geomean **2.01×**，2/2 win，136% of target |

> `dsa_attn` 最难（B200 也靠专用 flash_mla 核）：朴素 torch-SDPA 正确但慢 37× 被 gate 拒绝；
> f32-bmm 1.30×；plain flash-decode 因 M=16/32 网格太小 CU 空转只有 1.05×；tk-split 提并行度到
> 1.55×；**再把 combine 融进单个 triton 核**去掉 M=16 的 launch 尾巴 → **2.01×**。三个算子全部
> 先过正确性 gate 再达 target。

KDA-Pilot 原生任务目录见 `~/repos/KDA-Pilot/mi300x/`；flotilla 监控用 `kernel_harness` evaluator
（`~/repos/flotilla/flotilla/evaluator/kernel_harness_eval.py`）+ `demo/glm52_mi300x_demo.py` 播种。

> **两个 baseline 的区别**（勿混淆）：上文「实测 baseline 摘录」(o_proj 0.461) 是 rewardbench
> 直接测 **hipBLASLt per-tensor** GEMM（随机输入、per-tensor 量化）的吞吐；opbench/flow 的
> 参考基线 (o_proj ~0.05) 是 **triton blockwise-fp8** 参考核——它消费 opbench 冻结的 *blockwise*
> 量化输入（与 deep_gemm B200 路径同构，是正确性 oracle 兼延时分母），未调优故利用率低。二者
> 量化粒度不同、测的是不同问题；per-tensor hipBLASLt 无法直接作为 opbench 候选（数值与 blockwise
> 参考不符会 fail 正确性）。flow 的 target 一律基于 opbench 参考基线，内部自洽。
