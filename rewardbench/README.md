# GLM-5 算子性能测试 Reward Bench

GLM-5.2 prefill / decode 各 13 个算子的**性能 reward 评测**（B200/SM100，sglang ABI 对齐）。
reward = bound-aware roofline 利用率 ∈ [0,1]（compute-bound→TC 利用率，memory-bound→HBM 带宽利用率，
按 arithmetic intensity 自动判定）。指标设计与依据见 `GLM5_ops_reward_design.md`。

## 文件

| 文件 | 说明 |
|---|---|
| `bench_GLM5_ops_prefill.py` | prefill 评测脚本（双模式） |
| `bench_GLM5_ops_decode.py`  | decode 评测脚本（双模式） |
| `glm5_ops_common.py`        | 共享引擎（cost 模型 / kernel builders / roofline reward / candidate runner） |
| `GLM5_ops_reward_design.md` | 指标设计文档（算子 reference 表、dtype/硬件分支、接口说明） |
| `glm5_ops_prefill_reward.csv` / `glm5_ops_decode_reward.csv` | **baseline 测试结果**：13 个 sglang/llm_flops 参考算子的 reward（reward 分母基准） |
| `glm5_ops_prefill_candidates.csv` / `glm5_ops_decode_candidates.csv` | candidate 模式在 `best-kernels-reward-bench` 样例上的输出示例 |

## 两种模式

```bash
# 1) baseline —— 测 13 个参考算子（reward 基准）
python bench_GLM5_ops_prefill.py            # M sweep 1024/2048/4096 -> glm5_ops_prefill_reward.csv
python bench_GLM5_ops_decode.py             # batch sweep 1/4/8/16/32/64 -> glm5_ops_decode_reward.csv

# 2) candidate —— 输入「给定优化算子文件夹」，仅性能（无正确性）
python bench_GLM5_ops_prefill.py --kernels-dir <dir>   # 只测 phase==prefill 的候选
python bench_GLM5_ops_decode.py  --kernels-dir <dir>   # 只测 phase==decode  的候选
# <dir> 可为父目录（多个候选），也可直接指向单个算子目录（其下有 solution.py），供 flow 单算子调用
#   结果去向：① 终端逐行打印（带 UTC 时间戳）② 追加进该算子目录下的 reward_bench.csv（带时间戳历史）
#             ③ 汇总进 glm5_ops_{prefill,decode}_candidates.csv
#   选项：--repeat N（取最快）、--no-baseline（只出 reward）、--round K、--csv 自定义汇总文件
```

candidate 文件夹结构（参照 `best-kernels-reward-bench.zip` 解压树）：
```
<dir>/<candidate>/solution.py   # 必需：def run(...)；可选 def get_inputs(axes,device)
                  /task.json     # op/phase/family/sweep/K/N/E（可为空）
                  /META.md       # "reward operator: <op>" 归属
```
算子为**每阶段一类 13 个**；样例文件夹通常不全，未提供候选的算子在 candidate 模式下缺席，
其 roofline 参考值见 baseline 的 `*_reward.csv`。正确性测试是上游独立环节，本模块不实现。

## 环境

B200 上需 `DEEPGEMM_SCALE_UE8M0=True`（deep_gemm blockwise-FP8 要求 UE8M0 scale）。
依赖 kernel-harness venv 的 torch / deep_gemm / sgl_kernel / flashinfer：
```bash
CUDA_VISIBLE_DEVICES=0 DEEPGEMM_SCALE_UE8M0=True \
  ../kernel-harness/.venv/bin/python bench_GLM5_ops_prefill.py
```
