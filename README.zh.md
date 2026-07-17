# Kernel-Harness

面向 **GLM-5.2 / B200** 的 SGLang 算子优化任务集。

**24 个 task** = 12 个算子 × 2 个 phase（prefill / decode），位于
`testbench/tasks/glm52/`。全部算子只在 `testbench/harness/glm52_ops.py` 里定义一次；
每个 task 目录只声明"我是哪个问题"，一条命令同时判定正确性、延迟、speedup 与
roofline reward：

```bash
T=testbench/tasks/glm52/o_proj_decode

$T/run.sh --describe                        # 这是什么问题？
$T/run.sh --describe --json                 # 同上，机器可读（== problem.json）
$T/run.sh                                   # 判定门
$T/run.sh --candidate ~/kernels/mine.py     # 测任意 kernel，无需改动 task
```

退出码：`0` 正确且更快 · `1` 正确但没更快 · `2` 不正确 · `3` 基础设施/契约错误。

从这里开始：**[`AGENTS.md`](AGENTS.md)**。Triton 与 CUDA `.cu` 候选的实测示例：
[`testbench/docs/GLM52_CANDIDATES.md`](testbench/docs/GLM52_CANDIDATES.md)。

## 已退役

Kimi-K2.7、MiniMax-M3，它们使用的 `solution.py` + `definition.json` 契约，
`evaluate.py` / `integrate.py`，以及更早的代理基准目录，全部移入
[`legacy/`](legacy/README.md)。它们仍可运行，但不是本仓库的任务集，也不是任何东西的
oracle。

## 一次运行为什么可信

- **单一定义**：输入、reference、阈值、mask、成本模型、峰值只在 `glm52_ops.py` 里。
  task 若重述其中任何一项，直接 exit 3。
- **同一份字节**：一份 frozen inputs 同时喂 reference 和候选；两次调用之间把共享输出
  缓冲区毒化成 NaN，所以"什么都不算"的候选拿不到参考答案。
- **上游判据**：FlashMLA 的三层检查，聚合量用 DeepGEMM 的 `calc_diff` 原文——不是
  allclose，也不是对尺度失明的 cosine。
- **设备侧计时**：CUPTI cold-L2 device-kernel 中位数。用墙钟的话，这个算子"带宽利用率"
  的一半其实是 Python dispatch。
- **逐 shape 判赢**：至少一个 shape 领先、没有 shape 退化；候选可以在赢不了的 shape 上
  fallback 到 reference——SGLang 自己就是这么做的。
- **每次运行都留档**：`runs/<model>/<task>/<run_id>/` 存 `result.json`、终端日志、环境，
  以及跑过的候选的逐字节副本。
