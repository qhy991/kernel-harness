# 协作：如何接入你优化的 kernel

本目录是 **llm_flops 对齐的层延时测试标准**。别人优化了别的算子时，按下面步骤即可和现有结果并列对比。

## 契约（必须满足）

1. **ABI**：`run(inputs: dict) -> Tensor`，键名与 `glm52_ops.build_inputs` / llm_flops 冻结输入一致  
   - GEMM：`x_fp8, x_scale, w_fp8, w_scale, out`  
   - MoE：另加 `masked_m, expected_m`
2. **计时**：由 `bench_decode.py` / `bench_prefill.py` 统一用 CUDA Graph 测；不要自己再套一层 Graph
3. **可比性**：bench 会用 **llm_flops 同一套量化**建输入，再在同输入上测 stock 与你的 `run`

## 接入步骤

### 1. 放入 candidate

```text
archive/0720-Best-GLM-52/best/<your_campaign>/candidate/candidate.py
```

或任意路径，只要含 `candidate.py`（或单文件 `.py`）。

### 2. 在 `_common.py` 注册 swap

Decode 改 `DECODE_SWAPS`，Prefill 改 `PREFILL_SWAPS`：

```python
# (harness_op, archive_subdir, kind)
# kind: "fp8_gemm" | "moe_masked" | None(=不换，走 llm_flops stock)
"o_proj": ("o_proj", "o_proj_decode_hbm35", "fp8_gemm"),
```

`archive_subdir` = `best/` 下目录名。  
若 harness 尚无对应 task，先在 `testbench/tasks/glm52/` 有同 phase 的 task（用于 `candidate_loader` 解析）。

### 3. 跑 bench

```bash
cd /path/to/Kernel-Harness/archive/0720-Best-GLM-52/llm_flops_style
CUDA_VISIBLE_DEVICES=0 ../../../.venv/bin/python bench_decode.py   # 或 bench_prefill.py
```

输出：`results/glm5_*_swapped_perf.csv`，列含 `stock_ms / avg_ms / same_inputs`。

### 4. 更新对比表

把你的行补进 [`COMPARISON_TABLE.md`](COMPARISON_TABLE.md)，并注明 archive 路径。

## 可选：Harness CUPTI 门禁

正确性 + CUPTI cold-L2 仍用任务门禁：

```bash
testbench/tasks/glm52/<op>_<phase>/run.sh --candidate /path/to/your/candidate
```

层替换验收（advisory）：`testbench/bin/accept_layer.py --M 32 --swap o_proj=...`

## 当前已注册 swaps

见 `_common.py` 中 `DECODE_SWAPS` / `PREFILL_SWAPS`，结果见 [`COMPARISON_TABLE.md`](COMPARISON_TABLE.md)。
