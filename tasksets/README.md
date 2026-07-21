# Kernel Harness Tasksets

Tasksets define selected GLM-5.2 ROCm operator subsets. This PR does not attempt
to cover every generated GLM task directory; it only wires the 11 target tasks
plus local fused MoE total rollups needed for calibration.

`glm52_rocm_local.json` is the single committed taskset. It contains:

- 9 prefill tasks from the original target set.
- 2 added MoE decode tasks for routed expert gate/up and down.
- 2 local-only fused MoE total rollups, `moe_total_prefill` and
  `moe_total_decode`, for production-equivalent total reporting.

MoE still keeps both fused totals and split diagnostics locally:
`moe_total_*` is the production-equivalent Routed Expert Gate+Up/Down total used
for local official/leaderboard rollup, while `gate`, `up`, and `down` remain
separate upload/diagnostic component tasks.

The taskset carries machine-readable scoring metadata:

- `score_model.official_metrics` lists the metrics that should appear in the
  formal rollup.
- `score_model.moe_metrics` defines the MoE total metric and its component
  diagnostics.
- Per-task `score_scope`, `metric_group`, `metric_component`, and
  `production_equivalent` are copied into testbench JSON and rewardbench CSV
  outputs so workflow agents do not infer the scoring role from the task name.

Axis semantics:

- Prefill `M` is the token count / sequence length, swept over `1024, 2048, 4096`.
- Decode `M` is batch size / active query count for one decode step, swept over
  `1, 4, 8, 16, 32, 64` in the full AMD rewardbench. The KV context remains
  `S=65536`.
- Smoke mode uses one cheap representative shape: prefill `M=1024`, decode `M=16`.

Local wrappers default to `glm52_rocm_local.json`:

```bash
python testbench/bin/evaluate_glm52_taskset.py --smoke --candidate-reference
python rewardbench/amd/bench_AMD_GLM5_ops_taskset.py --smoke
```

Single-task checks select from the same taskset:

```bash
python testbench/bin/evaluate_glm52_taskset.py \
  --task index_score_prefill --smoke --candidate-reference

python rewardbench/amd/bench_AMD_GLM5_ops_taskset.py \
  --task index_score_prefill --smoke
```
