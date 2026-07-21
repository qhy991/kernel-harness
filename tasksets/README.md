# Kernel Harness Tasksets

Tasksets define GLM-5.2 ROCm operator subsets for two different purposes:

- **Local unified evaluation:** `glm52_rocm_local.json` runs all local checks in
  one command. It contains the 11 upload tasks plus fused MoE total rollups used
  only for local calibration and official-total reporting.
- **Upload / submission:** every upload task is separate. Use the one-task JSON
  files under `glm52_rocm_upload/<task_id>.json`; do not submit the local
  aggregate as one bundled task.

`glm52_rocm_11.json` is only an index of the 11 upload tasks. It intentionally
excludes fused MoE total rollups because those rollups are local scoring /
calibration helpers, not extra upload tasks.

The 11 upload tasks are:

- 9 prefill tasks from the original target set.
- 2 added MoE decode tasks for routed expert gate/up and down.

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

Upload-style single-task checks pass an explicit one-task taskset:

```bash
python testbench/bin/evaluate_glm52_taskset.py \
  --taskset tasksets/glm52_rocm_upload/index_score_prefill.json \
  --smoke --candidate-reference

python rewardbench/amd/bench_AMD_GLM5_ops_taskset.py \
  --taskset tasksets/glm52_rocm_upload/index_score_prefill.json \
  --smoke
```
