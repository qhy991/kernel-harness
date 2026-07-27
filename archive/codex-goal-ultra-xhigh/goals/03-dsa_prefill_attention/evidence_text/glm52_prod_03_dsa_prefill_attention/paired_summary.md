# Paired M4096 results

All rows use the isolated SGLang worktree and interleaved CUDA-event pairs.
The pooled percentiles use the runner's discrete order-statistic convention.

| Comparison | Runs × pairs | p10 | paired p50 | p90 | 3% gate |
|---|---:|---:|---:|---:|---:|
| isolated stock-vs-stock noise floor | 3 × 20 | 0.982199× | 0.999388× | 1.007681× | FAIL |
| isolated source-trial PDL-off vs stock | 3 × 30 | 0.987667× | 0.996817× | 1.004250× | FAIL |

The PDL-off trial is a 0.32% pooled median regression and is rejected.
The earlier `baseline_*.json` and `pdl_off_*.json` files are retained but
excluded from the headline because their runner default resolved a separate
same-SHA SGLang checkout instead of the explicitly isolated worktree.
