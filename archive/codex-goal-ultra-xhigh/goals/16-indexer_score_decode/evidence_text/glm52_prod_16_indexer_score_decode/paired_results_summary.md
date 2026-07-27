# Paired results summary

All ratios below are runner-recorded medians of same-process, alternating
`reference_ms / candidate_ms` pairs. Each row contains three independent
series from one wrapper-held GPU lease. A bucket passes only when every series
is correct and at least `1.03x`.

## Score plus top-k

| Mode | Bucket | CuTe-DSL series | Median | Minimum | Decision |
|---|---:|---|---:|---:|---|
| CUDA graph | M16 | 1.019197x, 1.003703x, 1.011231x | 1.011231x | 1.003703x | reject |
| CUDA graph | M32 | 0.977756x, 1.040790x, 1.040749x | 1.040749x | 0.977756x | reject |
| eager | M16 | 0.819856x, 0.805730x, 0.802370x | 0.805730x | 0.802370x | reject |
| eager | M32 | 0.798817x, 0.780203x, 0.791694x | 0.791694x | 0.780203x | reject |

The graph identity controls have median series ratios of `0.990377x` at M16
and `0.984434x` at M32. The eager identity controls are `1.003506x` and
`0.992583x`. Raw samples and percentile diagnostics are in
`runs/20260723T113910Z/paired_summary.json`.

## Complete indexer and selected DSA

| Scope | Bucket | CuTe-DSL series | Median | Minimum | Decision |
|---|---:|---|---:|---:|---|
| complete indexer | M16 | 1.014806x, 1.031918x, 1.056648x | 1.031918x | 1.014806x | reject |
| complete indexer | M32 | 1.028806x, 1.032633x, 1.015328x | 1.028806x | 1.015328x | reject |
| indexer + TRT-LLM DSA | M16 | 1.011807x, 1.014385x, 1.011058x | 1.011807x | 1.011058x | reject |
| indexer + TRT-LLM DSA | M32 | 1.021669x, 1.019186x, 1.020334x | 1.020334x | 1.019186x | reject |

These are 60-pair bidirectional CUDA-graph series. Every pre-timing,
post-timing, reference-graph, and candidate-graph correctness check passed.
Raw samples are in
`region_runs/20260723T120153Z/paired_summary.json`.

## Four-rank diagnostic

| Bucket | CuTe-DSL series | Median | Minimum | Decision |
|---|---|---:|---:|---|
| TP4/DP4 diagnostic M16 | 1.004793x, 1.002506x, 1.018696x | 1.004793x | 1.002506x | reject |
| TP4/DP4 diagnostic M32 | 1.023810x, 1.030381x, 1.017586x | 1.023810x | 1.017586x | reject |

These are 40-pair, maximum-rank CUDA-event results. They are explicitly
diagnostic and do not satisfy or weaken the TP8/DP8/EP8 gate. Raw samples are
in `tp4_runs/20260723T121417Z/paired_summary.json`.

## Correctness

For both M16 and M32, the backend validation reports CUDA-graph replay pass,
maximum score difference `2.384185791015625e-7`, exact 2048-element top-k set
equality on every row, and zero rows with a different set. Exact top-k order
is not required because equal/near-equal scores may be returned in a different
order; the physical slot set consumed by DSA is identical. The containing
region additionally checks the current-token K-cache mutation and TRT-LLM DSA
output with the runner's production tolerance contract.

## Disposition

No score, complete-indexer, selected-DSA, or TP4 diagnostic bucket clears the
three-series `1.03x` gate. No CuTe-DSL bucket is enabled.
