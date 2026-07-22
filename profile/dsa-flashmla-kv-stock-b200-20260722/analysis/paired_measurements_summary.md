# Paired eager and CUDA Graph measurement summary

This file is generated from the raw JSON artifacts by:

```bash
.venv/bin/python profile/dsa-flashmla-kv-stock-b200-20260722/harness/summarize_paired_measurements.py
```

`paired_median_speedup` is the producer-defined median of the 100 raw pair-wise speedups. It is not `reference_median / candidate_median`; the machine-readable JSON preserves both quantities at full precision. The per-session 3% gate is `paired_median_speedup >= 1.03`.

## Paired eager

| Bucket | Variant | Session | Raw file | Reference p50 (ms) | Candidate p50 (ms) | Paired p50 speedup | 3% gate | Graph correctness |
|---|---|---:|---|---:|---:|---:|---|---|
| M16 | control | 1 | `analysis/paired_control_m16_r1.json` | 0.05065600015223026 | 0.050255998969078064 | 1.0116785700126334 | FAIL | n/a |
| M16 | control | 2 | `analysis/paired_control_m16_r2.json` | 0.046271998435258865 | 0.0464479997754097 | 0.9926439545734452 | FAIL | n/a |
| M16 | control | 3 | `analysis/paired_control_m16_r3.json` | 0.045871999114751816 | 0.0448479987680912 | 1.0279251072229258 | FAIL | n/a |
| M16 | candidate | 1 | `analysis/paired_combine32_m16_r1.json` | 0.05559999868273735 | 0.05344000086188316 | 1.0368446908491809 | PASS | n/a |
| M16 | candidate | 2 | `analysis/paired_combine32_m16_r2.json` | 0.04580799862742424 | 0.04508800059556961 | 1.016080929397346 | FAIL | n/a |
| M16 | candidate | 3 | `analysis/paired_combine32_m16_r3.json` | 0.051343999803066254 | 0.05075199902057648 | 1.0208469237747453 | FAIL | n/a |
| M32 | control | 1 | `analysis/paired_control_m32_r1.json` | 0.05113599821925163 | 0.05023999884724617 | 1.018620630585207 | FAIL | n/a |
| M32 | control | 2 | `analysis/paired_control_m32_r2.json` | 0.05056000128388405 | 0.050383999943733215 | 1.0035909825232416 | FAIL | n/a |
| M32 | control | 3 | `analysis/paired_control_m32_r3.json` | 0.0498879998922348 | 0.04927999898791313 | 1.0101423187337268 | FAIL | n/a |
| M32 | candidate | 1 | `analysis/paired_combine32_m32_r1.json` | 0.05628800019621849 | 0.0568000003695488 | 0.9899187012353339 | FAIL | n/a |
| M32 | candidate | 2 | `analysis/paired_combine32_m32_r2.json` | 0.047919999808073044 | 0.047839999198913574 | 0.993736891518904 | FAIL | n/a |
| M32 | candidate | 3 | `analysis/paired_combine32_m32_r3.json` | 0.05660799890756607 | 0.05508799850940704 | 1.0109059803060507 | FAIL | n/a |

## Real CUDA Graph replay

| Bucket | Variant | Session | Raw file | Reference p50 (ms) | Candidate p50 (ms) | Paired p50 speedup | 3% gate | Graph correctness |
|---|---|---:|---|---:|---:|---:|---|---|
| M16 | control | 1 | `analysis/graph_control_m16_r1.json` | 0.032287999987602234 | 0.03246400132775307 | 0.9965673787027294 | FAIL | PASS |
| M16 | control | 2 | `analysis/graph_control_m16_r2.json` | 0.032287999987602234 | 0.032368000596761703 | 0.9990132931889931 | FAIL | PASS |
| M16 | control | 3 | `analysis/graph_control_m16_r3.json` | 0.031727999448776245 | 0.03223999962210655 | 0.9899497791600332 | FAIL | PASS |
| M16 | candidate | 1 | `analysis/graph_combine32_m16_r1.json` | 0.032416000962257385 | 0.03246400132775307 | 0.9980683526400792 | FAIL | PASS |
| M16 | candidate | 2 | `analysis/graph_combine32_m16_r2.json` | 0.03232000023126602 | 0.032735999673604965 | 0.9874070093797468 | FAIL | PASS |
| M16 | candidate | 3 | `analysis/graph_combine32_m16_r3.json` | 0.031727999448776245 | 0.032287999987602234 | 0.9831088361970577 | FAIL | PASS |
| M32 | control | 1 | `analysis/graph_control_m32_r1.json` | 0.036639999598264694 | 0.03713599964976311 | 0.9826796155360378 | FAIL | PASS |
| M32 | control | 2 | `analysis/graph_control_m32_r2.json` | 0.0366239994764328 | 0.03691200166940689 | 0.9928602419617203 | FAIL | PASS |
| M32 | control | 3 | `analysis/graph_control_m32_r3.json` | 0.03617599979043007 | 0.03652799874544144 | 0.9913718490734034 | FAIL | PASS |
| M32 | candidate | 1 | `analysis/graph_combine32_m32_r1.json` | 0.036607999354600906 | 0.03670400008559227 | 0.9939629592814309 | FAIL | PASS |
| M32 | candidate | 2 | `analysis/graph_combine32_m32_r2.json` | 0.036959998309612274 | 0.03705599904060364 | 0.9940551192379626 | FAIL | PASS |
| M32 | candidate | 3 | `analysis/graph_combine32_m32_r3.json` | 0.03614399954676628 | 0.036320000886917114 | 0.9991489296775157 | FAIL | PASS |

## Compiler/build control comparison

This is descriptive: it subtracts the median of the three control session speedups from the median of the three candidate session speedups. The control and candidate sessions were not paired to each other, so this delta is not itself an acceptance metric.

| Mode | Bucket | Control median session speedup | Candidate median session speedup | Candidate - control | Control passes | Candidate passes |
|---|---|---:|---:|---:|---:|---:|
| eager | M16 | 1.0116785700126334 | 1.0208469237747453 | 0.009168353762111892 | 0/3 | 1/3 |
| eager | M32 | 1.0101423187337268 | 0.993736891518904 | -0.016405427214822854 | 0/3 | 0/3 |
| cuda_graph | M16 | 0.9965673787027294 | 0.9874070093797468 | -0.009160369322982587 | 0/3 | 0/3 |
| cuda_graph | M32 | 0.9913718490734034 | 0.9940551192379626 | 0.0026832701645592705 | 0/3 | 0/3 |

## Unpaired stock baseline context

These warmup=5, repeat=50 runs expose cold-to-warm drift and are not used for acceptance.

| Bucket | Session | Raw file | Reference p50 (ms) |
|---|---:|---|---:|
| M16 | 1 | `analysis/baseline_stock_m16_r1.json` | 0.049456000328063965 |
| M16 | 2 | `analysis/baseline_stock_m16_r2.json` | 0.047839999198913574 |
| M16 | 3 | `analysis/baseline_stock_m16_r3.json` | 0.044544000178575516 |
| M32 | 1 | `analysis/baseline_stock_m32_r1.json` | 0.04992000013589859 |
| M32 | 2 | `analysis/baseline_stock_m32_r2.json` | 0.04798400029540062 |
| M32 | 3 | `analysis/baseline_stock_m32_r3.json` | 0.047728000208735466 |

## Evidence outcome and anomalies

- The specialization candidate clears 1.03 in only 1/3 M16 eager sessions. M16 eager r2/r3 do not reproduce it.
- The specialization candidate clears 1.03 in 0/6 real CUDA Graph sessions. Every graph correctness/mutation/anti-alias check passes: `true`.
- No candidate mode/bucket group has more than one session above the 3% threshold. The only favorable candidate row is `analysis/paired_combine32_m16_r1.json`; graph replay reverses that apparent win.
- The compiler/build control itself moves around unity. Its three-session medians and the candidate-minus-control deltas above bound the apparent build/toolchain and timing variation.
- The unpaired stock p50 changes from r1 to r3 by -0.09932061058122243 for M16 and -0.043910254831646234 for M32. This cold/warm drift is why those files are context only.
- `analysis/paired_combine32_m32_r2.json` has a slightly lower candidate median than reference median but a paired speedup below 1.0. This is not corruption: median(pair-wise ratios) and ratio(independent medians) are different statistics.
