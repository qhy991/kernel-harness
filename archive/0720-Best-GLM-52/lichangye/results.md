# Results - GLM-5.2 ROCm MI300X Best Candidates

Accepted gate artifacts are archived as each task's `result.json`.

| Task | Geomean primary-util ratio | Conservative geomean ratio | Min conservative ratio | Geomean MFU | Geomean BW util | Shapes won | Regressions | Worst calc_diff |
|------|----------------------------|-----------------------------|------------------------|-------------|-----------------|------------|-------------|-----------------|
| `moe_total_decode` | 1.0655 | 1.0532 | 1.0518 | 0.030953 | 0.340746 | 2/2 | 0 | 0 |
| `moe_total_prefill` | 1.0809 | 1.0587 | 1.0263 | 0.266527 | 0.061196 | 3/3 | 0 | 0 |
| `dsa_prefill_attn` | 1.3044 | 1.2766 | 1.2603 | 0.034010 | 0.005563 | 3/3 | 0 | 2.8841951178470993e-06 |
| `index_score_prefill` | 2.8371 | 2.8137 | 1.5375 | 0.121637 | 0.030325 | 3/3 | 0 | 0 |

Overall task-level geomean primary-util ratio across the four official targets:

```text
1.4369x
```

MFU and BW-util improvement relative to the reference implementation:

| Task | MFU ratio | MFU improvement | BW-util ratio | BW improvement |
|------|-----------|-----------------|---------------|----------------|
| `moe_total_decode` | 1.0655x | +6.55% | 1.0655x | +6.55% |
| `moe_total_prefill` | 1.0809x | +8.09% | 1.0809x | +8.09% |
| `dsa_prefill_attn` | 1.3044x | +30.44% | 1.3044x | +30.44% |
| `index_score_prefill` | 2.8371x | +183.71% | 2.8371x | +183.71% |

Overall task-level geomean improvement:

```text
MFU:     1.4369x (+43.69%)
BW util: 1.4369x (+43.69%)
```

The matching MFU and BW-util ratios are expected for a fixed shape: FLOPs and
HBM bytes are fixed, so reducing runtime scales both utilization measures by the
same factor. The scorer still uses the `roofline_mfu_bw` policy: compute-bound
shapes score by MFU, memory-bound shapes score by BW utilization.
