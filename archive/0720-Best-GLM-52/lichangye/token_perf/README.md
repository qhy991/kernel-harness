# Token / Performance Timeline

This directory visualizes the first GLM-5.2 ROCm/MI300X KDA-Pilot loop.

The x-axis is cumulative **fresh Claude tokens since loop start**, computed from
the first-loop Claude transcript:

```text
/home/lichangye/.claude/projects/-home-lichangye-kernel-harness-amd/ab5d0783-1275-46e7-bc91-2abf03b1bfd7.jsonl
```

Fresh tokens are:

```text
input_tokens + cache_creation_input_tokens + output_tokens
```

`cache_read_input_tokens` is excluded from the plotted x-axis because it inflates
the processed-token count without representing newly-created prompt/cache work.
The CSV still records effective tokens, including cache reads, for audit.

The y-axis is task performance:

```text
geomean primary-util ratio = candidate primary util / reference primary util
```

That is the same ratio family used by the `roofline_mfu_bw` scorer. For
compute-bound shapes the primary util is MFU; for memory-bound shapes it is HBM
bandwidth utilization.

## Files

| File | Meaning |
|------|---------|
| `token_perf_points.csv` | Source data used for every plot. |
| `token_perf_all_tasks.svg/png` | Overlay of all four official tasks. |
| `token_perf_moe_total_decode.svg/png` | Timeline for `moe_total_decode`. |
| `token_perf_moe_total_prefill.svg/png` | Timeline for `moe_total_prefill`. |
| `token_perf_dsa_prefill_attn.svg/png` | Timeline for `dsa_prefill_attn`. |
| `token_perf_index_score_prefill.svg/png` | Timeline for `index_score_prefill`. |
| `build_token_perf.py` | Rebuild script. Run from the repository root. |

## Rebuild

From the repository root:

```bash
python archive/0720-Best-GLM-52/lichangye/token_perf/build_token_perf.py
```
