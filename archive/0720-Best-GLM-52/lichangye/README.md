# GLM-5.2 ROCm MI300X Best Candidates - lichangye

This directory archives the accepted best candidates from the GLM-5.2 ROCm/MI300X
KDA-Pilot run on branch `codex/amd-glm52-rocm-evalbench-v2`.

Source commit:

```text
5efb3cfc292a45056fc455c9b9d610bb51d5d0c6
```

## Contents

| Task | Candidate | Accepted result |
|------|-----------|-----------------|
| `moe_total_decode` | `moe_total_decode/candidate/candidate.py` | `moe_total_decode/result.json` |
| `moe_total_prefill` | `moe_total_prefill/candidate/candidate.py` | `moe_total_prefill/result.json` |
| `dsa_prefill_attn` | `dsa_prefill_attn/candidate/candidate.py` | `dsa_prefill_attn/result.json` |
| `index_score_prefill` | `index_score_prefill/candidate/candidate.py` | `index_score_prefill/result.json` |

## Summary

All four official ROCm/MI300X targets are correct and pass the performance gate
in their accepted result JSONs:

- `moe_total_decode`: 2/2 shapes won, 0 regressions.
- `moe_total_prefill`: 3/3 shapes won, 0 regressions.
- `dsa_prefill_attn`: 3/3 shapes won, 0 regressions.
- `index_score_prefill`: 3/3 shapes won, 0 regressions.

See `results.md` for the metric summary and `SHA256SUMS` for file integrity
checks.

## Re-run

From the repository root on the ROCm/MI300X runner:

```bash
source /home/lichangye/kernel-harness-amd/.humanize/kernel-agent/runenv.sh
python testbench/bin/evaluate_glm52_taskset.py \
  --taskset tasksets/glm52_rocm_local.json \
  --task moe_total_decode \
  --repeat 10
```

Replace `moe_total_decode` with any archived task name to re-run that task.
