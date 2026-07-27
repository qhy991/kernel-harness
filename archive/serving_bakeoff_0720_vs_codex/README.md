# Serving bake-off: archive-0720 vs Codex

Unified **production ABI** comparison under [`serving_native/`](../../serving_native/).

| Field | Value |
|---|---|
| Protocol | interleaved paired A/B vs stock (`SGLANG_GLM52_OPT=0`) |
| Shapes | decode M=16/32, single GPU |
| Registry | [`registry.yaml`](registry.yaml) |
| 0720 ports | [`serving_ports/0720/`](serving_ports/0720/) |
| Codex sources | [`../codex-goal-ultra-xhigh/goals/`](../codex-goal-ultra-xhigh/goals/) |

## Why some 0720 entries are not runnable

Most 0720 TARGET_MET wins came from **online f32→UE8M0 scale packing** against the
synthetic harness. Production `serving_native` already supplies packed int32
scales, so those candidates cannot be lightly wrapped:

| Tag | Meaning |
|---|---|
| `absorbed_in_stock` | Pack/layout win already present in production stock |
| `already_stock` | 0720 kernel is the same backend as serving stock (DSA trtllm-gen) |
| `requires_goal_runtime` | Candidate needs goal-worktree Runtime helpers (not mainline) |
| `run` | Real `run(inputs, runtime)` candidate on packed ABI |

Runnable 0720 ports in this tree:

- `indexer_wq_b_packed.py` — goal-15 native port of hechenxi `index_q_upproj`
- `fused_qkv_a_packed.py` — new packed Triton for N=2624 / K=6144

## Run

```bash
cd /home/qinhaiyan/Kernel-Harness/archive/serving_bakeoff_0720_vs_codex

# All runnable cases (uses with_flexible_gpu.sh when CUDA_VISIBLE_DEVICES unset)
./run_bakeoff.sh

# Or pin a GPU / single case
CUDA_VISIBLE_DEVICES=0 ./run_bakeoff.sh indexer_wq_b_m16_0720
```

Outputs:

- `results/<case_id>.json` — raw serving_native JSON (or bakeoff_error)
- `results/<case_id>.log` — stdout/stderr
- `results/bakeoff_summary.csv` — flat table
- `BAKEOFF.md` — human summary
