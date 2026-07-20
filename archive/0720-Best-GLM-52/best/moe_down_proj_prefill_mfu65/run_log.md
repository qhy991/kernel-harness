# Run Log — moe_down_proj_prefill 65% MFU campaign (Round 0)

Environment: NVIDIA B200 (SM100, CC 10.0), host `dry-vm-embraces-fin-03`, torch 2.11.0+cu130,
CUDA 13.0, deep_gemm 0.1.4 (SGLang fork), sgl_kernel 0.4.4, driver 610.43.02. Harness
`/home/qinhaiyan/Kernel-Harness` @ git 7d79e5e (branch main). GPU pinned to physical id 1
(`CUDA_VISIBLE_DEVICES=1` -> `cuda:0`), idle-checked before every measurement.

All timestamps UTC, 2026-07-18.

| Time | Step | Action | Result |
|---|---|---|---|
| 08:42 | Seed copy (AC-1) | `cp` harness seed -> `candidate/candidate.py`; sha256 + cmp | byte-identical sha256 `f0bff53c…`; harness `git status` shows only the 3 pre-existing dirty candidates; moe_down clean |
| 08:47 | Baseline (AC-2/AC-5) | `./run.sh --candidate` (seed), idle GPU 1 | CORRECT, all neutral (exit 1); MFU 47.85 / 55.98 / 57.41%; calc_diff 0.0; compute-bound (AI 943-1440 > ridge 562.5) |
| 08:52 | Inventory (AC-4/AC-6) | dump masked_m/expected_m, tile-rounding waste | tile-waste 6.25 / 3.91 / 1.56%; masked_m near-uniform (~7% spread); cap-pad 12.5 / 6.25 / 3.12% |
| 08:54 | Recipe (AC-6) | `DG_PRINT_CONFIGS=1` | identical recipe all shapes: swap_ab=1, block 128^3, cluster (1,2), 8 stages, smem 213804, num_sms 148, tcgen05; waves 24/45/86, last-wave util 52/16/92% |
| 09:00 | Headroom analysis (AC-4) [codex] | `/humanize:ask-codex` gpt-5.5:xhigh | verified ceiling ~51/58/58% < 65%; ranked levers (num_sms/pdl top, all <=~1.03x); gave NCU metric list |
| 09:08 | Knob sweep (AC-3/AC-7) | `bench/sweep.py` cold-L2, 11 combos x 3 shapes | best: M1024 pdl 1.0016x, M2048 cdims_mnk 1.0010x, M4096 cdims_mnk 1.0105x; num_sms reductions regress; tc90 neutral |
| 09:14 | Candidate v1 (AC-3) | per-shape pdl/compiled_dims dispatch; `./run.sh` | **EXIT 0** — 2/3 WIN (M1024 1.014x, M4096 1.010x), 0 regress; MFU 48.55/56.01/57.96%; calc_diff 0.0 |
| 09:18 | Confirm runs 2-3 (AC-5) | 2 more `./run.sh`, idle-checked | run2 exit 0 (2 WIN); run3 exit 0 (1 WIN, M4096 flickered neutral). M1024 WIN robust 3/3; 0 regress in all 9 shape-runs |
| 09:23 | NCU (AC-6) | Nsight Compute 2026.1.1, per-shape + full M1024 | tensor active 66/72/75%; DRAM 32/27/23% (compute-side); stalls = long_scoreboard (TMA) + barrier (sync); math throttle ~0; 1 CTA/SM smem-limited |
| 09:30 | Escalation decision (AC-4/AC-6) [codex] | `/humanize:ask-codex` gpt-5.5:xhigh | **NO_GO all 3 shapes**; no recoverable headroom for a custom kernel; preserve PARTIAL WIN |

## Key commands (reproducible)

```bash
export KDA_HARNESS_ROOT=/home/qinhaiyan/Kernel-Harness
export CUDA_VISIBLE_DEVICES=1 REMOTE_GPU_ID=1
cd "$KDA_HARNESS_ROOT/testbench/tasks/glm52/moe_down_proj_prefill"

# idle check (before every measurement)
nvidia-smi -i 1 --query-gpu=utilization.gpu,memory.used --format=csv,noheader

# authoritative gate (candidate)
./run.sh --candidate $CLAUDE_PROJECT_DIR/candidate/candidate.py

# workload inventory + recipe
$KDA_HARNESS_ROOT/.venv/bin/python $CLAUDE_PROJECT_DIR/docs/evidence/inventory.py
DG_PRINT_CONFIGS=1 $KDA_HARNESS_ROOT/.venv/bin/python /tmp/recipe_probe.py

# knob sweep (ranking)
$KDA_HARNESS_ROOT/.venv/bin/python $CLAUDE_PROJECT_DIR/bench/sweep.py

# NCU per shape (idle-checked)
ncu --kernel-name "regex:sm100_fp8_fp4_gemm_1d1d_impl" --launch-skip 8 --launch-count 1 \
    --metrics <tensor/dram/l2/occupancy/stall list> \
    $KDA_HARNESS_ROOT/.venv/bin/python $CLAUDE_PROJECT_DIR/docs/ncu/ncu_driver.py <M>
```

## GPU-idle discipline

Every baseline / candidate / sweep / NCU measurement was preceded by an idle-check of physical GPU 1
(0% utilization, 0 MiB used, no compute processes). No measurement was taken while GPU 1 showed
concurrent compute. The shared `o_proj_decode_hbm35` campaign was not active on GPU 1 during any run.
