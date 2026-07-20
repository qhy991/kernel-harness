# Run Log — glm52 / o_proj_decode (Kernel-Harness, B200)

- Date (UTC): 2026-07-16
- Host: dry-vm-embraces-fin-03
- Harness root: /home/qinhaiyan/Kernel-Harness (KDA_HARNESS_ROOT); worktree Kernel-Harness/ symlink not materialized — resolved real path.
- Python: /home/qinhaiyan/Kernel-Harness/.venv/bin/python (conda env `sglang`); torch 2.11.0+cu130, cuda 13.0.
- GPU policy: pinned B200 GPU id 0. Exported `REMOTE_GPU_ID=0 CUDA_VISIBLE_DEVICES=0` for every measurement.

## GPU idle checks (GPU 0)
- Pre-work: util 0%, mem 0 MiB, no compute procs (all four B200s idle).
- Pre-probe recheck: util 0%, mem 0.
- Post-run recheck: util 0%, mem 0.
- Never waited on other GPUs; only GPU 0 used for baseline, probe, gate, integrate, closeout.

## Commands run (from $KDA_HARNESS_ROOT, .venv only, env REMOTE_GPU_ID=0 CUDA_VISIBLE_DEVICES=0)
1. Fast probe (current solution == reference):
   `.venv/bin/python testbench/bin/evaluate.py testbench/tasks/glm52/o_proj_decode --max-workloads 1 --no-baseline`
   → M=16 32.86 us, 3095 GB/s, 38.69% HBM peak, memory-bound (exit 1, no-baseline).
2. Baseline (both shapes, live):
   `.venv/bin/python testbench/bin/evaluate.py testbench/tasks/glm52/o_proj_decode --repeat 3`
   → solution==reference: M=16 base 32.91 us / M=32 base 33.26 us, ~38% HBM peak, exit 1 (correct, ~1.0). `.baseline_cache.json` created.
3. Edited ONLY `solution.py` `run()`: dispatch `deep_gemm.fp8_gemm_nt((x_fp8,x_scale),(w_fp8,w_scale),out,compiled_dims="nk")` + `import deep_gemm`.
4. Correctness/latency probe (both shapes):
   `.venv/bin/python testbench/bin/evaluate.py testbench/tasks/glm52/o_proj_decode --no-baseline`
   → M=16 30.56 us / M=32 31.33 us, correct (rel_err 0), 41.6% / 40.7% HBM peak.
5. Authoritative gate:
   `.venv/bin/python testbench/bin/evaluate.py testbench/tasks/glm52/o_proj_decode --repeat 3`
   → **exit 0 WIN**: sp_cons 1.078 (M=16) / 1.061 (M=32), geomean 1.0715, min_speedup_conservative 1.0612.
6. Drop-in:
   `.venv/bin/python testbench/bin/integrate.py testbench/tasks/glm52/o_proj_decode`
   → DROP-IN VERIFIED (invoked=1, match_ratio=1.0, shape_ok, restored, no nan/inf), exit 0.
7. Closeout:
   `.venv/bin/python testbench/bin/agent_closeout.py glm52/o_proj_decode --repeat 3 --owner qinhaiyan`
   → exit 0, win=true, drop_in_verified=true, geomean 1.071, min_speedup_conservative 1.0633.

## Scope check
- `git -C /home/qinhaiyan/Kernel-Harness status --porcelain -- testbench/tasks/glm52/o_proj_decode/`
  → ` M testbench/tasks/glm52/o_proj_decode/solution.py` (only file changed; +11/-3). reference.py / workload.jsonl / definition.json / harness infra untouched. M sweep not edited.
- No fabricated evidence; all numbers copied from live harness stdout (VERDICT_JSON / CLOSEOUT_JSON).

## Round 1 (2026-07-16): num_sms sweep (task6) + final re-close

GPU 0 idle rechecked (0% / 0 MiB) before the sweep. Sweep harness: temporarily made `run()` read a
`DG_NUM_SMS` env var (set_num_sms(n) around the `compiled_dims="nk"` call), then reverted to the clean
`nk` candidate (no env var) as the final artifact.

Commands (env REMOTE_GPU_ID=0 CUDA_VISIBLE_DEVICES=0, .venv):
- For n in {0,32,40,48,56,64,80,96,128,148}: `DG_NUM_SMS=$n .venv/bin/python testbench/bin/evaluate.py testbench/tasks/glm52/o_proj_decode --no-baseline` → per-shape correctness + candidate us (see docs/results.md sweep table).
- Best candidate confirmation: `DG_NUM_SMS=64 ... evaluate.py ... --repeat 3` → win, sp_cons M16 1.075 / M32 1.066, min 1.0657 (within noise of plain nk; M16 marginally worse). Reverted to plain nk.
- Final candidate (plain `compiled_dims="nk"`, DG_NUM_SMS unset):
  - `evaluate.py ... --repeat 3` → exit 0 WIN, min_speedup_conservative 1.062, M16 30.51us/sp_cons 1.076/41.67% peak, M32 31.3us/sp_cons 1.062/40.73% peak.
  - `integrate.py ...` → DROP-IN VERIFIED (invoked=1, match_ratio 1.0, shape_ok, restored, no nan/inf).
  - `agent_closeout.py glm52/o_proj_decode --repeat 3 --owner qinhaiyan` → exit 0, win=true, drop_in_verified=true, min_speedup_conservative 1.057 (run-to-run noise; > 1.0 both shapes).

Harness status at round-1 close:
- Task-scoped: `git -C /home/qinhaiyan/Kernel-Harness status --porcelain -- testbench/tasks/glm52/o_proj_decode/` → only `solution.py` (M), diff +14/-3.
- Global: also ` M testbench/tasks/glm52/dsa_prefill_attn/solution.py` — a DIFFERENT task's pre-existing, unrelated modification; not touched by this campaign. The AC-6 "only solution.py" claim is task-scoped, not a global-tree claim.

