# Vendored winning candidate — glm52 / o_proj_decode

## Why this directory exists

The optimized kernel for this campaign is a change to a single file:
`$KDA_HARNESS_ROOT/testbench/tasks/glm52/o_proj_decode/solution.py`
(default `KDA_HARNESS_ROOT=/home/qinhaiyan/Kernel-Harness`).

That file lives in the **sibling Kernel-Harness checkout**, reached from this repo through the
repo-root `Kernel-Harness/` symlink — which is **not tracked** by this repo. Consequently a commit
in this repo does not, by itself, carry the executable change: a clean checkout / PR application
would still run the old solution while the docs claim a WIN.

To make the executable change **committed and reproducible from this repo**, the exact winning
`solution.py` is vendored here alongside a deterministic apply script.

## Contents

- `solution.py` — byte-identical copy of the winning harness `solution.py` (verify with `sha256sum`).
- `apply.sh` — copies `solution.py` to the harness task path and verifies the copy byte-for-byte.

## The change (one edit to `run()`)

`run()` dispatches `deep_gemm.fp8_gemm_nt((x_fp8, x_scale), (w_fp8, w_scale), out, compiled_dims="nk")`
directly — baking N=6144, K=16384 as compile-time template constants (the production sglang wrapper
leaves them dynamic, `compiled_dims=""`), keeping M dynamic (drop-in safe), passing the packed UE8M0
int32 scales through unchanged. Everything else (imports, `get_inputs`) matches `reference.py`.

## Apply + reproduce

```bash
# (custom checkout only) export the harness root FIRST so the same value governs
# both apply.sh and the gate commands below. Default: /home/qinhaiyan/Kernel-Harness
export KDA_HARNESS_ROOT=/path/to/Kernel-Harness

# 1) place the candidate at the harness gate path
./apply.sh                      # apply.sh prints the resolved harness root + gate commands

# 2) run the authoritative gate (idle B200 GPU 0, harness .venv)
cd "${KDA_HARNESS_ROOT:-/home/qinhaiyan/Kernel-Harness}"
export REMOTE_GPU_ID=0 CUDA_VISIBLE_DEVICES=0
.venv/bin/python testbench/bin/evaluate.py testbench/tasks/glm52/o_proj_decode --repeat 3
.venv/bin/python testbench/bin/integrate.py testbench/tasks/glm52/o_proj_decode
.venv/bin/python testbench/bin/agent_closeout.py glm52/o_proj_decode --repeat 3 --owner qinhaiyan
```

## Verified result (see ../docs/results.md, ../docs/run_log.md)

`evaluate.py --repeat 3` → exit 0 WIN; min_speedup_conservative ≈ 1.06 (M=16 ≈ 1.076 / M=32 ≈ 1.062);
correct (matched_ratio 1.0, rel_err 0); `integrate.py` DROP-IN VERIFIED; `agent_closeout.py` exit 0.

## Provenance

The `reference.py`, `workload.jsonl`, `definition.json`, and harness infrastructure are unchanged and
are NOT vendored (only the single editable `solution.py` is). This vendored copy is the source of truth
for the executable change committed to this repo; the live harness working tree currently has it applied.
