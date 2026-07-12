# Kernel optimization agent guide

The agent-facing task suite is [`testbench/`](testbench/README.md). Each task compares an
editable `solution.py` against the **real SGLang kernel** in `reference.py` — that kernel is
both the correctness oracle and the latency baseline. The root `benchmarks/` directory is an
older proxy catalogue and is **not** the oracle; ignore it.

Optimize **one task per session**. All commands are run from the repo root.

## Environment

- Run on the target GPU node; the comparison uses the real SGLang kernels.
- Use only the repo-local `.venv` (`./testbench/setup_env.sh`).
- Set checkout paths via env vars or `testbench/harness.env` — never hardcode machine paths.
- Verify once before testing: `.venv/bin/python testbench/bin/check_env.py`.

## Optimization loop

1. Choose one directory `testbench/tasks/<model>/<task>/` (see starter tasks below).
2. Read its `task.json`, `definition.json`, `reference.py`, and the full `workload.jsonl` sweep.
3. Edit **only** that task's `solution.py`. It must define `run(<input names>)` with the same
   signature as `reference.py`'s `run`, returning the same output tensor(s). Import anything
   in the venv (Triton, CUDA/cpp_extension, CUTLASS, a different deep_gemm config, a fused path).
4. Quick check (one shape, correctness only, fast):
   `.venv/bin/python testbench/bin/evaluate.py <task> --max-workloads 1 --no-baseline`
5. Full gate (worst-case margin over independent runs):
   `.venv/bin/python testbench/bin/evaluate.py <task> --repeat 3`
6. Read the `VERDICT_JSON` block and the exit code: **0 = WIN** (correct AND faster on every
   shape), **1 = correct, not faster**, **2 = incorrect / error / incomplete**.
7. On a WIN, verify it is a real SGLang drop-in:
   `.venv/bin/python testbench/bin/integrate.py <task>` (exit 0 = verified).

## Fast probe vs. authoritative verdict

- **Authoritative (the gate):** `evaluate.py` — CUPTI device-kernel timing, cold L2, many reps,
  correctness, anti-cheat. It is the **only** thing that decides WIN/lose.
- **Advisory (exploration):** `PYTHONPATH=testbench .venv/bin/python -m harness.profile <task>
  --shape M` (or, from a script on `testbench`'s path, `from harness.profile import
  quick_latency, profile_task`). CUDA events, warm L2, ~20 reps,
  plus a roofline hint. Milliseconds. It runs a few µs above the CUPTI number and its noise
  floor swamps small deltas — **trust it for direction and large wins; confirm with
  `evaluate.py`.** It never gates a result.

## Starter tasks (prefer these first)

Memory-bound / fused ops have the most launch/fusion headroom. Do **not** start with FP8
DeepGEMM GEMMs — they beat a hand-tuned Blackwell kernel and are intentionally difficult.

- `testbench/tasks/kimi_k27/input_embedding_decode`
- `testbench/tasks/kimi_k27/q_a_layernorm_decode` (or any `rmsnorm` / `fused-add-rmsnorm` task)
- `testbench/tasks/kimi_k27/mla_qk_rope_decode`
- `testbench/tasks/kimi_k27/q_nope_absorb_bmm_decode`

List everything with `.venv/bin/python testbench/bin/inventory.py`.

## Do not edit (forbidden)

- `reference.py` — it is the oracle and the baseline.
- `workload.jsonl` tolerances or the shape sweep.
- `bin/evaluate.py`, `harness/` timing/correctness code, or `task.json`.
- Anything outside the chosen task's `solution.py`.

Do not game tolerances, timing, or reward-hack the evaluator (input aliasing, monkey-patched
timers, lazy outputs are all detected and rejected). Prefer real algorithmic/kernel wins.
