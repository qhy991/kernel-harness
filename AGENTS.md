# Kernel optimization agent guide

The agent-facing task suite is [`testbench/`](testbench/README.md). Each task compares an
editable `solution.py` against the **real SGLang kernel** in `reference.py` — that kernel is
both the correctness oracle and the latency baseline. The root `benchmarks/` directory is an
older proxy catalogue and is **not** the oracle; ignore it.

Optimize **one task per session**. All commands are run from the repo root.

Before editing, check the integration contract (drop-in vs fused-only):

```bash
python3 testbench/bin/integration_status.py <model>/<task>
```

See [`testbench/docs/AGENT_INTEGRATION.md`](testbench/docs/AGENT_INTEGRATION.md) for GLM-5.2 examples and failure modes.

## Environment

- Run on the target GPU node; the comparison uses the real SGLang kernels.
- Use only the repo-local `.venv` (`./testbench/setup_env.sh`).
- Set checkout paths via env vars or `testbench/harness.env` — never hardcode machine paths.
- Verify once before testing: `.venv/bin/python testbench/bin/check_env.py`.
- Structural pre-flight runs anywhere, no GPU/venv needed: `python3 testbench/bin/selftest.py`.

## Optimization loop

1. Choose one directory `testbench/tasks/<model>/<task>/` (see starter tasks below).
2. Read its `task.json`, `definition.json`, `reference.py`, and the full `workload.jsonl` sweep.
   Then check prior recipes: `python3 testbench/bin/knowledge.py query --task <model>/<task>`
   (broaden with `--family` / `--op` / `--bottleneck`; dead ends recorded there are real).
3. Edit **only** that task's `solution.py`. It must define `run(<input names>)` with the same
   signature as `reference.py`'s `run`, returning the same output tensor(s). Import anything
   in the venv (Triton, CUDA/cpp_extension, CUTLASS, a different deep_gemm config, a fused path).
4. Quick check (one shape, correctness only, fast):
   `.venv/bin/python testbench/bin/evaluate.py <task> --max-workloads 1 --no-baseline`
5. Full gate (worst-case margin over independent runs):
   `.venv/bin/python testbench/bin/evaluate.py <task> --repeat 3`
6. Read the `VERDICT_JSON` block and the exit code: **0 = WIN** (correct AND faster on every
   shape), **1 = correct, not faster**, **2 = incorrect / error / incomplete**.
7. On a WIN, verify it is a real SGLang drop-in **when the contract is `drop-in`**:
   `.venv/bin/python testbench/bin/integrate.py <task>` (exit 0 = verified).
   For `fused-only` tasks (e.g. `sparse-mla-decode`), skip integrate and record
   `integrate=no-recipe` in knowledge — the WIN is interface-exact at `run()`.
8. Optional structured closeout (evaluate + integrate + recommendation):
   `.venv/bin/python testbench/bin/agent_closeout.py <model>/<task> --repeat 3`
9. **Win or not**, end the session by recording one knowledge-base entry (next section).

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

**Kimi K27 (general):**

- `testbench/tasks/kimi_k27/input_embedding_decode`
- `testbench/tasks/kimi_k27/q_a_layernorm_decode` (or any `rmsnorm` / `fused-add-rmsnorm` task)
- `testbench/tasks/kimi_k27/mla_qk_rope_decode`
- `testbench/tasks/kimi_k27/q_nope_absorb_bmm_decode`

**GLM-5.2 (B200, good agent headroom — check contract first):**

- `testbench/tasks/glm52/routed_swiglu_prefill` — drop-in, fusion + buffer write wins
- `testbench/tasks/glm52/routed_down_decode` — drop-in, dispatch/sync wins
- `testbench/tasks/glm52/sparse_mla_decode` — fused-only; preserve device tensor scales
- Avoid first: `o_proj_decode`, `routed_gateup_*` (DeepGEMM floor)

List everything with `.venv/bin/python testbench/bin/inventory.py`
(or `inventory.py --headroom glm52` for agent routing by difficulty + integrate contract).

## Knowledge base (recipes)

`testbench/knowledge/` accumulates one structured entry per completed session: the
bottleneck diagnosis (with evidence), every approach tried — failures included, each
with a one-sentence "why" — the final measured result, and a transferable lesson.
Schema and honesty rules: [`testbench/knowledge/README.md`](testbench/knowledge/README.md).

- Query it before editing (loop step 2); write one entry when the session ends (step 8):
  draft the JSON, then `python3 testbench/bin/knowledge.py add <file>` (validates, installs).
- Every number must come from the final `evaluate.py` `VERDICT_JSON` — never from
  `harness.profile` or an estimate. Copy `hardware`/`stack` facts from `check_env.py`.
- Entries are append-only. Never edit or delete an existing entry; supersede it by
  adding a new one. A `no-win` entry with honest "why" lines is a valuable result.

## Do not edit (forbidden)

- `reference.py` — it is the oracle and the baseline.
- `workload.jsonl` tolerances or the shape sweep.
- `bin/evaluate.py`, `harness/` timing/correctness code, or `task.json`.
- Existing files under `testbench/knowledge/entries/` — append new entries only, via
  `knowledge.py add`.
- Anything else outside the chosen task's `solution.py`.

Do not game tolerances, timing, or reward-hack the evaluator (input aliasing, monkey-patched
timers, lazy outputs are all detected and rejected). Prefer real algorithmic/kernel wins.
