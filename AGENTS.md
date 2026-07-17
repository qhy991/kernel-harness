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

## GLM-5.2 uses a different contract — read this instead

The rest of this guide (`solution.py`, `definition.json`, `evaluate.py`) describes the
Kimi-K2.7 / MiniMax-M3 tasks. GLM-5.2's 24 tasks do **not** follow it. There, all 12
operators are defined exactly once, in
[`testbench/harness/glm52_ops.py`](testbench/harness/glm52_ops.py); a task directory
names which problem it is and nothing else, so it has nothing it could contradict.

```
testbench/tasks/glm52/<task>/
  task.json       operator + phase + the performance bar. Restating anything
                  glm52_ops owns is rejected with exit 3.
  problem.json    the whole problem definition, machine-readable. Generated —
                  read it, never edit it.
  workload.jsonl  the M sweep
  candidate.py    the default candidate: run(inputs: dict) -> output
  run.sh          the ONLY command
  README.md       generated; identical to `run.sh --describe`
```

```bash
T=testbench/tasks/glm52/o_proj_decode

$T/run.sh --describe          # what is this problem? (tensor table included)
$T/run.sh --describe --json   # ...the same, machine-readable (== problem.json)
$T/run.sh                     # the gate (warmup=3, repeat=10)
```

The whole ABI is `run(inputs: dict) -> output`. You do **not** have to edit the task
to be measured — `--candidate` takes any file or directory, anywhere on disk:

```bash
$T/run.sh --candidate ~/kernels/o_proj.py     # PyTorch, or Triton — both are just .py
$T/run.sh --candidate ~/kernels/o_proj_cu/    # a dir: candidate.py compiles its .cu
```

Triton needs nothing special (`@triton.jit`/`@triton.autotune` live in that .py). A
`.cu` goes through a candidate.py that `torch.utils.cpp_extension.load()`s it at
import time, so compilation stays outside the timed window. A bare `.cu` cannot be
passed: nothing in it says which `__global__` to launch, with what grid, or how the
inputs map to its arguments — `run(inputs)` is exactly that missing statement.
Worked, measured examples of both:
[`testbench/docs/GLM52_CANDIDATES.md`](testbench/docs/GLM52_CANDIDATES.md).

Exit codes: **0** correct and faster · **1** correct, not faster · **2** incorrect ·
**3** infrastructure or contract error. "Faster" means **at least one shape wins and
none regresses** — not every shape. A shape wins when the candidate is ahead on the
reading least favourable to it, regresses when it is behind on the most favourable
one, and is otherwise neutral. So `run()` **may branch on the shape and fall back to
`glm52_ops.reference` where it cannot win** — that is what SGLang itself does
(`deepgemm_w8a8_block_fp8_linear_with_fallback`), and the fallback shapes land as
neutral instead of vetoing the win. Falling back everywhere scores zero wins and
still fails. One command reports correctness, latency,
speedup and roofline reward, and persists the run under `runs/glm52/<task>/<run_id>/`.

Things that differ from the loop below, and will bite if you assume otherwise:

- `inputs` is a frozen dict from `glm52_ops.build_inputs`, shared byte-for-byte with
  the reference. Re-quantizing or re-seeding inside `run()` measures a different
  problem than the one the gate checked.
- `inputs["out"]`, where present, is **NaN-poisoned** before `run()` is called.
  Returning it unwritten fails — a no-op cannot inherit the reference's answer.
- Correctness is not allclose and not cosine. It is FlashMLA's three-layer check:
  anomaly positions, then per-element `abs OR rel`, then DeepGEMM's `calc_diff`.
  `--describe` prints the exact tolerances and where each came from.
- `--repeat 1` is a probe, not a verdict: noise is ±4%, so a candidate identical to
  the reference passes a `>1.0` gate a good fraction of the time. The default is
  **warmup=3, repeat=10**.
- The baseline is deep_gemm's f32-blockwise-scale path, which is **~1.6x slower than
  SGLang's production int32-ue8m0 dispatch**. A sub-1.6x speedup here does not mean
  you beat production. `--describe` repeats this warning per task.
- `integrate.py`, `definition.json` and `evaluate.py` do not apply to these tasks.

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

**GLM-5.2 (B200) — different contract, see below.** Ask the task itself which
problem it is and where the headroom is:

```bash
./testbench/tasks/glm52/o_proj_decode/run.sh --describe
```

Every decode shape is memory- or launch-bound (arithmetic intensity ~30 against an
fp8 ridge of 562); every prefill shape except the indexer and BMM ops is
compute-bound. Same operator, opposite bottleneck — they are separate tasks for
that reason. `--describe` prints the bound, and the `reward` column of a run is the
utilisation of whichever resource binds, so a task sitting at reward 0.24 has
headroom and one at 0.9 does not. That measurement is the routing signal; there is
no hand-maintained difficulty list to go stale.

List everything with `.venv/bin/python testbench/bin/inventory.py`.

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
