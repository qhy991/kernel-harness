# Kernel optimization agent guide

The task suite is **GLM-5.2 on B200**: 12 operators × 2 phases = 24 tasks under
[`testbench/tasks/glm52/`](testbench/tasks/glm52/). Optimize **one task per session**.
All commands run from the repo root.

Everything else in this repo is retired and lives under [`legacy/`](legacy/README.md)
— the Kimi-K2.7 / MiniMax-M3 tasks, the `solution.py` + `definition.json` contract,
`evaluate.py`, `integrate.py`, and the proxy benchmark catalogue. None of it applies
here; don't copy patterns from it.

## The contract

All 12 operators are defined exactly once, in
[`testbench/harness/glm52_ops.py`](testbench/harness/glm52_ops.py). A task directory
names which problem it is and nothing else, so it has nothing it could contradict:

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

One command reports correctness, latency, speedup and roofline reward, and persists
the run under `runs/glm52/<task>/<run_id>/`.

**Exit codes:** `0` correct and faster · `1` correct, not faster · `2` incorrect ·
`3` infrastructure or contract error.

### Acceptance (not the gate)

After a per-op result, optionally measure what the candidate does to the **full
12-op layer budget** (same operator set as llm_flops / PR1 `allLatency`):

```bash
.venv/bin/python testbench/bin/accept_layer.py --M 32 --task o_proj_decode
.venv/bin/python testbench/bin/accept_layer.py --M 4096 --op o_proj \
    --candidate ~/kernels/o_proj.py
```

This swaps only the focused op onto its candidate (the other 11 stay on the
reference), reports layer total + end-to-end speedup, and exits 0 on a successful
measurement. It does **not** check correctness and does **not** replace `run.sh`.

## Candidates

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
inputs map to its arguments — `run(inputs)` is exactly that missing statement. Worked,
measured examples of both:
[`testbench/docs/GLM52_CANDIDATES.md`](testbench/docs/GLM52_CANDIDATES.md).

Setup that is not the kernel — JIT compilation, autotune warmup, building a CUDA graph
— belongs at **import time**, not inside `run()`. The harness imports your file once
and then calls `run()` under CUPTI, so import-time work is outside the measured window
and work inside `run()` is your latency.

## Things that will bite

- `inputs` is a frozen dict from `glm52_ops.build_inputs`, shared byte-for-byte with
  the reference. Re-quantizing or re-seeding inside `run()` measures a different
  problem than the one the gate checked. Changing *layout* (`.contiguous()`, `.view()`)
  is fine — that is your kernel's business, and it is timed.
- `inputs["out"]`, where present, is **NaN-poisoned** before `run()` is called.
  Returning it unwritten fails — a no-op cannot inherit the reference's answer.
- Correctness is not allclose and not cosine. It is FlashMLA's three-layer check:
  anomaly positions, then per-element `abs OR rel`, then DeepGEMM's `calc_diff`.
  Cosine and best-fit scale are reported as **diagnostics, never gates** — cosine ~1
  with a large calc_diff means a magnitude error and best_fit_scale is the factor to
  look for; a low cosine means an algorithm or layout error instead.
- **"Faster" means at least one shape wins and none regresses** — not every shape. So
  `run()` **may branch on the shape and fall back to `glm52_ops.reference` where it
  cannot win**; that is what SGLang itself does
  (`deepgemm_w8a8_block_fp8_linear_with_fallback`), and the fallback shapes land as
  neutral instead of vetoing the win. Falling back everywhere scores zero wins and
  still fails.
- `--repeat 1` is a probe, not a verdict: noise is ±4%, so a candidate identical to
  the reference passes a `>1.0` gate a good fraction of the time. The default is 10.
- The baseline is deep_gemm's f32-blockwise-scale path, which is **~1.6x slower than
  SGLang's production int32-ue8m0 dispatch**. A sub-1.6x speedup here does not mean
  you beat production. `--describe` repeats this warning per task.

## Where the headroom is

Ask the task, don't consult a list:

```bash
$T/run.sh --describe
```

Every decode shape is memory- or launch-bound (arithmetic intensity ~30 against an
fp8 ridge of 562); most prefill shapes are compute-bound. Same operator, opposite
bottleneck — which is why the phases are separate tasks. A run prints `AI`, `bound`,
`MFU`, `BW` and `reward` per shape, plus a `└ ref baseline` row with the reference's
own utilisation. That last row is the one to read: a candidate at reward 0.24 looks
like a 4x opportunity until the reference's own 0.24 says the op is simply at its
roof. `index_k_proj_decode` is the clearest case — AI 28 calls it memory-bound, but BW
0.12% says the time isn't going into moving data at all, so its leverage is fusion.

List everything with `.venv/bin/python testbench/bin/inventory.py`.

## Environment

- Run on the target GPU node; the comparison uses the real DeepGEMM / sgl_kernel calls.
- Use only the repo-local `.venv` (`./testbench/setup_env.sh`).
- Verify once before testing: `.venv/bin/python testbench/bin/check_env.py`.
- Structural pre-flight runs anywhere, no GPU/venv: `python3 testbench/bin/selftest.py`.
- Layer-swap acceptance (advisory, after a per-op result):
  `.venv/bin/python testbench/bin/accept_layer.py --M 32 --task <task>`.
- After changing `glm52_ops.py`, re-project it onto the tasks:
  `.venv/bin/python testbench/bin/sync_glm52_tasks.py` (`--check` for CI; it never
  overwrites `candidate.py`).

## Roofline-reward bench (folder of optimized ops → one CSV)

[`rewardbench/`](rewardbench/README.md) is a standalone tool that scores a **folder of
already-optimized GLM-5 operators** against the B200 roofline. Unlike the per-task gate
(`run.sh` / CUPTI correctness + speedup), it is **performance-only** and reports a
**bound-aware roofline-utilization reward ∈ [0,1]** per op: compute-util for
compute-bound ops, HBM-bandwidth-util for memory-bound ops (auto-classified by
arithmetic intensity). Two phase-specific scripts, prefill and decode:

```bash
cd rewardbench
python bench_GLM5_ops_prefill.py --kernels-dir <dir>   # one candidate folder OR many
python bench_GLM5_ops_decode.py  --kernels-dir <dir>
```

`--kernels-dir` accepts a parent folder of candidates or a single operator folder.
Each candidate's rows print to the terminal (timestamped) and append to a
`reward_bench.csv` inside that operator's own directory, plus an aggregate CSV. It
never gates a WIN — `run.sh` remains the gate; this is for tracking how close a
kernel is to the hardware ceiling. See its README for the CSV schema and the reward
design.

## Knowledge base (recipes)

`testbench/knowledge/` accumulates one structured entry per completed session: the
bottleneck diagnosis (with evidence), every approach tried — failures included, each
with a one-sentence "why" — the final measured result, and a transferable lesson.
Schema and honesty rules:
[`testbench/knowledge/README.md`](testbench/knowledge/README.md).

- Query it before editing; write one entry when the session ends:
  draft the JSON, then `python3 testbench/bin/knowledge.py add <file>`.
- Every number must come from the run's `RESULT_JSON` / `result.json` — never an
  estimate. Copy `hardware`/`stack` facts from `check_env.py`.
- A `win` needs `shapes_won >= 1` with `shapes_regressed == 0`, from `result.json`'s
  aggregate.
- Entries are append-only. Never edit or delete one; supersede it by adding a new one.
  The pre-consolidation `glm52--*` entries are **not reproducible** — see that README.

## Do not edit (forbidden)

- `testbench/harness/glm52_ops.py` — the operator definitions, the reference, and the
  tolerances. It is the oracle.
- `testbench/harness/evaluate_task.py`, `timing.py`, `reward_hack.py` — the runner,
  the timer, the anti-cheat.
- `task.json`, `problem.json`, `workload.jsonl`, `README.md` in a task directory —
  all generated; the runner exits 3 if they disagree with `glm52_ops`.
- Existing files under `testbench/knowledge/entries/` — append new entries only.
- Anything under `legacy/`.
- Anything outside the chosen task's `candidate.py` (or your own `--candidate` file).

Do not game the evaluator. Input aliasing, monkey-patched timers and lazy outputs are
detected and rejected; the shared output buffer is poisoned; correctness is re-checked
on fresh inputs after timing. Prefer real algorithmic or kernel wins.
