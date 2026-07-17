# kernel-harness

Agent-ready **SGLang kernel-optimization tasks** for Kimi-K2.7, MiniMax-M3, and GLM-5.2.

> 中文版：[README.zh.md](README.zh.md)

Each task under [`testbench/tasks/`](testbench/tasks) is one `(operator, phase)` kernel
with its whole shape sweep bundled. You (or an agent) edit a single `solution.py` to
**match the real SGLang kernel's output** and **beat its latency** across the sweep. The
oracle and the speed baseline are the same thing: the real SGLang kernel, embedded in each
task's `reference.py`. If `solution.py` is correct and faster on every shape, it is a
candidate drop-in for that kernel.

> New here? This root README is the whole human onboarding path. Agents should read
> [`AGENTS.md`](AGENTS.md). The harness internals are in
> [`testbench/README.md`](testbench/README.md) and
> [`testbench/docs/HARNESS_DESIGN.md`](testbench/docs/HARNESS_DESIGN.md).

## 1. What this repo is

- **Tasks, not a framework.** 106 tasks (Kimi-K2.7 39 + MiniMax-M3 43 + GLM-5.2 24) across
  28 unique family names. No optimization-framework lock-in — any agent or human runs the same loop.
  GLM-5.2 is the exception: its 12 operators x 2 phases share one definition
  (`testbench/harness/glm52_ops.py`) and one command (`<task>/run.sh`); see AGENTS.md.
- **Real oracle.** `reference.py` *is* the SGLang production kernel. Correctness is judged
  against its output; efficiency against its CUPTI-measured latency.
- **Self-contained harness.** The evaluator, correctness checks, timing, and anti-gaming
  guards live in [`testbench/harness/`](testbench/harness) — torch-only, no external test
  dependency.
- **A win is a candidate, not a deployment.** `evaluate.py` proves a better kernel exists
  for that op's I/O contract; `integrate.py` then proves it drops into SGLang's dispatch.

## 2. Prerequisites

- A **GPU node** whose CUDA toolkit matches your SGLang checkout (tasks build/run real
  Blackwell/Hopper kernels).
- [`uv`](https://docs.astral.sh/uv/) installed.
- An **SGLang checkout** available to install from (default location: `../sglang`, i.e. a
  sibling of this repo). It supplies the matching torch / Triton / sgl-kernel / DeepGEMM /
  FlashInfer versions.
- *Optional:* a second checkout `../sglang-m3` containing the MiniMax sparse (DSA) stack —
  only needed for the MiniMax DSA tasks.

## 3. Setup

```bash
git clone git@github.com:qhy991/kernel-harness.git
cd kernel-harness

cp testbench/harness.env.example testbench/harness.env   # edit only if your checkouts differ
./testbench/setup_env.sh                                 # creates repo-local .venv via uv
.venv/bin/python testbench/bin/check_env.py              # verifies the environment
```

`harness.env` overrides checkout locations (`SGLANG_DIR`, `MM_M3_SGLANG_DIR`, `CUDA_HOME`);
relative paths resolve from the repo root. The only supported environment is the repo-local
`.venv` — do not reuse an unrelated conda/venv (native ABI mismatches invalidate correctness
and timing).

## 4. Pick a task

One task = one directory under `testbench/tasks/<model>/<name>/`. List the live inventory:

```bash
.venv/bin/python testbench/bin/inventory.py            # families and task counts per model
ls testbench/tasks/kimi_k27 testbench/tasks/minimax_m3 testbench/tasks/glm52
```

**Starter pack (recommended first targets).** Prefer memory-bound / fused ops, where
launch and fusion headroom is largest. Avoid leading with FP8 DeepGEMM GEMMs — those beat a
heavily hand-tuned Blackwell kernel and are intentionally hard.

- `testbench/tasks/kimi_k27/input_embedding_decode` — memory-bound gather; easiest start.
- `testbench/tasks/kimi_k27/q_a_layernorm_decode` — RMSNorm (also see the `rmsnorm` /
  `fused-add-rmsnorm` family).
- `testbench/tasks/kimi_k27/mla_qk_rope_decode` — RoPE apply; fusion opportunity.
- `testbench/tasks/kimi_k27/q_nope_absorb_bmm_decode` — small batched matmul (absorb BMM).
- `testbench/tasks/glm52/o_proj_decode` — GLM-5.2 block-FP8 DeepGEMM, memory-bound at decode.
  Ask any GLM-5.2 task what it is with `run.sh --describe`.

## 5. Dispatch one task to an agent

Optimize **one task per session**. Paste a prompt like this, with `<TASK_DIR>` set to a task
directory from §4:

```text
Optimize ONE kernel task only:
  <TASK_DIR>

Rules:
- Edit only <TASK_DIR>/solution.py
- Do not modify reference.py, workload tolerances, evaluate.py, or any harness code
- Read AGENTS.md and the task's task.json / reference.py first
- Loop: edit solution.py -> run evaluate.py -> read VERDICT_JSON -> iterate
- Quick check:  .venv/bin/python testbench/bin/evaluate.py <TASK_DIR> --max-workloads 1 --no-baseline
- Full gate:    .venv/bin/python testbench/bin/evaluate.py <TASK_DIR> --repeat 3
- Exit 0 = candidate WIN (correct AND faster on every shape). Then run integrate.py.
- Prefer real algorithmic/kernel improvements; do not game tolerances or timing.
```

**How you know it worked.** `evaluate.py` prints a per-shape table plus a machine-readable
block, and sets its exit code:

```
VERDICT_JSON_BEGIN
{"task":"...","correct":true,"win":true,"geomean_speedup":1.23,"min_speedup":1.08,"per_shape":[...]}
VERDICT_JSON_END
```

| Exit | Meaning |
|---|---|
| `0` | **WIN** — correct on every shape AND faster on every shape |
| `1` | correct, but not faster everywhere |
| `2` | incorrect / error / reward-hack / incomplete sweep |

`--repeat 3` gates the win on the worst-case margin across independent runs, so a small
speedup must survive run-to-run noise. For the tight inner loop there is also a fast,
**advisory** probe (`PYTHONPATH=testbench .venv/bin/python -m harness.profile <TASK_DIR>
--shape M`) — it never decides the verdict; `evaluate.py` does.

## 6. After a WIN

```bash
.venv/bin/python testbench/bin/integrate.py <TASK_DIR>   # exit 0 = verified SGLang drop-in
.venv/bin/python testbench/bin/migrate.py  <TASK_DIR>    # optional: emit a reversible SGLang patch
```

`integrate.py` hot-swaps the exact SGLang dispatch symbol with the candidate and drives a
**real SGLang forward** to confirm it is invoked, matches the unpatched output, and reverts
cleanly. `migrate.py` refuses unless integrate is green. A green integrate is still **not** a
deployment — remaining gates are full shape-space coverage, real-model accuracy, SGLang's own
unit tests, CUDA-graph safety, and AOT/JIT build + non-Blackwell fallback. See
[`testbench/README.md`](testbench/README.md) for details.

## 7. Pointers

- [`AGENTS.md`](AGENTS.md) — the agent-facing loop (read this if you are the agent).
- [`testbench/README.md`](testbench/README.md) — task contract, evaluate / integrate /
  migrate, anti-cheat guards, inventory.
- [`testbench/docs/HARNESS_DESIGN.md`](testbench/docs/HARNESS_DESIGN.md) — correctness,
  timing, and the two-tier fast/authoritative probe design.

## 8. Legacy proxy baselines (not the agent oracle)

An older framework-light **proxy** microbenchmark catalogue (pure `torch` / `flash_attn` /
`torch._scaled_mm`, no `sgl_kernel`/`deep_gemm`) lives in
[`docs/legacy_proxy_baselines.md`](docs/legacy_proxy_baselines.md). It is useful for shape
inspection and rough sanity checks, but its proxy timings are **not** valid SGLang speedup
denominators and it is **not** the optimization oracle. Do not start an optimization loop
there.

## License

Same as the SGLang repo you point to — Apache 2.0. The harness itself is BSD-3 so you can
drop it into internal repos without friction.
