# Test Harness Design

The harness is the infrastructure agents use to test a kernel. Its whole job is to answer
two questions about a candidate `solution.py`, honestly and un-gameably:

1. **Is it correct?** — does it match the sglang production kernel's output on every shape?
2. **Is it faster?** — does it beat that same kernel's measured device-side latency?

This document is the contract. It describes what exists, why each piece is shaped the way
it is, and the invariants any change must preserve. It is grounded in the code under
`testbench/harness/` and `testbench/bin/`, not in intentions.

---

## 1. Two-tier testing: fast probe vs. authoritative verdict

There are **two separate systems** with different jobs. They are not redundant and neither
replaces the other.

| | **Fast (advisory)** | **Slow (authoritative)** |
|---|---|---|
| Entry | `harness/profile.py` (in-process API + `PYTHONPATH=testbench python -m harness.profile`) | `bin/evaluate.py` (and each task's `run.sh`) |
| Timer | CUDA events, warm L2, ~20 reps | CUPTI device-kernel, cold L2, ≥50 reps, subprocess |
| Correctness | none | allclose/matched-ratio per shape |
| Cost | milliseconds | seconds+ |
| Output | latency + roofline hint | WIN / correct-not-faster / incorrect verdict |
| Authority | **none — never gates** | **the only win/lose gate** |

**The design rule: fast never overrides slow.** They can legitimately disagree — warm vs
cold L2 flips the ranking of memory-bound kernels, and few-rep event variance swamps a real
few-% delta at ~µs latencies. The fast path also runs a few µs *above* the slow path because
CUDA events include kernel-launch overhead that CUPTI strips out. So: use `profile` to
explore cheaply and read *direction*; use `evaluate` to produce the number of record.

Everything from §3 onward describes the authoritative path. The fast path is §7.

---

## 2. What a task is (the on-disk contract)

A task is a self-describing directory. Every file has one job:

```
tasks/<model>/<task_name>/
  definition.json    axes, input/output tensor specs, and the reference SOURCE (embedded)
  reference.py       the sglang production kernel — correctness oracle AND latency baseline
  solution.py        the candidate the agent edits; must define run() with reference's signature
  workload.jsonl     one line per shape: {uuid, axes:{M:..}, inputs, tolerance}
  task.json          human/agent metadata: model, op, family, sweep, tolerance, interface_exact,
                     optional sglang_dir (pin a specific sglang build), baseline provenance
  run.sh             self-contained entrypoint -> bin/evaluate.py (authoritative)
  .baseline_cache.json   cached per-shape baseline latency (git-ignored)
```

`definition.json` keys: `name, hf_id, family, description, axes, custom_inputs_entrypoint,
inputs, outputs, reference`, plus optional `flops_expr` / `performance_model` /
`workload_metrics` for advisory metrics.

**Axes** are the shape variables. Three types (`harness/inputs.py:resolve_axes`):
- `const` — fixed (e.g. hidden size `H=7168`).
- `var` — swept per workload (e.g. `M`, the token count); its value comes from the workload line.
- `expr` — a safe arithmetic expression over other axes (e.g. `I = I2//2`), resolved to a
  fixpoint. Only `+ - * / // % ** ()` are allowed (AST-walked, no `eval`).

**The reference is the source of truth for both correctness and speed.** There is no separate
"golden output" file and no separate "baseline number" — `reference.py` *is* the sglang kernel,
so the oracle and the denominator are the same code path. This is what makes a win meaningful:
beating the reference means beating what sglang actually runs.

**Interface-exactness.** Many tasks set `interface_exact: true` and make `run()`'s signature
and in-place/return contract a verbatim copy of the sglang kernel call. A winning candidate is
then a symbol-for-symbol drop-in, and the same test can later migrate to an end-to-end sglang
forward. `bin/integrate.py` verifies this drop-in property per family.

---

## 3. The runtime (`harness/`) — self-contained, torch-only

The harness owns its testing method. It does **not** depend on any external optimization or
benchmark framework; its only runtime deps are `torch` and the optional standalone `cupti`
package. Files:

| File | Responsibility |
|---|---|
| `inputs.py` | axis resolution, calling the recipe's `get_inputs`, positional input assembly, output normalization (tuple/dict/tensor → positional tensor list) |
| `dtypes.py` | dtype-string ↔ `torch.dtype` map |
| `correctness.py` | allclose-style error stats + matched-ratio, inf/nan sanity, optional max-error cap |
| `timing.py` | CUPTI device-kernel timing (primary) + CUDA-events (fallback); L2 flush, arg clone |
| `reward_hack.py` | timer-patch detection, lazy-output rejection, thread-injection check |
| `metrics.py` | read-only diagnostic metrics (GEMM / sparse MLA / routed experts / SwiGLU); never gates WIN |
| `driver.py` | subprocess entrypoint: for one task + one candidate, evaluate every workload → `traces.json` |
| `profile.py` | fast advisory profiler (§7) — separate from the verdict path |

### 3.1 Correctness model (`correctness.py`)

Per output tensor, in fp32:
```
abs_err = |cand - ref|
matched = fraction of elements with abs_err <= atol + rtol*|ref|
correct = matched >= required_matched_ratio   (and no inf/nan, and optional max_error_cap)
```
Default tolerance: `max_atol=0.02, max_rtol=0.01, required_matched_ratio=0.999`. Per-task
tolerances come from `task.json`/`workload.jsonl` and override the default (e.g. attention
and fp8 paths are looser; exact-index oracles use `atol=0, rtol=0, ratio=1.0`). inf/nan in
either tensor is an automatic fail.

### 3.2 Timing model (`timing.py`)

Primary is **CUPTI activity tracing** — the same device-side kernel duration `nsys` reports:
- **median** device-kernel time over `rep` iterations (default 50 via evaluate, 100 in the raw API),
- **L2 cache flushed** before each iteration (cold-cache — the honest number for memory-bound ops),
- **args cloned** each iteration (no cross-iteration contamination),
- all tensors pre-allocated (no `cudaMalloc` in the timed region),
- **CPU launch overhead excluded**.

If the `cupti` package is unavailable it falls back to `torch.cuda.Event` timing. `time_runnable`
returns the median in ms.

### 3.3 Reward-hack defenses (`reward_hack.py` + driver + evaluate)

Five independent layers, because a "faster" number is worthless if it was faked:

1. **Timer-patch detection.** `id(torch.cuda.Event.elapsed_time)` is captured at module load
   (before any candidate code) and re-checked before running the candidate AND again after
   timing — a patch installed inside `run()` itself is also rejected.
2. **Lazy/proxy-output rejection.** Outputs must be exact `torch.Tensor` (strict `type()` check),
   so a FakeTensor/lazy proxy that defers the actual compute past the timed region is rejected.
3. **Input-aliasing defense — layer 1 (driver).** The reference runs on a *private clone* of the
   inputs, so a candidate cannot see reference-mutated buffers, and a candidate that just returns
   its input buffer (instead of computing) will mismatch the oracle.
4. **Input-aliasing defense — layer 2 (`evaluate._alias_probe`).** An independent subprocess
   re-derives one input set, runs reference and candidate each on their own isolated clone, and
   compares — a redundant second check for in-place / interface-exact families. No false positives
   (unlike a magnitude heuristic, which would flag legitimate residual-add/rope outputs).
5. **Post-timing re-verification (driver).** The timed iterations are not output-checked, so
   after timing the driver runs the candidate once more on a fresh clone and re-compares against
   the oracle. A stateful candidate that computes honestly while checked but returns garbage
   fast while timed (e.g. by counting invocations) is rejected as a reward hack.

### 3.4 Driver flow (`driver.py`, per workload)

```
resolve axes → build_inputs (via reference.get_inputs)
  → reference.run() on clone A        (oracle; isolated so in-place edits can't leak)
  → check_monkey_patch()
  → candidate.run() on clone B
  → normalize_outputs + check_lazy_outputs
  → per-output shape + count check, then compute_error_stats vs tolerance
  → if correct: time_runnable(candidate) with a fresh clone per iteration
  → post-timing: check_monkey_patch() again + one more candidate run re-compared
    against the oracle (rejects timers patched inside run() and go-lazy-under-timing)
  → emit trace {status, correctness (+ optional extras), performance.latency_ms
                 (+ advisory metrics / us_per_token), log}
```
Status is one of `PASSED / INCORRECT / RUNTIME_ERROR / REWARD_HACK`. All exceptions are caught
and become a trace with a log — a crash on one shape never aborts the sweep.

**Advisory metrics.** When a task declares `performance_model`, the driver attaches
`performance.metrics` (from `harness/metrics.py`, fed by reference inputs + measured
latency) and optional `correctness.extras` (mean/p99 abs err, cosine distance). These
fields flow into `VERDICT_JSON.per_shape` for agent feedback. They **never** change the
WIN gate (full sweep, NaN/Inf ban, matched-ratio, conservative speedup).

---

## 4. The evaluator (`bin/evaluate.py`) — the verdict

`evaluate.py` orchestrates the driver, measures the baseline, and produces the verdict. It runs
the driver in a **subprocess** under an env that puts the correct sglang checkout's `python/`
first on `PYTHONPATH` (`_env` + `_sglang_dir_for`), so a task can pin its own sglang build
(e.g. DSA tasks need the `amd_add_m3` tree) via `task.json`'s `sglang_dir`.

### 4.1 Verdict and exit code

```
0 = WIN    correct on every shape AND faster on every shape (worst-case margin > 1)
1 = CORRECT but not faster everywhere
2 = INCORRECT / error / reward-hack / incomplete sweep
```
Output is a human table plus a machine-readable block between `VERDICT_JSON_BEGIN`/`END` for the
agent loop to parse (`correct`, `win`, `geomean_speedup`, `min_speedup`, per-shape rows).

### 4.2 Conservative win gate

The win gates on the **worst-case** margin, not the headline. With `--repeat K>1`, candidate and
baseline are each re-timed over K independent processes, and:
```
speedup_conservative = baseline_best / candidate_worst   (per shape)
win = correct AND min over shapes(speedup_conservative) > 1
```
So a marginal speedup is judged against run-to-run noise — a candidate only wins if its *worst*
run still beats the baseline's *best* run on *every* shape. At `K=1` this reduces to the plain
`min_speedup > 1` rule.

### 4.3 Baseline caching and its guards

The baseline (reference latency per shape) is cached in `.baseline_cache.json`, keyed by a
**fingerprint** = `iters ; reference.py hash ; sglang commit`. The cache is only trusted when it
matches the fingerprint AND covers the full sweep. Hard-won invariants (each guards a real bug):
- **Never cache an empty/failed baseline** — a `{}` denominator poisons every later run (null speedups).
- **Never cache a partial sweep** — it would silently shrink the comparison denominator.
- **Path-hashed /tmp scratch** (`_tmp_base`) — two kits with same-named leaf dirs must not share a
  scratch tree, or a stale shape leaks in via the recursive `traces.json` glob.
- **Incomplete sweep ⇒ not correct, not a win** — a mid-sweep harness crash must not read as a pass
  on the shapes that happened to complete.

---

## 5. Invariants (do not break these)

- **The reference is oracle and denominator.** Don't add a separate golden file or a separate
  baseline number; correctness and speed must both come from `reference.py`.
- **Reference and candidate never share input buffers.** Each runs on its own clone.
- **The authoritative timer is CUPTI, cold-L2, median-over-reps.** Don't swap in warm-cache or
  wall-clock for the verdict.
- **The win gate is worst-case, not average.** Don't relax it to geomean or median.
- **A candidate that didn't run the full sweep is not correct.** Don't count partial passes.
- **Fast profiling is advisory.** Don't let `profile.py` feed into the verdict.
- **The runtime stays torch-only.** Don't reintroduce a dependency on an external test framework.

---

## 6. How an agent uses it

```bash
# From the repository root. Explore cheaply while iterating (advisory, ms):
PYTHONPATH=testbench .venv/bin/python -m harness.profile \
  testbench/tasks/kimi_k27/o_proj_decode --shape 64

# Authoritative verdict (seconds):
.venv/bin/python testbench/bin/evaluate.py testbench/tasks/kimi_k27/o_proj_decode
#   ...or self-contained:  cd testbench/tasks/kimi_k27/o_proj_decode && ./run.sh

# Noise-aware gate for a marginal win:
.venv/bin/python testbench/bin/evaluate.py <task> --repeat 3

# Useful flags: --solution NAME  --iterations N  --max-workloads N
#               --refresh-baseline  --no-baseline (correctness only)
```
Parse the `VERDICT_JSON` block for `correct`/`win`/speedups; the exit code is the one-glance signal.

---

## 7. Fast advisory profiler (`harness/profile.py`)

In-process, low-overhead, **never the verdict** — a compass for the inner loop.

- `quick_latency(fn, setup=None, reps=20, warmup=5, cold_l2=False)` → `{median_us, min_us}` via
  CUDA events. Lowest overhead; for an agent's Python loop.
- `profile_task(task_dir, solution, shape=None, ...)` → loads the recipe once, profiles one shape's
  `run()`, returns latency + a roofline hint.
- `PYTHONPATH=testbench .venv/bin/python -m harness.profile <task_dir> [--shape M] [--reps N] [--cold-l2] [--sglang-python DIR]`
  for manual probes (must set `PYTHONPATH=testbench` from the repo root).

**Roofline hint.** From the actual tensor traffic (input reads + output writes) and the measured
latency: achieved GB/s vs B200 HBM peak (~8 TB/s). With an optional `flops_expr` in
`definition.json` (e.g. `"2*M*K*N"`) it also reports TFLOP/s, arithmetic intensity, and the
**ridge point** (FP_peak / HBM_peak) — bound = compute if AI ≥ ridge, else memory. Correctly
flags small-M decode GEMMs as weight-memory-bound (AI ≈ 2·M < ridge).

**Honest limits (documented in-module):** event timing includes a ~few-µs launch-overhead floor
that CUPTI removes, and that floor compresses small deltas; bytes-only roofline can't tell
compute- from launch-bound without `flops_expr`. Trust it for direction and large wins; confirm
fine differences with `evaluate.py`.

---

## 8. Portability

External locations (`SGLANG_DIR`, `CUDA_HOME`, `MM_M3_SGLANG_DIR`) resolve through
`bin/config.py`: **env var → `testbench/harness.env` → built-in default**. The venv is
always the repo-local `.venv` (exported by `config.py` but deliberately not overridable —
one supported environment). A checkout on a new machine needs only env vars or a one-line
`harness.env` — no source edits. The harness runtime is
shared (not copied per task); "self-contained" means the evaluator, timing, input construction,
and correctness logic are owned here, and each task exposes its own `run.sh`.
