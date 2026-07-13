# kernel-harness / testbench

Standalone per-kernel optimization tasks for the Kimi-K2.7 / MiniMax-M3 / DeepSeek-V3.2
inventory (`../logs/kimi_minimax_20260710-070816/all_models_kernel_inventory.csv`).

**Inventory:** 82 tasks across 24 families (Kimi-K2.7 39 + MiniMax-M3 43, incl. 6 DSA
sparse-attention families on the `sglang-m3` build). `bin/integrate.py` has a drop-in
recipe for every family with a single rebindable sglang dispatch symbol; the fused
sparse-attention families (`dsa-decode-attn`, `dsa-prefill-attn`, `dsa-prefill-topk`)
have no isolated symbol and report `SKIP` rather than a faked recipe.

Each **task** is one `(op, phase)` kernel with its whole shape sweep bundled. An agent
optimizes `solution.py` to **match the sglang production kernel's output** (within a
per-shape tolerance) and **beat its latency** across the sweep.

> **Harness design & invariants:** [`docs/HARNESS_DESIGN.md`](docs/HARNESS_DESIGN.md) —
> the full contract (two-tier fast/authoritative testing, correctness & timing models,
> reward-hack defenses, the conservative win gate). Read it before changing the harness.

## Task contract

```
tasks/kimi_k27/<name>/
  definition.json   # axes (M sweep, K/N const), fp8 input schema, and the inline
                    #   reference = the sglang kernel (the correctness ORACLE)
  reference.py      # same code as definition.reference; the sglang baseline kernel
  solution.py       # starts identical to reference.py — the file an agent edits
  workload.jsonl    # one line per shape in the sweep + tolerance {atol, rtol, ratio}
  task.json         # metadata: op, phase, K, N, sweep, baseline_us, backend, tolerance
```

- **Oracle = the sglang kernel itself.** `reference.py` builds pre-quantized fp8 inputs
  in the exact deep_gemm/sglang layout (ue8m0, tma-aligned) and calls the real kernel.
  Its output is ground truth; its latency is the number to beat.
- **Correctness:** the harness runs reference + candidate on identical fresh inputs (10
  rounds) and checks `max_atol` / `max_rtol` / `required_matched_ratio` per workload.
- **Win bar:** candidate PASSES correctness on every shape AND `solution_us < baseline_us`.

## Environment

The one supported environment is the repository-local **`.venv` managed by `uv`**.
The selected SGLang checkout's `python/pyproject.toml` supplies matching torch, Triton,
SGLang-kernel, DeepGEMM, and FlashInfer versions.

```bash
cp testbench/harness.env.example testbench/harness.env  # optional path overrides
./testbench/setup_env.sh
.venv/bin/python testbench/bin/check_env.py
```

The setup script installs the selected local SGLang checkout and the standalone CUPTI
package. Do not reuse an unrelated conda/venv: native-package ABI mismatches can
invalidate correctness and timing.

## Agent contract (framework-neutral)

Any agent optimizes a kernel in **one loop**:

1. Edit `solution.py`. It must define `run(<input names>)` returning the output
   tensor(s) — same signature as `reference.py`'s `run` (e.g.
   `run(x_fp8, x_scale, w_fp8, w_scale)`). It may import anything in the venv:
   Triton, `torch.utils.cpp_extension` / CUDA, CUTLASS/CuTe, a different deep_gemm
   config, a fused path — whatever beats the baseline.
2. Evaluate and read the feedback:

```bash
.venv/bin/python testbench/bin/evaluate.py testbench/tasks/kimi_k27/o_proj_decode
```

Output = a per-shape table (correctness, sol_us, base_us, speedup, max_rel_err) plus
a machine-readable block your loop parses:

```
VERDICT_JSON_BEGIN
{"task":..., "correct":true, "win":false, "geomean_speedup":1.00, "min_speedup":..., "per_shape":[...]}
VERDICT_JSON_END
```

Exit code is the one-glance signal: **0 = WIN** (correct on every shape AND faster on
every shape), **1 = correct but not faster**, **2 = incorrect**. The sglang baseline
is measured once and cached in `<task>/.baseline_cache.json` (use
`--refresh-baseline` to remeasure). Useful flags: `--solution NAME`, `--iterations N`,
`--max-workloads N` (quick check), `--no-baseline` (correctness only).

Each task dir also ships a **self-contained `run.sh`**: `cd <task> && ./run.sh [flags]`
evaluates that folder without the caller knowing `bin/` (it forwards to `evaluate.py`).

### Fast probe while iterating (advisory, not the verdict)

`evaluate.py` above is the **authoritative** test — CUPTI, cold-L2, 100 reps, correctness,
reward-hack defenses — and the only thing that decides WIN/lose. It costs seconds. For the
tight inner loop there's a **fast, in-process advisory** profiler that answers "did that
rewrite move the needle, and which way should I push?" in milliseconds:

```bash
PYTHONPATH=testbench .venv/bin/python -m harness.profile \
  testbench/tasks/kimi_k27/o_proj_decode --shape 64
#   latency: median 7.25 us   min 5.86 us
#   roofline: 508 GB/s (6.4% HBM)  bound=... (memory vs compute)
```

```python
from harness.profile import quick_latency, profile_task
profile_task(task_dir, "solution.py", shape=64)   # {'median_us','min_us','roofline':{...}}
quick_latency(fn, setup)                           # lowest overhead, for an agent loop
```

It uses CUDA events (warm L2, ~20 reps) so it runs a few µs **above** the CUPTI number and
its noise floor swamps small deltas — **trust it for direction and large wins, use
`evaluate.py` to confirm anything fine.** It never gates a result.

That's the whole contract — no optimization-framework lock-in. Correctness is judged
against the sglang kernel's output; efficiency against its measured latency.

## Portability (no hardcoded paths)

All external locations (`SGLANG_DIR`, `CUDA_HOME`, `MM_M3_SGLANG_DIR`)
resolve through `bin/config.py`: **env var → `testbench/harness.env` → built-in default**.
`evaluate.py`, `integrate.py`, `migrate.py`, `emit_sglang.py`, `check_env.py`, and
`run.sh` all resolve paths through it, so a checkout on a new machine only needs env
vars (or a one-line `harness.env`) — no source edits.

The harness is deliberately **shared, not copied** per task. "Self-contained" means
the evaluator, timing, input construction, and correctness logic are owned here, while
each task exposes its own `run.sh` entrypoint.

## Optional drivers

- **Low-level:** `bin/run.sh <task> <solution.py|reference.py> <out> <iters>` runs one
  file through the harness; `bin/report.py <sol_out> <base_out>` diffs two runs.
- **Pre-flight:** `python3 bin/selftest.py [task_dir]` — stdlib-only structural check of
  the task dirs (contract files, embedded-reference identity, `run()` presence,
  workload/tolerance schema). Runs anywhere: no GPU, no venv.
- **Knowledge base:** `python3 bin/knowledge.py {query,add,lint}` — structured
  optimization recipes accumulated one entry per session (bottleneck, approaches tried,
  measured outcome, transferable lesson). Contract: [`knowledge/README.md`](knowledge/README.md).

## Timing & trustworthiness (read before believing a speedup)

The harness times with **CUPTI** (what `nsys` uses): the **median device-side kernel
duration** over N reps, with the **L2 cache cleared between iterations**, args cloned,
and all tensors pre-allocated (no `cudaMalloc` in the timed region). It **excludes CPU
launch overhead**. Config: `warmup=10`, `iterations` (default 50, set via
`--iterations`); the harness does not lock GPU clocks.

Consequences:
- The baseline is **measured live** each run (reference.py = the sglang kernel) and
  cached per shape in `<task>/.baseline_cache.json`. It is the only valid denominator.
- The `csv_wallclock_us_reference` in `task.json` (e.g. 57.6µs for O_proj decode) is a
  **launch-bound wall-clock** figure from the original microbench — for decode it is
  ~6× the real kernel time (~9µs). **Do not use it as the comparison denominator.**
- For FP8 GEMM tasks you are beating **DeepGEMM**, a heavily hand-tuned Blackwell
  kernel, not a strawman. These are intentionally difficult targets. Start autonomous
  searches with memory-bound or fused families such as RMSNorm, RoPE, embedding,
  MoE-combine, gate/top-k, or absorb BMM, where launch/fusion opportunities are larger.
  Treat every "win" as a candidate:
  - clocks are **not** locked by default → a few-% margin can be boost/thermal noise.
    Repeat the eval and require the margin to survive; consider locking clocks
    (needs node privileges) before trusting sub-10% wins.
  - tolerances are family-specific: exact for index/gather outputs; 99.9% matched with
    BF16, FP8, or attention-specific error bounds for floating outputs. Add real-model
    accuracy before proposing anything for SGLang.
  - these are fixed `(K,N)` + sampled `M`; a kernel that special-cases the sweep may
    lose on the broader shape space sglang dispatches. Validate wider before adoption.
- A fast + output-matching result from `evaluate.py` is **necessary, not sufficient**
  for sglang. It proves a better *kernel* exists for that op's I/O contract; it does
  **not** prove the candidate can replace sglang's kernel in place. That is what
  `bin/integrate.py` checks (below).

### Anti-cheat guards (what stops a fake WIN)

- **Input-aliasing**: for in-place / DPS ops (fused-add-rmsnorm, rope, moe-combine) the
  reference runs on a **clone** of the inputs (`harness/driver.py`), so a
  candidate can't return the reference-mutated buffer and pass at zero cost.
  `evaluate.py` adds an **independent second-layer probe** (`_alias_probe`): it runs
  reference and candidate on isolated clones of identical inputs and compares, catching
  aliasing even if the harness patch is ever lost.
- **Workload completeness**: on a full run (no `--max-workloads`) the candidate must be
  evaluated on the **entire** `workload.jsonl` sweep; a partial run (harness crash mid-
  sweep) is reported `INCOMPLETE` and is neither `correct` nor a `win`.
- **Baseline cache fingerprint**: `.baseline_cache.json` is keyed on
  `iters;ref.py-hash;sglang-commit` and only reused if it covers the full sweep — a
  stale/partial cache can't silently shrink or poison the denominator.
- **Migration double-gate**: `bin/migrate.py` refuses unless the candidate is BOTH an
  `evaluate.py` **WIN** (correct + faster on every shape) AND `integrate.py` green.


## Drop-in integration (`bin/integrate.py`)

`evaluate.py` compares two `run()` shims. But the candidate's interface is a benchmark
shim, not sglang's dispatch signature — e.g. the fp8 task's
`run(x_fp8, x_scale, w_fp8, w_scale)` vs sglang's
`w8a8_block_fp8_matmul_deepgemm(A, B, As, Bs, block, out_dtype)` (different arg order).
`integrate.py` closes that gap:

```
python bin/integrate.py <task_dir> [--solution solution.py]
# exit 0 = drop-in verified   1 = mismatch / not invoked   2 = no recipe (fused-only)
```

For a task it (1) hot-swaps the **exact sglang dispatch symbol** with a thin adapter
that reconciles sglang's call signature with the candidate's `run()`, (2) drives a
**real sglang module/function forward** (e.g. `deepgemm_w8a8_block_fp8_linear_with_fallback`,
which quantizes the activation inline then calls the patched matmul; or a real
`RMSNorm` / `SiluAndMul` module), and verifies **(a)** the candidate was actually
invoked inside sglang's own code path, **(b)** the real-forward output matches the
unpatched sglang output within the task tolerance, **(c)** the swap is fully reversible.
A green result is the deployable form of an `evaluate.py` win.

Recipes now cover **every non-fused task family**. Standalone call sites
(`fp8-linear-gemm`, `bf16-linear`, `rmsnorm`, `swiglu`, `embedding`, `router-gemm`,
`lm-head`, `act-fp8-quant`, `moe-gate`, `moe-combine`, `grouped-moe` masked +
contiguous) drive a real sglang module/kernel; fused-in-place ops
(`fused-add-rmsnorm`, `gemma-*`, `rope`) use a dedicated interface-exact task. The MLA
absorb `bmm` (global `torch.bmm`) now has a **shape-guarded** hook — only the
absorb-shaped call routes to the candidate, everything else passes through, and the
real `bmm` is restored while the candidate runs so a `torch.bmm`-based candidate can't
self-recurse. All verified — candidate invoked, 1.0000 match, restored; a
deliberately-wrong-output candidate fails (bf16-linear: 0.08), and a wrong-*signature*
candidate fails with `INTERFACE MISMATCH`. (Two families constructed via zero-weight
layers — `bf16-linear`, `lm-head` — fill the weight with random data before comparing,
so the check isn't vacuously `0 == 0`.)

### Interface-exact fused ops (why this matters for e2e later)

Some ops sglang runs as a welded, in-place unit — the main layernorm folds the
residual add into the norm (`fused_add_rmsnorm`), and MLA rope rotates q/k in place
(`apply_rope_with_cos_sin_cache_inplace`). There's no separate sub-op boundary to
swap, so we carve out a **dedicated task whose `run()` signature is a verbatim copy of
the sglang kernel** — e.g. `run(x, residual, weight, eps)` ≡
`fused_add_rmsnorm(x, residual, weight, eps)`, same in-place contract, same
`(normed, residual)` return (`eps` is a scalar input so the signature stays exact; the
mutated `residual` is renamed `residual_out` on the output side to avoid an
input/output name clash). `rope` is built the same way — `run(q, k, cos_sin_cache,
positions, *, is_neox, rope_dim, fused_args)` matches the kernel exactly, with the
keyword-only tail (which the validator doesn't bind to tensor inputs) preserved.

Because the interface is exact, `integrate.py` swaps the candidate in with an
**identity pass-through adapter** (no argument reshuffling). A real sglang forward
succeeding through that pass-through is the machine-check that the candidate is
callable as the sglang symbol — which is exactly the property that lets **today's
kernel-level test migrate to end-to-end testing later**: an e2e run just registers the
same `run` at the same dispatch point. A candidate whose signature drifts fails loudly
(`INTERFACE MISMATCH`), so staying sglang-compatible is enforced, not assumed.

Still out of scope even after a green integrate (the remaining deployment gates):
full **shape-space** coverage (not just the sweep), **real-model accuracy**, sglang's
own **unit tests** (`sgl-kernel/tests/*`), **CUDA-graph capture** safety, and
AOT/JIT build + non-Blackwell fallback.

## SGLang-native test + benchmark (`bin/emit_sglang.py`)

`evaluate.py` scores through the harness's own CUPTI runtime (fast inner loop). To make the **same
kernel comparable inside SGLang's own harness**, generate SGLang-native files:

```
python bin/emit_sglang.py <task_dir> [--dest <sgl-kernel dir>] [--stdout]
```

Writes `sgl-kernel/tests/test_<name>.py` (pytest + `torch.testing`-style tolerance,
`sys.exit(pytest.main([__file__]))` footer → auto-collected by `pytest tests/`) and
`sgl-kernel/benchmark/bench_<name>.py` (`triton.testing.perf_report` +
`do_bench_cudagraph(quantiles=[0.5,0.2,0.8])`, an `IS_CI` gate that shrinks to one
shape under the CI 60s/file budget, and a `speedup` provider = baseline_time /
candidate_time → auto-run by CI's `bench_*.py` loop). Both import the task's own
`reference.py` (real sglang kernel) and `solution.py` by path and use **seeded**
identical inputs (cloning fp8 scale tensors would drop their TMA-aligned stride).
No CI edit or manifest entry — SGLang discovers both by filename glob.

Verified: the generated test passes with the reference and fails with a wrong
candidate; the generated bench runs in SGLang's format and reports the speedup.

## Migrate a verified win into SGLang (`bin/migrate.py`)

The deployable step after a green `integrate.py`:

```
python bin/migrate.py <task_dir> [--solution solution.py] [--apply]
```

It (1) **refuses** unless `integrate.py` is green (only a machine-verified drop-in is
migratable), (2) emits a **reversible** sglang source patch that rebinds the exact
dispatch symbol through the candidate (the permanent form of integrate.py's swap) —
`results/migrate.patch` + `results/revert.patch`, repo-relative for `git apply`,
(3) with `--apply` verifies the round-trip (symbol routes to candidate, then restored
byte-exact), and (4) prints the remaining deployment gates as a loud checklist. Families
whose dispatch isn't a rebindable module-level symbol (`bmm` global `torch.bmm`,
`lm-head` in-body `torch.matmul`, `bf16-linear`/`embedding`/`rope` methods) are reported
as **no clean source site** rather than faked.

## Regenerate / extend

From the repository root, `.venv/bin/python testbench/gen_tasks.py` regenerates both
models from declarative `TaskSpec`s in `taskgen/families/` and kernel sources in
`recipes/`. It writes Kimi tasks to `tasks/kimi_k27/` and MiniMax tasks to
`tasks/minimax_m3/`. Workloads are grounded in each model's canonical config; new ops
are added one at a time and API-verified on GPU before their recipe is accepted.

## Coverage

The task directories and `task.json` files are authoritative; the README intentionally
does not maintain a second hand-edited family-status table. Print the live inventory:

```bash
.venv/bin/python testbench/bin/inventory.py
```

Current generated snapshot:
- Kimi-K2.7: 39 tasks in 12 families, including three MLA-attention tasks
  (`mla_prefill`, `mla_decode_seq2048`, `mla_decode_seq32768`).
- MiniMax-M3: 43 tasks in 18 families, including 11 tasks across six DSA families.
- Combined: 82 tasks across 24 unique family names.

MiniMax DSA tasks pin `MM_M3_SGLANG_DIR` because they need a checkout containing the
M3 sparse stack. They are available when that checkout passes `bin/check_env.py`; they
are not replaced with proxy baselines when it is absent.

### Known caveats
- `qa_kva_fused_*` uses the generic w8a8_block_fp8 path. sglang also has a
  `dsv3_fused_a_gemm` fused fast path for M≤16 (fuses act-quant+GEMM) that could be a
  separate, more faithful decode baseline later.
- Floating tolerances are assigned by family in `taskgen/spec.py`: BF16
  `(0.02, 0.01, 0.999)`, FP8 `(0.1, 0.05, 0.999)`, and attention
  `(0.03, 0.02, 0.999)` as `(atol, rtol, matched_ratio)`.
- **Index-producing / routing ops** (`moe-gate`) use an **exact-index oracle**
  (`atol=0, rtol=0, matched_ratio=1.0`) and output the integer indices only — the
  harness applies one tolerance per workload, so int-exact ids can't be co-judged with
  float weights. Verified with a negative test (rerouting one expert per token fails).
  Assumes no score ties (measure-zero for random floats); a same-set-different-order
  result fails by design.
- **Single canonical Kimi-K2.7 config** (one source of truth in `taskgen/config.py`):
  hidden=7168, q_lora=1536, kv_lora=512, num_heads=64, qk_nope=192, v_head=128,
  qk_rope=64, dense_inter/TP=2304, moe_inter/TP=256, vocab=163840, and MoE
  **n_routed_experts=384, topk=6, routed_scaling=2.872**. Both `router-gemm` (N=384)
  and `moe-gate` (384 experts) use it. `dsv3_router_gemm` accepts N∈{256,384} per
  sglang's dispatch (deepseek_v2.py:585); the flashinfer 256-only path is a
  DeepSeek/MiMo variant and is deliberately not used here.
