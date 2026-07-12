# kernel-harness / testbench

Standalone per-kernel optimization tasks for the Kimi-K2.7 / MiniMax-M3 / DeepSeek-V3.2
inventory (`../logs/kimi_minimax_20260710-070816/all_models_kernel_inventory.csv`).

Each **task** is one `(op, phase)` kernel with its whole shape sweep bundled. An agent
optimizes `solution.py` to **match the sglang production kernel's output** (within a
per-shape tolerance) and **beat its latency** across the sweep.

## The contract (FlashInfer-Bench / sol-execbench format)

```
tasks/kimi_k27/<name>/
  definition.json   # axes (M sweep, K/N const), fp8 input schema, and the inline
                    #   reference = the sglang kernel (the correctness ORACLE)
  reference.py      # same code as definition.reference; the sglang baseline kernel
  solution.py       # starts identical to reference.py — the file an agent edits
  workload.jsonl    # one line per shape in the sweep + tolerance {atol, rtol, ratio}
  task.json         # metadata: op, phase, K, N, sweep, baseline_us, backend, tolerance
  kersor-note.txt   # ready-to-paste KerSor drive command (Correctness/Benchmark/Baseline)
```

- **Oracle = the sglang kernel itself.** `reference.py` builds pre-quantized fp8 inputs
  in the exact deep_gemm/sglang layout (ue8m0, tma-aligned) and calls the real kernel.
  Its output is ground truth; its latency is the number to beat.
- **Correctness:** the harness runs reference + candidate on identical fresh inputs (10
  rounds) and checks `max_atol` / `max_rtol` / `required_matched_ratio` per workload.
- **Win bar:** candidate PASSES correctness on every shape AND `solution_us < baseline_us`.

## Environment

Everything runs in the **unified sglang venv** `../.venv` (torch 2.11, full sglang +
sgl_kernel + deep_gemm + flashinfer) into which `sol-execbench` was installed
`--no-deps` (plus `cupti-python`). `bin/run.sh` sets `PATH`/`PYTHONPATH` for you.

## Agent contract (framework-neutral)

Any agent optimizes a kernel in **one loop**:

1. Edit `solution.py`. It must define `run(<input names>)` returning the output
   tensor(s) — same signature as `reference.py`'s `run` (e.g.
   `run(x_fp8, x_scale, w_fp8, w_scale)`). It may import anything in the venv:
   Triton, `torch.utils.cpp_extension` / CUDA, CUTLASS/CuTe, a different deep_gemm
   config, a fused path — whatever beats the baseline.
2. Evaluate and read the feedback:

```bash
python testbench/bin/evaluate.py tasks/kimi_k27/o_proj_decode
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

That's the whole contract — no KerSor, no framework lock-in. Correctness is judged
against the sglang kernel's output; efficiency against its measured latency.

## Portability (no hardcoded paths)

All external locations (`VENV`, `SOLEXEC`, `SGLANG_DIR`, `CUDA_HOME`, `MM_M3_SGLANG_DIR`)
resolve through `bin/config.py`: **env var → `testbench/harness.env` → built-in default**.
`evaluate.py`, `run.sh`, `report.py`, and `gen_tasks.py` all import it, so a checkout on
a new machine only needs env vars (or a one-line `harness.env`) — no source edits.

Note: the harness is deliberately **shared, not vendored** per folder. "Self-contained"
means each task dir declares its own `run.sh` entrypoint and resolves the shared drivers
in `bin/` — not that it copies SOL-ExecBench 68×. One driver set, many self-describing tasks.

## Optional drivers

- **Low-level:** `bin/run.sh <task> <solution.py|reference.py> <out> <iters>` runs one
  file through the harness; `bin/report.py <sol_out> <base_out>` diffs two runs.
- **KerSor:** each task ships a `kersor-note.txt` for
  `/kersor:optimize <task_dir> --note "$(cat <task_dir>/kersor-note.txt)"`. Entirely
  optional — the task dirs and `evaluate.py` don't depend on it.

## Timing & trustworthiness (read before believing a speedup)

The harness times with **CUPTI** (what `nsys` uses): the **median device-side kernel
duration** over N reps, with the **L2 cache cleared between iterations**, args cloned,
and all tensors pre-allocated (no `cudaMalloc` in the timed region). It **excludes CPU
launch overhead**. Config: `warmup=10`, `iterations` (default 50, set via
`--iterations`), `lock_clocks=false`.

Consequences:
- The baseline is **measured live** each run (reference.py = the sglang kernel) and
  cached per shape in `<task>/.baseline_cache.json`. It is the only valid denominator.
- The `csv_wallclock_us_reference` in `task.json` (e.g. 57.6µs for O_proj decode) is a
  **launch-bound wall-clock** figure from the original microbench — for decode it is
  ~6× the real kernel time (~9µs). **Do not use it as the comparison denominator.**
- You are beating **deep_gemm**, a heavily hand-tuned Blackwell FP8 GEMM, not a
  strawman. Treat a "win" as a *candidate*:
  - clocks are **not** locked by default → a few-% margin can be boost/thermal noise.
    Repeat the eval and require the margin to survive; consider locking clocks
    (needs node privileges) before trusting sub-10% wins.
  - correctness here is loose (atol 0.1 / rtol 0.05 / 98% on random inputs) — tighten
    per-op and add real-model accuracy before proposing anything for sglang.
  - these are fixed `(K,N)` + sampled `M`; a kernel that special-cases the sweep may
    lose on the broader shape space sglang dispatches. Validate wider before adoption.
- A fast + output-matching result from `evaluate.py` is **necessary, not sufficient**
  for sglang. It proves a better *kernel* exists for that op's I/O contract; it does
  **not** prove the candidate can replace sglang's kernel in place. That is what
  `bin/integrate.py` checks (below).

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

Recipes now cover **every task family** (all 17). Standalone call sites
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

`evaluate.py` scores through NVIDIA SOL-ExecBench (fast inner loop). To make the **same
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

`python gen_tasks.py` regenerates every family for both models from its
`recipes/<family>.py`, writing Kimi tasks to `tasks/kimi_k27/` and MiniMax tasks to
`tasks/minimax_m3/`. Each family is a table + an `emit_*()` in `gen_tasks.py`, keyed by
`(model, family)`; the fp8 tasks are byte-stable across regeneration. Workloads are
grounded in each model's real config (Kimi from the repo's fp8 shapes; MiniMax-M3 from
the HF `text_config` via `bench_minimax_m3.py`) — new ops are added one at a time,
API-verified on-GPU before the recipe is written.

## Coverage

Kimi-K2.7 coverage (tasks generated + evaluate-passed):

"Done" = tasks generated + smoke-passed via `evaluate.py`. "drop-in" = has a verified
`bin/integrate.py` recipe (candidate replaces the real sglang kernel in a live forward).

| family | tasks | recipe | evaluate | drop-in |
|---|---|---|---|---|
| fp8-linear-gemm | 15 | `recipes/fp8_linear_gemm.py` (deep_gemm w8a8_block_fp8) | **✅** | **✅** |
| rmsnorm | 4 | `recipes/rmsnorm.py` (sgl_kernel.rmsnorm) | **✅** | **✅** |
| fused-add-rmsnorm | 2 | `recipes/fused_add_rmsnorm.py` (sgl_kernel.fused_add_rmsnorm) | **✅** | **✅ interface-exact** |
| rope | 2 | `recipes/rope.py` (apply_rope_with_cos_sin_cache_inplace) | **✅** | **✅ interface-exact** |
| swiglu | 4 | `recipes/swiglu.py` (sgl_kernel.silu_and_mul) | **✅** | **✅** |
| embedding | 2 | `recipes/embedding.py` (F.embedding) | **✅** | **✅** (scoped `quant_method.embedding`) |
| router-gemm (decode) | 1 | `recipes/kimi_router_gemm.py` (sgl_kernel.dsv3_router_gemm, M≤16, N=256) | **✅** | todo (needs MoE-gate driver) |
| moe-gate (routing) | 1 | `recipes/kimi_moe_gate.py` (sgl_kernel.kimi_k2_moe_fused_gate, **exact-index oracle**) | **✅** | todo (needs topk driver) |
| grouped-moe (masked decode) | 2 | `recipes/kimi_grouped_moe_masked.py` (deep_gemm.fp8_m_grouped_gemm_nt_masked, E=12) | **✅** | todo (needs fused-MoE driver) |
| lm-head | 1 | `recipes/kimi_lm_head.py` (torch.matmul bf16, vocab=163840) | **✅** | todo (needs logits-proc driver) |
| bmm-absorb | 2 | `recipes/bmm.py` (torch.bmm) | **✅** | n/a (global symbol) |
| prefill router / grouped-moe contiguous | — | F.linear / m_grouped_fp8_gemm_nt_contiguous | todo | todo |
| attention (MLA) | 2 | flashinfer trtllm / flash_mla | todo | todo |
| moe-gate | 2 | sgl_kernel.kimi_k2_moe_fused_gate | todo | todo |
| moe-combine | 2 | sgl_kernel.moe_sum | todo | todo |

### MiniMax-M3 (`tasks/minimax_m3/`)

**Authoritative** config from the live HF `text_config` of `MiniMaxAI/MiniMax-M3`
(`MiniMaxM3SparseForConditionalGeneration`, `model_type=minimax_m3_vl` — a DSA
sparse-attention + MoE **vision-language** model): hidden=6144; GQA 64q/4kv, head_dim=128,
rotary_dim=64 (partial 0.5); MoE **128 experts / top-4**, moe_inter=3072, 1 shared expert;
scoring=sigmoid, routed_scaling=2.0; **use_gemma_norm=True**, qk_norm=per_head;
hidden_act=**swigluoai**; vocab=200064; DSA: index_dim=128, 4 index heads, topk_blocks=16,
block=128. (Distinct from MiniMax-M2/M2.5/M2.7, which are hidden=3072 / 256 experts / top-8.)

| family | tasks | recipe | evaluate | drop-in |
|---|---|---|---|---|
| gemma-rmsnorm | 2 | `recipes/minimax_gemma_rmsnorm.py` (sgl_kernel.gemma_rmsnorm, ×(1+w)) | **✅** | **✅ interface-exact** |
| gemma-fused-add-rmsnorm | 2 | `recipes/minimax_gemma_fused_add_rmsnorm.py` (sgl_kernel.gemma_fused_add_rmsnorm) | **✅** | **✅ interface-exact** |
| bf16-linear | 12 | `recipes/bf16_linear.py` (cuBLAS bf16: Dense FFN, QKV, O_proj, shared expert pre/dec) | **✅** | todo |
| router-gemm | 2 | `recipes/minimax_router.py` (fp32 cuBLAS, 6144→128; dsv3_router_gemm rejects N=128) | **✅** | todo |
| moe-gate (routing) | 1 | `recipes/minimax_moe_gate.py` (sgl_kernel.topk_sigmoid, **exact-index oracle**) | **✅** | todo |
| grouped-moe (masked decode) | 2 | `recipes/grouped_moe_masked.py` (deep_gemm masked, E=16) | **✅** | todo |
| grouped-moe (contiguous prefill) | 2 | `recipes/grouped_moe_contiguous.py` (deep_gemm contiguous, E=16) | **✅** | todo |
| moe-combine | 2 | `recipes/minimax_moe_combine.py` (sgl_kernel.moe_sum, DPS) | **✅** | todo |
| rope (partial 64/128) | 2 | `recipes/rope.py` (real max_pos=1,048,576, theta=5e6) | **✅** | **✅ interface-exact** |
| embedding | 2 | `recipes/embedding.py` (F.embedding, vocab=200064) | **✅** | **✅** scoped |
| lm-head | 1 | `recipes/kimi_lm_head.py` (torch.matmul bf16, vocab=200064) | **✅** | todo |
| **P1: fp8-linear (MXFP8 proxy)** | — | deep_gemm w8a8_block_fp8 for QKV/O/expert; act-quant `sglang_per_token_group_quant_fp8` | todo | — |
| swiglu-oai | — | fused into MoE runner / ROCm-only Triton — no clean CUDA standalone; silu_and_mul is a proxy | n/a | n/a |
| **P2: DSA sparse stack (op29-34)** | — | `minimax_qknorm_rope`, `minimax_decode_topk`, `store_kv_index`, `minimax_sparse_ops`, `fmha_sm100` | **blocked** | — |

**Blocked-on-sglang-version:** the M3-specific DSA kernels (op29-34) and `mega_moe` (op38 whole-op) are **absent from this sglang checkout** (predates M3 DSA support; the inventory CSV was built against a newer build / `amd_add_m3` worktree). Building them here would mean inventing a baseline that doesn't exist — deferred rather than proxied. `fmha_sm100` is present but its `minimax_sparse_ops` caller is not.

### Known caveats
- `qa_kva_fused_*` uses the generic w8a8_block_fp8 path. sglang also has a
  `dsv3_fused_a_gemm` fused fast path for M≤16 (fuses act-quant+GEMM) that could be a
  separate, more faithful decode baseline later.
- Tolerance defaults (`atol=0.1, rtol=0.05, ratio=0.98`) are set for fp8-GEMM bf16
  output at 1/sqrt(K) weight scale; tighten per-op if an agent games the tolerance.
- **Index-producing / routing ops** (`moe-gate`) use an **exact-index oracle**
  (`atol=0, rtol=0, matched_ratio=1.0`) and output the integer indices only — the
  harness applies one tolerance per workload, so int-exact ids can't be co-judged with
  float weights. Verified with a negative test (rerouting one expert per token fails).
  Assumes no score ties (measure-zero for random floats); a same-set-different-order
  result fails by design.
- **Single canonical Kimi-K2.7 config** (one source of truth in `gen_tasks.py`):
  hidden=7168, q_lora=1536, kv_lora=512, num_heads=64, qk_nope=192, v_head=128,
  qk_rope=64, dense_inter/TP=2304, moe_inter/TP=256, vocab=163840, and MoE
  **n_routed_experts=384, topk=6, routed_scaling=2.872**. Both `router-gemm` (N=384)
  and `moe-gate` (384 experts) use it. `dsv3_router_gemm` accepts N∈{256,384} per
  sglang's dispatch (deepseek_v2.py:585); the flashinfer 256-only path is a
  DeepSeek/MiMo variant and is deliberately not used here.
