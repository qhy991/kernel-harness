# Draft — GLM-5.2 `index_k_proj_decode` (Phase 1)

> **Retarget note (2026-07-15):** Harness shapes updated to **DP=1 / TP=1 / EP=32** (commit `88d7044`). Numeric axes below were patched in-place; re-benchmark after this change.


Kernel-Harness task: `testbench/tasks/glm52/index_k_proj_decode/`.
Target machine: NVIDIA **B200** (sm_100, cap 10.0), CUDA 13.0, torch 2.11.0, triton 3.6.0.

Goal of this draft: define the **first correct, independent** B200 implementation of
`solution.py` that matches `reference.py` (the SGLang DeepGEMM `w8a8_block_fp8`
baseline) within tolerance on both sweep shapes `M ∈ {16, 32}`, and that leaves a
clear runway for later performance work. Beating the baseline is desirable but
**not** required to exit Phase 1.

---

## 1. Task contract (authoritative facts)

| Item | Value |
|------|-------|
| Op | DSA Index-K projection, decode: `out[M,N] = x_fp8[M,K] @ w_fp8[N,K].T` |
| `K` (hidden) | `6144` (const) |
| `N` (index head dim) | `128` (const) |
| `M` (decode batch) | sweep `{16, 32}` (variable) |
| K-block scale words | `K // 512 = 12` int32 per row (ue8m0, 4 blocks/word) |
| Output dtype | `bfloat16`, shape `[M, N]` |
| FLOPs | `2*M*K*N` → 25.2 MFLOP @ M=16, 50.3 MFLOP @ M=32 |
| Tolerance | `max_atol=0.1`, `max_rtol=0.05`, `required_matched_ratio=0.999` |
| Oracle + baseline | live `reference.py` = `w8a8_block_fp8_matmul_deepgemm` |
| Deployment context | B200, DP1/TP1/EP32 (informational) |

**Harness fact that dictates the whole design:** the driver
(`testbench/harness/driver.py`) builds inputs with the **reference's**
`get_inputs` and only imports **`run()`** from `solution.py`. My candidate's own
`get_inputs` is never called. Therefore `run(x_fp8, x_scale, w_fp8, w_scale)`
must consume the **exact pre-quantized DeepGEMM/SGLang ue8m0 layout** produced by
the reference and emit a matching `[M,128] bf16` result. Only `run()` is timed
(CUPTI cold-L2, median device-kernel span), on freshly `clone()`-ed args per
iteration.

---

## 2. Baseline behavior & measured evidence

`reference.run()` calls `w8a8_block_fp8_matmul_deepgemm(x_fp8, w_fp8, x_scale,
w_scale, [128,128], output_dtype=bf16)` — DeepGEMM's Blackwell blockwise FP8 GEMM
(tcgen05/TMA, 128×128 output tiles, persistent scheduler).

Advisory smoke (`harness.profile`, warm-L2, 20 reps) — current `solution.py` is a
byte-copy of the baseline, so this is the baseline itself:

| Shape | median | min | eff. BW | % HBM | achieved TFLOP/s | AI (FLOP/B) | bound label |
|-------|--------|-----|---------|-------|------------------|-------------|-------------|
| M=16  | 51.81 µs | 49.34 µs | 17.2 GB/s | 0.2% | 0.49 | 28.1 | memory-bound |
| M=32  | 52.85 µs | 49.47 µs | 18.7 GB/s | 0.2% | 0.95 | 50.4 | memory-bound |

**Diagnosis.** Latency is essentially **flat** from M=16→M=32 while FLOPs double,
and effective HBM use is **0.2%** of a ~8 TB/s B200. This op is neither
compute- nor bandwidth-bound in practice — it is **fixed-overhead / launch-and-
schedule bound**. The weight matrix is `128*6144 = 768 KiB` of FP8; at even 2 TB/s
that streams in ≈0.4 µs, so ~99% of the observed time is DeepGEMM's per-call
machinery (grid launch, persistent-scheduler spin-up, descriptor setup) applied to
a single tiny output tile (`N=128` = one 128-tile; `M≤32` uses 1 of 128 MMA rows).

**Caveats on these numbers.** `harness.profile` is advisory warm-L2 (CUDA-event
style, includes host launch). The authoritative WIN gate uses **CUPTI cold-L2
device-kernel span** and reads the per-shape denominator from
`.baseline_cache.json` (not yet materialized — created on first `evaluate.py`).
The device-only span will be smaller than 52 µs, but the flat-in-M signature means
the DeepGEMM kernel's own fixed cost still dominates. **Re-measure the real
baseline with `evaluate.py` before making any performance claim.**

**Implication for Phase 1+.** There is large *potential* headroom, but it lives in
fixed overhead, not arithmetic. A lean single-launch kernel (small grid, no JIT
dispatch, no persistent scheduler) is the credible path to beating the baseline
later. For Phase 1 we only need a correct, independent, overhead-lean-ish kernel.

---

## 3. Numerical contract — input layout (the correctness crux)

Reference `get_inputs` produces (via `sglang_per_token_group_quant_fp8(...,
column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True)` and
`requant_weight_ue8m0(...,[128,128])`):

- `x_fp8` : `[M, K]` `float8_e4m3fn`, row-major.
- `x_scale` : `[M, K//512=12]` `int32`, **ue8m0-packed, mn-major TMA-aligned,
  column-major** (allocated as `empty(12, align(M,4)).transpose(-1,-2)[:M,:]`, so
  stride ≈ `(1, align(M,4))`). Per-**token**, per-128-K-block exponents.
- `w_fp8` : `[N=128, K]` `float8_e4m3fn`, row-major.
- `w_scale` : `[N=128, 12]` `int32`, ue8m0-packed; produced by
  `transform_scale_ue8m0(out_s, mn=128)` which **repeats the single 128×128
  N-block's scale across all 128 rows** then packs. Effectively `[1, 48]` distinct
  values broadcast over N.

**ue8m0 semantics.** Each byte is an e8m0 exponent `e`; the scale is the exact
power of two `2^(e-127)`. Packing/unpacking (confirmed from the shipped torch
reference impls `_get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl` and
`_inverse_transform_scale_ue8m0_impl`):

- forward: `ue8m0 = (fp32_scale.view(int32) >> 23).to(uint8)`; 4 consecutive
  K-block bytes → one little-endian int32 word (word `j` ↔ K-blocks `4j..4j+3`).
- inverse: `fp32_scale = (uint8_exp.to(int32) << 23).view(float32)`.

**Reference math I must reproduce (within tolerance).** With `kb = k // 128`:

```
x_deq[m,k] = float(x_fp8[m,k]) * 2^(x_exp[m,kb]   - 127)
w_deq[n,k] = float(w_fp8[n,k]) * 2^(w_exp[kb]      - 127)     # single N-block
out[m,n]   = bf16( Σ_k  x_deq[m,k] * w_deq[n,k] )             # fp32 accumulate
```

DeepGEMM accumulates per 128-K-block in fp32 and applies the combined block scale
at block boundaries; a straightforward fp32 dequant + fp32 accumulation matches
this to far better than `atol=0.1` (output magnitude is O(1) by the `1/sqrt(K)`
weight init).

**Unpacking recipe (verified against shipped impls).**
- Weights: reuse SGLang's own `inverse_transform_scale_ue8m0(w_scale, mn=128)` →
  `[1, 48]` fp32 (it self-checks by re-packing and asserting equality), or the
  equivalent `(w_scale.contiguous().view(uint8).view(128,48).int()<<23).float()`
  and take row 0 (all rows identical).
- Activations (no 128-repeat): `sx = (x_scale.contiguous().view(torch.uint8)
  .view(M,48).to(int32) << 23).view(torch.float32)` → `[M, 48]` fp32 power-of-two
  scales. `.contiguous()` first normalizes the column-major/cloned stride.

This unpacking is the single highest-risk correctness item and gets an explicit
offline check (see §6, step 2) before any kernel is trusted.

---

## 4. Ranked candidate directions

### C0 — Reference-independent torch dequant + bf16 matmul  *(first-correct / safety net)*
Unpack both ue8m0 scales to fp32 (§3), dequantize `x_fp8`,`w_fp8` to bf16 with a
broadcast multiply, then `out = (x_deq @ w_deq.t()).to(bf16)`.
- **Pros:** trivially, provably correct; independent of DeepGEMM; validates the
  unpack helper end-to-end; ~10 lines. Establishes Phase-1 `correct=true`.
- **Cons:** extra HBM traffic (materializes bf16 `x`,`w`), several separate kernels
  → likely slower than baseline. Acceptable: Phase 1 does not require a WIN.
- **Role:** land first to bank correctness and de-risk the numeric contract.

### C1 — Custom Triton blockwise FP8 GEMM with split-K  *(primary Phase-1 deliverable)*
One Triton kernel: tile `BLOCK_N=128` (single N-tile), `BLOCK_M ∈ {16,32}` (one
M-tile), loop K in 128-wide blocks; load `x_fp8`/`w_fp8` tiles, apply the per-block
ue8m0 scale (either unpacked in a tiny torch prologue or, better, read the int32
scale words directly in-kernel and reconstruct `2^(e-127)` with shifts), fp32
accumulate, write bf16.
- **Split-K** across the 48 K-blocks (e.g. `SPLIT_K ∈ {4,8,16}`) so multiple CTAs
  stream disjoint K-slices in parallel — the lever for occupancy on this
  single-output-tile shape — with a small fp32 partial-sum reduction (atomic add
  into fp32 accumulator, or a `[SPLIT_K, M, N]` scratch + tiny reduce kernel).
- **Pros:** independent, owned, and tunable (tile/split/num_warps/stages); native
  Triton FP8 `tl.dot` on sm_100; a plausibly leaner launch than DeepGEMM's
  persistent path → the real shot at the fixed-overhead floor. Reuses C0's verified
  unpack for correctness.
- **Cons:** must get in-kernel scale indexing exactly right; Triton launch still has
  fixed cost — beating 50 µs is not guaranteed but the flat-in-M evidence says it's
  in reach.
- **Role:** the Phase-1 kernel of record; carries into Phase-2/3 tuning.

### C2 — Lean CUDA / CuTe-DSL single-launch kernel  *(deferred; performance phase)*
Hand-written sm_100 kernel treating this as a fat GEMV: minimal grid, TMA/`cp.async`
weight streaming, warp-level fp32 reduction, one launch, no JIT dispatch — directly
attacks the fixed-overhead floor the roofline exposes. Optionally CUDA-graph capture
if the harness clone/timing path allows it.
- **Role:** not Phase 1. Revisit only if C1 lands correct and profiling shows the
  remaining gap is launch/scheduler overhead a custom kernel can remove.

### Rejected / non-deliverable options
- **Thin re-export of `w8a8_block_fp8_matmul_deepgemm`** — explicitly disallowed by
  the phase contract (calling the exact baseline op as the whole body of `run()`).
- **Calling SGLang's Triton `w8a8_block_fp8_matmul`** (the non-DeepGEMM path) — it
  needs fp32 scales so I'd unpack first; it is a *different* kernel from the
  baseline, but using it verbatim is a borderline re-export. **Use only as an
  independent correctness cross-check for C0/C1, not as the deliverable.**

---

## 5. First concrete steps

1. **Write & unit-check the ue8m0 unpack helper** (`_unpack_x_scale`,
   `_unpack_w_scale`) offline against the reference dequant (§3). Gate: bit-for-bit
   match to SGLang's `inverse_transform_scale_ue8m0` on weights; for activations,
   confirm `x_deq @ w_deq.T` reproduces `reference.run(...)` to `max_abs_err ≪ 0.1`.
2. **Land C0** into `solution.py` (dequant + bf16 matmul). Run the one-workload
   smoke then the full authoritative gate. Target: `correct=true` on M=16 and M=32.
3. **Implement C1** (Triton, no split-K first for a clean correctness baseline),
   reusing the verified unpack. Re-run the gate; record latency.
4. **Add split-K + a small tuning sweep** (`SPLIT_K`, `num_warps`, `num_stages`,
   in-kernel vs prologue scale unpack). Keep the fastest correct variant.
5. **Profile** the leading correct variant with `ncu-report-skill` only if a WIN is
   plausibly in reach; log the report under `kda/profile/`.
6. Record each attempt in `kda/candidates.jsonl` and every measured run in
   `kda/benchmark.csv`.

Phase-1 exit is reached at step 2/3 (first correct independent kernel); steps 4–6
are performance follow-through, not exit requirements.

---

## 6. Exact validation commands

Use the shared Kernel-Harness venv:

```bash
cd /home/qinhaiyan/KDA-Exp/worktrees/glm52-index_k_proj_decode
PY=/home/qinhaiyan/Kernel-Harness/.venv/bin/python

# (a) Fast advisory smoke — latency + roofline, per shape (not the gate):
PYTHONPATH=testbench $PY -m harness.profile \
  testbench/tasks/glm52/index_k_proj_decode --shape 16
PYTHONPATH=testbench $PY -m harness.profile \
  testbench/tasks/glm52/index_k_proj_decode --shape 32

# (b) Authoritative gate — 1 workload (fast correctness), then full sweep:
$PY testbench/bin/evaluate.py testbench/tasks/glm52/index_k_proj_decode --max-workloads 1
$PY testbench/bin/evaluate.py testbench/tasks/glm52/index_k_proj_decode
# exit 0 = WIN (correct AND faster on every shape); 1 = correct but not faster;
# 2 = incorrect.  Phase-1 success = correct=true on both shapes (exit 0 or 1).
```

Offline unpack/dequant sanity (step 1) is a scratch script under `kda/` comparing a
pure-torch dequant against `reference.run(...)` on the reference's own
`get_inputs`; it is not part of the gate.

---

## 7. Evidence rules — promote / revise / reject

- **Promote a variant** iff `evaluate.py` (full sweep) reports `correct=true` on
  **both** M=16 and M=32, with no NaN/Inf and matched-ratio ≥ 0.999. Record its
  latency. A variant that is *also* `solution_us < baseline_us` on every shape
  (exit 0) is a WIN and becomes the new leader.
- **Revise** (keep iterating, don't discard) a variant that is correct but slower
  than baseline, or that is close but fails matched-ratio only at the numeric edge
  (tighten accumulation: strictly fp32 accumulate, per-128-block scale application,
  avoid premature bf16 rounding).
- **Reject** a variant that is `INCORRECT` (tolerance/NaN/shape), `REWARD_HACK`
  (lazy/stateful output, monkey-patch — see `harness/reward_hack.py`), or
  `RUNTIME_ERROR`. Also reject any design that reduces to calling
  `w8a8_block_fp8_matmul_deepgemm` as the body of `run()`.
- **Trust only `evaluate.py`** for the verdict; `harness.profile` numbers are
  advisory (warm-L2) and must never be quoted as the WIN margin.

---

## 8. Risks & unknowns

1. **ue8m0 unpack correctness (highest).** Column-major/TMA-aligned stride,
   little-endian byte order within each int32, weight 128-row repeat, and the
   post-`clone()` stride. Mitigation: §3 recipe cross-checked against the shipped
   `inverse_transform_scale_ue8m0` self-check and an offline dequant match (§6).
2. **`clone_args` stride handling.** The harness clones with `preserve_format`, so
   the column-major `x_scale` stays column-major inside `run()`. The unpack does
   `.contiguous()` first; verify no assumption of row-major sneaks in.
3. **Beating the baseline is overhead-limited, not FLOP/BW-limited.** A correct
   Triton kernel may still not clear 50 µs if its own launch cost is comparable.
   This does **not** block Phase 1; it scopes C2 for later.
4. **Triton FP8 `tl.dot` on sm_100 with `BLOCK_M<32`.** Small-M MMA tiles / `tl.dot`
   constraints may force padding M up to 16/32 or a non-tensor-core dot. Fallback:
   fp32 dequant-in-kernel + regular `tl.dot` on bf16, or the C0 path.
5. **Authoritative baseline not yet cached.** `.baseline_cache.json` is created on
   first `evaluate.py`; the ~52 µs advisory figure is a rough scale only. Establish
   the real per-shape denominator before any WIN claim.
6. **DeepGEMM JIT warmup noise.** First `evaluate.py`/profile pays a one-time
   DeepGEMM pre-compile; ignore warmup, trust the median.

---

## 9. Performance theory (for later phases, non-blocking)

- Roofline: `AI ≈ 28 (M16) / 50 (M32) FLOP/B`; B200 machine balance (~4.5 PFLOP/s
  FP8 ÷ ~8 TB/s) ≈ 560 → deeply below the ridge → memory-bound *in theory*, but the
  measured 0.2% HBM shows the real limiter is **fixed per-call overhead**.
- Absolute floor ≈ weight-stream time: `768 KiB / 8 TB/s ≈ 0.1 µs` + one launch.
  The realistic target is "few µs", set by the minimum device-kernel span of a lean
  single-launch kernel — an order of magnitude under the baseline if C1/C2 succeed.
- Levers, in order: (1) collapse launch/scheduler overhead (lean grid, no
  persistent scheduler, no JIT dispatch); (2) split-K for SM occupancy while
  weight-streaming; (3) vectorized FP8 loads + in-kernel ue8m0 reconstruct to avoid
  prologue kernels; (4) fp32 accumulate to stay inside tolerance cheaply.
```
