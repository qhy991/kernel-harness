# Plan — GLM-5.2 `index_k_proj_decode` First Correct Independent B200 Kernel (Phase 1)

> **Retarget note (2026-07-15):** Harness shapes updated to **DP=1 / TP=1 / EP=32** (commit `88d7044`). Numeric axes below were patched in-place; re-benchmark after this change.


## Goal Description

Produce the first **correct, independent** B200 implementation of
`testbench/tasks/glm52/index_k_proj_decode/solution.py` whose `run()` matches the
live oracle `reference.py` (SGLang DeepGEMM `w8a8_block_fp8_matmul_deepgemm`)
within tolerance (`max_atol=0.1`, `max_rtol=0.05`, `required_matched_ratio=0.999`)
on **both** sweep shapes `M ∈ {16, 32}` (with `K=6144`, `N=128`, bf16 output
`[M,128]`).

The op is a skinny, block-scaled FP8 GEMM: `out[M,128] = x_fp8[M,6144] @
w_fp8[128,6144].T` with per-token 1×128 activation scales and 128×128 weight
scales, both delivered as **ue8m0-packed int32** in DeepGEMM's mn-major
TMA-aligned layout. The harness builds inputs with the **reference's**
`get_inputs` and imports only `run()` from `solution.py`; therefore the deliverable
must consume that exact pre-quantized layout and reproduce the reference result.

Phase-1 success = authoritative `evaluate.py` reports `correct=true` on both
shapes. Beating the baseline latency is **desirable but not required**. The chosen
path must be genuinely independent (not a re-export of the baseline op) and must
leave a clean runway for later performance work.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive tests (expected to PASS)
and negative tests (expected to FAIL / be rejected when the implementation is
correct). "Gate" = `PY=/home/qinhaiyan/Kernel-Harness/.venv/bin/python; $PY
testbench/bin/evaluate.py testbench/tasks/glm52/index_k_proj_decode`.

- AC-1: **The ue8m0 input contract is empirically verified before any kernel is trusted.**
  A throwaway probe (under `kda/`, not `solution.py`) reconstructs the reference
  output from `(x_fp8, x_scale, w_fp8, w_scale)` via a self-contained dequant and
  compares against `reference.run(...)` on the reference's own `get_inputs`.
  - Positive Tests:
    - Probe prints `x_scale` / `w_scale` `.shape`, `.stride()`, `.storage_offset()`,
      `.is_contiguous()`, decodes a few exponent bytes, and its dequant reproduces
      `reference.run(...)` with `max_abs_err ≪ 0.1` for M=16 and M=32.
    - The self-contained weight unpack agrees with SGLang's
      `inverse_transform_scale_ue8m0(w_scale, mn=128)` bit-for-bit.
  - Negative Tests:
    - An unpack that ignores the column-major stride / little-endian byte order (e.g.
      reads `x_scale` without `.contiguous()` normalization, or transposes K vs mn)
      produces `max_abs_err ≫ 0.1` and is rejected.
    - Treating `w_scale` as 128 independent per-row scales instead of one repeated
      128×128 N-block scale changes the result and is rejected.

- AC-2: **`solution.py:run()` is an independent implementation, not a baseline re-export.**
  - Positive Tests:
    - `run()` computes the GEMM through owned code (torch blockwise dequant + matmul,
      or a custom Triton/CUDA kernel) and unpacks scales with self-contained bit ops;
      grep of `solution.py` shows no call to `w8a8_block_fp8_matmul_deepgemm`.
    - `solution.py` imports no SGLang/DeepGEMM GEMM kernel; any SGLang scale helper is
      used only in offline probes, never in `run()`.
  - Negative Tests:
    - A `run()` whose body is `return w8a8_block_fp8_matmul_deepgemm(...)` is rejected
      as a disallowed thin re-export.
    - A `run()` relying on the candidate's own `get_inputs` (never called by the
      driver) fails because it receives the reference's ue8m0 tensors.

- AC-3: **Authoritative correctness on every sweep shape.**
  - Positive Tests:
    - Gate `--max-workloads 1` then full Gate report `correct=true`, matched-ratio ≥
      0.999, no NaN/Inf, for M=16 and M=32; output dtype `bfloat16`, shape `[M,128]`,
      on CUDA.
    - Exit code is `0` (WIN) or `1` (correct, not faster) — both satisfy Phase 1.
  - Negative Tests:
    - Any shape returning `INCORRECT`, `REWARD_HACK`, or `RUNTIME_ERROR` (exit `2`)
      fails.
    - Output as fp32/fp16, wrong shape, or with aliasing/lazy tricks is rejected by the
      driver's normalization and post-timing recheck.

- AC-4: **A pure-torch blockwise-dequant safety-net variant (C0) lands first and passes AC-3.**
  - Positive Tests:
    - C0 accumulates over the 48 K-blocks — `acc += (x_fp8_block.float() *
      sx[:,kb,None]) @ (w_fp8_block.float() * sw[kb]).T`, cast to bf16 — and passes the
      full Gate on both shapes.
  - Negative Tests:
    - A C0 variant that applies a single global scale (ignoring per-128-K-block
      granularity) exceeds tolerance and fails.

- AC-5: **A custom single-kernel Triton FP8 GEMM (C1) passes AC-3 as the Phase-1 kernel of record.**
  - Positive Tests:
    - One Triton kernel: `BLOCK_N=128`, one M-tile (`BLOCK_M ∈ {16,32}`), `BLOCK_K=128`
      loop over 48 blocks, per-block ue8m0 scale applied, fp32 accumulator, bf16 store,
      a single output allocation, **no split-K, no atomics**; passes the full Gate on
      both shapes.
    - Scale reconstruction `2^(e-127)` via integer `(e << 23)` is done in-kernel (or an
      unavoidable tiny prologue) and produces correct results.
  - Negative Tests:
    - A C1 variant that loses per-block scale granularity (e.g. `BLOCK_K>128` without
      internal sub-scaling) fails tolerance.
    - A C1 variant with an uninitialized/globally-cached output or partial buffer fails
      the post-timing oracle recheck (statefulness) and is rejected.

- AC-6: **Performance is measured and recorded, but Phase-1 pass does not depend on it.**
  - Positive Tests:
    - Advisory `harness.profile` (M=16, M=32) and authoritative `evaluate.py` latencies
      are recorded in `kda/benchmark.csv`; every candidate is logged in
      `kda/candidates.jsonl`.
    - Any "faster than baseline" claim cites only `evaluate.py` (CUPTI cold-L2), never
      warm-L2 `harness.profile`.
  - Negative Tests:
    - Quoting the warm-L2 advisory number as a WIN margin is rejected.
    - Declaring a WIN without a fresh per-shape baseline from `evaluate.py`
      (`.baseline_cache.json`) is rejected.

- AC-7: **Process hygiene and edit boundaries are respected.**
  - Positive Tests:
    - Only `testbench/tasks/glm52/index_k_proj_decode/solution.py` is edited for the
      candidate; NCU artifacts (if any) live under `kda/profile/`.
  - Negative Tests:
    - Edits to `reference.py`, `workload.jsonl`, tolerances, or `testbench/harness/*`
      are rejected.
    - Copying a final contest release kernel into the workspace is rejected.

- AC-8 (optional, performance-phase only): **Split-K is added only after C1 is correct and measured.**
  - Positive Tests:
    - A split-K variant with deterministic zero-initialized fp32 partials and a correct
      reduction passes the Gate on both shapes with matched-ratio not regressed vs C1,
      and is faster.
  - Negative Tests:
    - A split-K variant with nondeterministic/uninitialized partials, or that regresses
      matched-ratio, is rejected in favor of the single-kernel C1.

## Path Boundaries

The numeric contract (I/O layout, dtypes, tolerance, oracle) is **fixed** by the
task; only the internal algorithm/kernel is a free choice.

### Upper Bound (Maximum Acceptable Scope)
The custom single-kernel Triton FP8 GEMM (C1) is correct on M=16 and M=32 and is the
kernel of record, with the pure-torch C0 retained as a reference/fallback; a bounded
tuning exploration (tile sizes, `num_warps`, `num_stages`, in-kernel vs prologue scale
unpack, and optionally split-K per AC-8) keeps the fastest correct variant, all
measured via `evaluate.py` and logged. Deeper single-launch CUDA/CuTe work (C2) is
noted as a future direction, not built in Phase 1.

### Lower Bound (Minimum Acceptable Scope)
The pure-torch blockwise-dequant C0 variant is correct on both shapes and passes the
authoritative Gate (exit `0` or `1`), delivering the first correct independent
implementation even if slower than the baseline.

### Allowed Choices
- Can use: Triton (native FP8 `tl.dot` on sm_100), custom CUDA/CuTe-DSL, torch
  dequant + `torch.matmul`, self-contained ue8m0 unpack via integer bit ops, fp32
  accumulation, optional split-K with a correct reduction; SGLang's
  `inverse_transform_scale_ue8m0` **only as an offline correctness cross-check**.
- Cannot use: calling `w8a8_block_fp8_matmul_deepgemm` (or any DeepGEMM/SGLang GEMM
  kernel) as the body of `run()`; editing `reference.py`, `workload.jsonl`,
  tolerances, or the harness; depending on the candidate's own `get_inputs`; copying
  contest release kernels; CUDA-graph capture inside `run()` (incompatible with the
  per-iteration arg clone).

## Feasibility Hints and Suggestions

> Reference only — conceptual, not prescriptive.

### Conceptual Approach
Self-contained ue8m0 unpack (both scales decode to exact powers of two):
```
# activation scale: [M,12] int32, ue8m0-packed, column-major TMA-aligned
sx = (x_scale.contiguous().view(torch.uint8).view(M, 48).to(torch.int32) << 23) \
        .view(torch.float32)                    # [M,48] = 2^(e-127) per token,K-block
# weight scale: [128,12] int32; the single 128x128 N-block scale repeated over rows
sw = (w_scale.contiguous().view(torch.uint8).view(128, 48).to(torch.int32) << 23) \
        .view(torch.float32)[0]                 # [48]  (all 128 rows identical)
```
C0 (safety net — mirrors DeepGEMM's per-block granularity, easiest to debug):
```
acc = zeros(M, 128, float32)
for kb in range(48):
    xb = x_fp8[:, kb*128:(kb+1)*128].float() * sx[:, kb:kb+1]      # [M,128]
    wb = w_fp8[:, kb*128:(kb+1)*128].float() * sw[kb]              # [128,128]
    acc += xb @ wb.T
return acc.to(torch.bfloat16)                                     # [M,128]
```
C1 (Triton, single kernel): grid = one `[M,128]` output tile; loop 48 K-blocks with
`BLOCK_K=128`; per block, load `x_fp8`/`w_fp8` tiles, `tl.dot` (or fp32 dequant then
dot), multiply by `sx[:,kb]` (per-M-row) and `sw[kb]` (scalar), fp32-accumulate,
`tl.store` bf16. Reconstruct scales in-kernel from int32. Add split-K (AC-8) only
after this is correct and measured.

Empirically verify the layout **first**: decode the ue8m0 bytes and confirm the
dequant matches `reference.run(...)`; inspect `stride()`/`is_contiguous()` of the
**cloned** `x_scale` inside a probe (the harness clones with `preserve_format`, so a
column-major scale stays column-major — `.contiguous()` normalizes it).

### Relevant References
- `testbench/tasks/glm52/index_k_proj_decode/reference.py` — oracle + input layout
  (`sglang_per_token_group_quant_fp8(..., scale_ue8m0=True)`, `requant_weight_ue8m0`).
- `testbench/harness/driver.py` — driver uses reference `get_inputs`; imports only
  candidate `run()`; correctness → timing → post-timing oracle recheck.
- `testbench/harness/timing.py` — CUPTI cold-L2 device-kernel span; `clone_args`.
- `testbench/harness/reward_hack.py` — lazy/stateful/monkey-patch rejection.
- SGLang `.../quantization/fp8_utils.py` — `inverse_transform_scale_ue8m0`,
  `_inverse_transform_scale_ue8m0_impl`, `transform_scale_ue8m0` (layout ground truth).
- SGLang `.../quantization/fp8_kernel.py` — `create_per_token_group_quant_fp8_output_scale`
  (activation scale column-major TMA-aligned allocation).

## Dependencies and Sequence

### Milestones
1. Contract verification: empirically prove the ue8m0 unpack + dequant against the
   oracle (AC-1).
   - Phase A: read `task.json`, `definition.json`, `reference.py`, full
     `workload.jsonl`; confirm sweep is exactly `{16,32}`.
   - Phase B: probe script decodes scales, inspects strides, reproduces
     `reference.run(...)` within tolerance on both shapes.
2. First correct independent kernel: land C0 into `solution.py` and pass the Gate
   (AC-2, AC-3, AC-4) — this satisfies Phase-1 exit.
   - Step 1: implement self-contained unpack + blockwise dequant + matmul.
   - Step 2: run Gate `--max-workloads 1`, then full Gate; record results.
3. Kernel of record: implement the single-kernel Triton C1 reusing the verified
   unpack; pass the Gate (AC-5); record latency.
4. Bounded optimization (does not gate Phase 1): tune C1; optionally add split-K
   (AC-8); keep the fastest correct variant; profile with `ncu-report-skill` only if a
   WIN is plausibly in reach (AC-6).

Dependencies: M2 depends on M1 (verified unpack). M3 depends on M1 and reuses M2's
unpack. M4 depends on a correct M3. C0 remains the correctness fallback throughout.

## Task Breakdown

Each task carries exactly one routing tag (`coding` = Claude implements; `analyze` =
Codex via `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Read task files + full `workload.jsonl`; confirm sweep `{16,32}`, dtypes, output contract | AC-1 | coding | - |
| task2 | Probe: decode ue8m0 bytes, inspect cloned scale strides/contiguity, reproduce `reference.run(...)` within tolerance (M=16,32) | AC-1 | coding | task1 |
| task3 | Analyze DeepGEMM per-128-block accumulation vs fp32 dequant-accumulate: confirm loose tolerance is safe; confirm Triton fp8 `tl.dot` constraints at `BLOCK_M∈{16,32}` on sm_100 | AC-3, AC-5 | analyze | task2 |
| task4 | Implement self-contained ue8m0 unpack + C0 blockwise dequant `run()` in `solution.py` | AC-2, AC-4 | coding | task2 |
| task5 | Run Gate (`--max-workloads 1` then full); record correctness + latency in `kda/benchmark.csv`, `kda/candidates.jsonl` | AC-3, AC-6, AC-7 | coding | task4 |
| task6 | Implement single-kernel Triton C1 (no split-K/atomics, fp32 accum, bf16 store) reusing verified unpack | AC-5 | coding | task3, task5 |
| task7 | Gate C1 on both shapes; record; keep the leading correct variant | AC-3, AC-5, AC-6 | coding | task6 |
| task8 | Bounded C1 tuning; optional split-K with deterministic zero-init + correct reduction; profile if WIN plausible | AC-6, AC-8 | coding | task7 |

## Claude-Codex Deliberation

### Agreements
- The single highest risk is the ue8m0 scale layout (byte order, column-major stride,
  `clone()` interaction, weight 128-row repeat); it must be proven empirically against
  the oracle before trusting any kernel.
- C0 should accumulate blockwise over the 48 K-blocks (matching DeepGEMM granularity),
  not full-matrix dequant — easier to debug and numerically faithful.
- Start C1 as a single Triton kernel with **no split-K and no atomics**; split-K is a
  weak lever for this tiny shape and only justified later, with deterministic zero-init.
- In-kernel scale reconstruction is preferred over a torch prologue (fewer timed
  launches).
- The baseline is fixed-overhead/launch-bound, not FLOP/BW-bound; beating it is not a
  Phase-1 requirement, and Triton may not clear it — CUDA/CuTe single-launch (C2) is the
  realistic performance path, deferred.
- Performance verdicts come only from `evaluate.py` (CUPTI cold-L2), never warm-L2
  `harness.profile`.

### Resolved Disagreements
- **May `solution.py` import SGLang scale helpers?** Codex flagged the independence
  bar. Resolution: the shipped `solution.py` unpacks scales with self-contained integer
  bit ops and imports **no** SGLang/DeepGEMM kernel; `inverse_transform_scale_ue8m0` is
  used only in the offline probe as a cross-check oracle. Rationale: guarantees genuine
  independence while still leveraging a trusted reference for verification.
- **Pure-torch first vs straight-to-Triton?** Resolution: land C0 (pure-torch
  blockwise) first to bank correctness and de-risk the layout, then C1 Triton. Rationale:
  correctness is the Phase-1 gate; C0 isolates layout bugs from kernel bugs.
- **Split-K in the first C1?** Codex judged split-K a complexity/correctness multiplier
  with weak payoff here. Resolution: exclude split-K from the first C1 (AC-5); admit it
  only under AC-8 after C1 is correct and measured.
- **Command syntax (`integration_status.py`, `knowledge.py`, `--no-baseline`, short task
  names) suggested by Codex.** Resolution: follow the worktree contract in
  `kda/prompts/phase1.md` verbatim (full task path, `evaluate.py --max-workloads 1`
  then full sweep); treat those extra tools as optional context-gathering only, since the
  phase contract does not require them.

### Convergence Status
- Final Status: `partially_converged` (direct mode: Codex first-pass analysis was
  incorporated, but the iterative second-Codex convergence loop was intentionally skipped
  per `--direct`). Human review remains available before the RLCR loop.

## Pending User Decisions

The `--direct` autonomous session resolved the open questions with the rationale above;
they are recorded here for visibility and may be overridden before/at RLCR start.

- DEC-1: Independence bar for scale unpacking in `solution.py`.
  - Claude Position: `solution.py` uses self-contained bit-op unpack, no SGLang kernel
    imports; SGLang helper only in offline probe.
  - Codex Position: N/A - open question (raised whether SGLang helpers are acceptable in
    the final file).
  - Tradeoff Summary: Self-contained unpack maximizes independence and portability at a
    few extra lines; importing an SGLang util is convenient but couples the deliverable to
    SGLang internals. Autonomously decided in favor of self-contained.
  - Decision Status: Resolved (self-contained) — override to allow the SGLang helper in
    `run()` if strict independence is not desired.
- DEC-2: First-submission target (C0 pure-torch vs C1 Triton directly).
  - Claude Position: Land C0 first for guaranteed correctness, then C1.
  - Codex Position: N/A - open question (either simplest-correct-first or straight to
    Triton after layout is verified).
  - Tradeoff Summary: C0-first de-risks the layout and guarantees a Phase-1 pass; going
    straight to C1 saves one step but conflates layout and kernel bugs. Autonomously
    decided C0-first.
  - Decision Status: Resolved (C0 first) — override to skip C0 and go straight to C1 if
    preferred.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as
  "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. `dequant_block`,
  `unpack_ue8m0_scale`, `block_fp8_gemm`).

### Additional Notes
- Only `run(x_fp8, x_scale, w_fp8, w_scale)` is imported from `solution.py`; keep that
  exact signature and return a `bfloat16 [M,128]` CUDA tensor. The candidate's
  `get_inputs` is never called by the driver.
- Guard against the reward-hack checks: allocate/zero any output or partial buffer inside
  the timed `run()`; do not cache outputs globally or go lazy under timing (the driver
  re-verifies against the oracle after timing).
- DeepGEMM JIT warmup adds a one-time cost on the first `evaluate.py`/`profile`; ignore
  warmup and trust the median.

--- Original Design Draft Start ---

# Draft — GLM-5.2 `index_k_proj_decode` (Phase 1)

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

--- Original Design Draft End ---
