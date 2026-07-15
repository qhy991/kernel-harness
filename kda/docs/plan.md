# GLM-5.2 o_proj_decode — Phase-1 Implementation Plan (Correct Independent B200 FP8 Blockwise GEMM)

> **Retarget note (2026-07-15):** Harness shapes updated to **DP=1 / TP=1 / EP=32** (commit `88d7044`). Numeric axes below were patched in-place; re-benchmark after this change.


## Goal Description

Replace `testbench/tasks/glm52/o_proj_decode/solution.py` (currently an exact copy of
`reference.py`) with the **first correct, independent** B200 (sm_100) implementation of
the GLM-5.2 Attention O-Projection decode GEMM:

```
out[M, 6144] (bf16) = x_fp8[M, 16384] @ w_fp8[6144, 16384].T
```

with `M ∈ {16, 32}`, FP8 (`e4m3`) operands and UE8M0-packed blockwise scales
(1×128 activation scales, 128×128 weight scales), matched against the live SGLang
`deep_gemm w8a8_block_fp8` oracle within tolerance (`max_atol=0.1`, `max_rtol=0.05`,
`required_matched_ratio=0.999`).

Phase-1 exit is **correctness on both sweep shapes**. The candidate must compute the GEMM
itself — calling `w8a8_block_fp8_matmul_deepgemm` (or an equivalent DeepGEMM GEMM entry)
as the sole body of `run()` is explicitly **not** an acceptable deliverable. Beating the
baseline latency is desirable but **not required** to exit Phase 1; the chosen path must,
however, be one that can later be optimized toward the bandwidth roofline.

This plan is grounded in measurements taken on the actual B200 in this worktree and in the
prior-art knowledge entry `testbench/knowledge/entries/glm52--o_proj_decode--b200--20260714a.json`
(a previous session that ended **no-win** and whose lessons are folded in below).

## Acceptance Criteria

Following TDD philosophy, each criterion lists positive tests (must pass) and negative
tests (must be rejected/fail when the implementation is correct). The authoritative
verifier for AC-1/AC-2/AC-5/AC-6 is `testbench/bin/evaluate.py`.

- AC-1: **Correct on every sweep shape.** `evaluate.py` reports status `PASSED` /
  `correct=true` (matched_ratio ≥ 0.999, no nan/inf, output shape `[M,6144]`, dtype
  `bfloat16`) for **both** M=16 and M=32.
  - Positive Tests (expected to PASS):
    - `$PY testbench/bin/evaluate.py testbench/tasks/glm52/o_proj_decode` → both workloads PASSED.
    - `$PY testbench/bin/evaluate.py testbench/tasks/glm52/o_proj_decode --max-workloads 1` → M=16 PASSED.
  - Negative Tests (expected to FAIL / be rejected when working correctly):
    - A build that mis-decodes the UE8M0 scales (e.g. wrong byte order or missing the
      2^(e-127) power-of-two) → matched_ratio < 0.999 → `INCORRECT` (exit 2).
    - A build that returns fp32/fp16 or shape `[6144,M]` → normalization/shape mismatch → `INCORRECT`.

- AC-2: **Genuine independence (not a baseline re-export).** `run()` produces its numeric
  result from an owned computation (own kernel or own dequant+matmul), not from
  `w8a8_block_fp8_matmul_deepgemm`/`fp8_gemm_nt`/`deep_gemm_fp8_fp8_bf16_nt` as the sole body.
  - Positive Tests:
    - `grep -nE "w8a8_block_fp8_matmul_deepgemm|fp8_gemm_nt|deep_gemm_fp8_fp8_bf16_nt" solution.py`
      returns nothing on the compute path (imports for a *fallback-only* helper, if any, are
      not the measured path).
    - The measured candidate output is reproducible by the independent UE8M0-unpack oracle (AC-3).
  - Negative Tests:
    - `solution.py` identical to `reference.py` (calls the DeepGEMM wrapper) → violates AC-2
      even though it is correct.
    - `run()` whose body only re-dispatches DeepGEMM with different `num_sms`/config → violates AC-2.

- AC-3: **Consumes provided inputs verbatim; UE8M0 unpack matches the verified recipe.**
  `run(x_fp8, x_scale, w_fp8, w_scale)` uses the given tensors with no host-side
  re-quantization or input regeneration; the decode is
  `scale = 2^(e-127)`, `e = (scale_i32[row, kb//4] >> (8*(kb%4))) & 0xFF`, weight scale
  broadcast within each 128-row N-block.
  - AC-3.1: An offline unit check diffs the candidate against the pure-torch reconstruction
    oracle (§1.6 of the draft) at M=16 and M=32.
    - Positive: `max_abs ≤ ~0.01` and matched_ratio = 1.0 vs the oracle at both shapes.
    - Negative: ignoring `x_scale`/`w_scale`, or decoding weight scale per-row without the
      128-row broadcast, → large error vs oracle.
  - Positive Tests:
    - Candidate output equals the reconstruction oracle within tolerance for both shapes.
  - Negative Tests:
    - Re-running `sglang_per_token_group_quant_fp8` inside `run()` (re-quant) → wrong result / disallowed.

- AC-4: **Survives anti-reward-hack + post-timing recheck.** Output is a real CUDA
  `torch.Tensor` (not lazy/proxy), no monkey-patching of `torch.cuda.Event.elapsed_time`,
  and the output is stable when re-verified against the oracle after timing.
  - Positive Tests:
    - `evaluate.py` final status is `PASSED` (never `REWARD_HACK`).
  - Negative Tests:
    - Returning a `FakeTensor`/proxy or a cached/lazy result → `REWARD_HACK`.
    - A stateful path that computes honestly while checked but degrades under timing → post-timing recheck fails.

- AC-5: **Performance is measured and recorded (non-gating for Phase-1).** Every correct
  candidate's per-shape device-kernel median and speedup vs baseline are logged.
  - Positive Tests:
    - `kda/benchmark.csv` gains a row per shape (device_us, baseline_us, speedup, matched_ratio, max_abs_err).
    - Phase-1 accepts either exit `0` (WIN) or exit `1` (correct-but-slower).
  - Negative Tests:
    - Reporting a speedup without a corresponding `evaluate.py`/CUPTI measurement → not acceptable evidence.
    - Timing via CUDA events + Python launch overhead and claiming it as the device-kernel result → rejected (the ~53 µs CUDA-event figure is launch-dominated; the CUPTI device-kernel figure is ~9 µs).

- AC-6: **Scope discipline.** Only `solution.py` in the task directory is edited;
  `reference.py`, `definition.json`, `task.json`, `workload.jsonl`, and the harness are unchanged;
  no contest-release kernels are copied in.
  - Positive Tests:
    - `git status` shows changes limited to `testbench/tasks/glm52/o_proj_decode/solution.py`
      plus KDA bookkeeping under `kda/` (candidates/benchmark/profile/docs).
  - Negative Tests:
    - Any diff to `reference.py` → rejected.
    - Editing the harness timing/correctness to loosen the gate → rejected.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
A single-kernel Triton FP8 blockwise-scaled **weight-streaming** GEMM (C1) that decodes the
UE8M0 scales inline, streams the 12.58 MB weight once via tensor-core `tl.dot`, is autotuned
for M=16 and M=32 (BLOCK_N, warps, persistent/split-N scheduling), passes all ACs, and
**beats** the baseline on both shapes — with the pure-torch reconstruction (C2) retained as
a development-time correctness oracle, and a CuTe/CUTLASS `tcgen05` path (C3) attempted only
if profiling shows Triton cannot approach the bandwidth roofline. No over-engineering beyond
what closes the gap to the ~1.6 µs HBM floor.

### Lower Bound (Minimum Acceptable Scope)
The verified independent pure-torch dequant+GEMM (C2) installed as `solution.py`: unpack the
UE8M0 int32 scales, dequantize `x_fp8`/`w_fp8`, compute the matmul, cast to bf16 — correct on
both M=16 and M=32 (satisfying AC-1..AC-4, AC-6) even if slower than the DeepGEMM baseline.
This is a legitimate, independent Phase-1 deliverable (it does not call the baseline kernel),
though it is not the preferred optimization vehicle.

### Allowed Choices
- Can use: Triton (3.6.0, present), CUDA C++, CuTe DSL, CUTLASS, Python-wrapped CUDA
  extensions (all explicitly allowed by `phase1.md`); the harness `.venv` libraries; the
  verified UE8M0-unpack recipe; fp32 accumulation with per-128-K-block scale promotion;
  shape-specialized dispatch for M=16 vs M=32 with a safe fallback for unexpected M.
- Cannot use: `w8a8_block_fp8_matmul_deepgemm` / `deep_gemm.fp8_gemm_nt` /
  `deep_gemm_fp8_fp8_bf16_nt` as the sole compute body; any host-side re-quantization or
  input regeneration; edits to `reference.py` or the harness; monkey-patching timers or
  returning lazy/proxy outputs; copying contest-release kernels.

> **Note on Determinism**: The *numerics* are fully deterministic (the exact UE8M0 dequant
> is fixed and verified — the lower/upper bounds agree on what must be computed). The
> *implementation vehicle* is a genuine choice (torch vs Triton vs CuTe/CUTLASS); the upper
> and lower bounds differ only in performance ambition, not in the correctness contract.

## Feasibility Hints and Suggestions

> **Note**: Reference/understanding only — conceptual, not prescriptive.

### Conceptual Approach

The dominant cost is streaming the 12.58 MB FP8 weight once (activations, scales, output are
negligible). Prior art confirms the op is memory-bandwidth-bound and that DeepGEMM-wrapper
tricks do not win; the open, un-ruled-out direction is a custom kernel that changes the
actual device work (per the prior entry's own caveat). The verified UE8M0 unpack is the key
differentiator: a previous `sglang-triton-w8a8-block-fp8` attempt was *incorrect* because it
mis-read the packed scales — this plan decodes them with the numerically-verified recipe.

C1 (Triton) inner structure, per output N-tile:
```
acc = zeros([BLOCK_M, BLOCK_N], fp32)            # BLOCK_M = M (16 or 32); BLOCK_N aligned to 128
# x_fp8[M,K] is tiny (32-64 KiB) — keep resident/reused across the K loop
for kb in range(16):                              # K/128 = 16 scale blocks; BLOCK_K fixed = 128
    xk = load x_fp8[:, kb*128:(kb+1)*128]         # fp8
    wk = load w_fp8[n0:n0+BLOCK_N, kb*128:(kb+1)*128]   # fp8 — the streamed traffic
    p  = tl.dot(xk, wk.T)                          # fp8 tensor-core dot -> fp32 partial [M,BLOCK_N]
    sx = decode_ue8m0(x_scale, row=:, kb)          # 2^(e-127); e = (i32 >> 8*(kb%4)) & 0xFF, col kb//4
    sw = decode_ue8m0(w_scale, row=n0.., kb)       # one scale per 128-row N-block (exploit broadcast)
    acc += p * sx[:,None] * sw[None,:]             # rank-1 outer-product promotion, per K-block
store bf16(acc) -> out[:, n0:n0+BLOCK_N]
```
Notes distilled from Codex review + prior art:
- Apply the scale **per 128-K-block before summing across blocks** (a single dot over all K
  then one scale is wrong).
- Keep `BLOCK_K == 128` (scale granularity); other values risk incorrect scaling.
- Align `BLOCK_N` to 128 and load **one** weight-scale vector per 128-row N-block (all 128
  rows share it) instead of per row.
- Do **not** split M across programs (redundant weight reads); fix `BLOCK_M = M`.
- Occupancy: N=6144 / BLOCK_N=128 = 48 tiles < 148 SMs → consider smaller BLOCK_N or a
  persistent / split-N schedule to fill the machine; `num_stages` autotune may not help a
  memory-streaming tiny-M GEMM.
- Confirm `tl.dot` actually emits FP8 tensor-core ops on sm_100 (inspect Triton IR / PTX or
  NCU counters); a scalarized fp8→fp32 multiply-accumulate would be a correctness fallback
  only, not a performance path.

C2 (safety net / oracle): unpack scales → `(x_fp8.float()*sx) @ (w_fp8.float()*sw).T` → bf16.
Already verified to match DeepGEMM at matched_ratio 1.0 (max_abs 0.002–0.008).

### Relevant References
- `testbench/tasks/glm52/o_proj_decode/reference.py` — oracle + `get_inputs` (input layout, do not edit).
- `sglang.srt.layers.quantization.fp8_utils` — `requant_weight_ue8m0`, `quant_weight_ue8m0`,
  `transform_scale_ue8m0`, `_get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl` (packing spec).
- `testbench/harness/{driver,correctness,timing,reward_hack,inputs,metrics}.py` — gate/timing/anti-hack semantics.
- `testbench/bin/evaluate.py` — authoritative gate; `harness.profile` — advisory metrics (`bound`, `pct_of_hbm_peak`; HBM_GBPS=8000, FP_PEAK_TFLOPS=2250, ridge≈281).
- `testbench/knowledge/entries/glm52--o_proj_decode--b200--20260714a.json` — prior no-win session; memory-bandwidth bottleneck; DeepGEMM-wrapper bypasses do not improve CUPTI timing; sglang-triton mis-read scales; custom-kernel direction not ruled out.
- KernelWiki (Blackwell FP8/DeepGEMM/CUTLASS/tcgen05) and `ncu-report-skill` for profiling.

## Dependencies and Sequence

### Milestones
1. **M1 — Secure correctness (independent lower bound).**
   - Phase A: Capability probe — compile a trivial FP8 `tl.dot` (fp32 acc) on this
     B200/Triton 3.6.0 to confirm tensor-core FP8 support; if it fails, note the manual-accumulate fallback.
   - Phase B: Write the shared UE8M0 unpack helper + pure-torch reconstruction oracle;
     unit-check against §1.6 values.
   - Phase C: Install C2 as `solution.py`; run the authoritative gate; confirm `correct=true`
     on M=16 and M=32 (Phase-1 exit criterion met safely). Log to `kda/candidates.jsonl` + `kda/benchmark.csv`.
2. **M2 — Independent optimizable kernel (preferred deliverable).**
   - Step 1: Implement C1 (Triton) with `BLOCK_M=M`, `BLOCK_N=128`, 16-block K loop,
     inline scale decode, per-block fp32 promotion; diff vs the C2 oracle at both shapes.
   - Step 2: Promote C1 to `solution.py` once it passes the gate on both shapes.
   - Step 3: Exploit weight-scale broadcast (one scale per 128-row N-block) and keep x resident.
3. **M3 — Performance toward the roofline (Phase-2+ direction, non-gating).**
   - Step 1: Autotune BLOCK_N ∈ {64,128,256}, num_warps ∈ {4,8}, persistent/split-N to fill 148 SMs.
   - Step 2: Profile with `ncu-report-skill` (achieved DRAM %, tail effect, tensor-core use); artifacts under `kda/profile/`.
   - Step 3: Only if Triton stalls well below the BW floor, prototype the CuTe/CUTLASS `tcgen05` path (C3).

Dependencies: M1.C depends on M1.B (unpack helper + oracle). M2.Step1 depends on M1.B (oracle for diffing) and M1.A (dot capability). M2 supersedes C2 as `solution.py` once green. M3 depends on a correct C1 from M2. C3 is gated on M3 profiling evidence.

## Task Breakdown

Each task has exactly one routing tag (`coding` = Claude implements; `analyze` = Codex via `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | FP8 `tl.dot` capability probe on B200/Triton 3.6.0 (standalone; decide tensor-core vs manual-accumulate fallback) | AC-2, AC-5 | coding | - |
| task2 | Shared UE8M0 unpack helper + pure-torch reconstruction oracle; unit-check vs verified §1.6 values | AC-3 | coding | - |
| task3 | Install C2 (pure-torch dequant+GEMM) as `solution.py`; run authoritative gate; confirm correct on M=16 & M=32 | AC-1, AC-3, AC-4, AC-6 | coding | task2 |
| task4 | Implement C1 Triton weight-streaming blockwise-scaled FP8 GEMM (BLOCK_M=M, BLOCK_N=128, 16-block K, per-block promotion); diff vs C2 oracle | AC-1, AC-2, AC-3 | coding | task1, task2 |
| task5 | Promote C1 to `solution.py`; re-run gate on both shapes; log candidate + benchmark rows | AC-1, AC-4, AC-5, AC-6 | coding | task3, task4 |
| task6 | Optimize C1: weight-scale broadcast reuse, x residency, occupancy fill (persistent/split-N), autotune BLOCK_N/warps | AC-5 | coding | task5 |
| task7 | Profile C1 with Nsight Compute (DRAM %, tensor-core use, tail); write findings under `kda/profile/` | AC-5 | analyze | task5 |
| task8 | Decision gate: if Triton stalls below BW floor, scope a CuTe/CUTLASS tcgen05 prototype (C3) | AC-5 | analyze | task7 |

## Claude-Codex Deliberation

### Agreements
- The op is memory-bandwidth-bound (weight stream dominant); confirmed by both fresh
  measurement (~1.43–1.51 TB/s, ~18–19% of 8 TB/s peak) and the prior knowledge entry.
- Scales must be applied **per 128-wide K-block** before summing across K (rank-1
  outer-product promotion); `BLOCK_K` fixed at 128.
- A serious speed candidate must be a single device kernel using FP8 tensor-core `tl.dot`,
  avoid materializing a full dequantized weight, and consume the provided inputs verbatim.
- Weight scale is broadcast across the 128 rows of each N-block; align `BLOCK_N` to 128 and
  load one scale vector per N-block rather than per row. Do not split M across programs.
- Final verification is `evaluate.py` (CUPTI device-kernel), not CUDA-event or `harness.profile` numbers.
- C2 (pure-torch reconstruction) is a valid correctness lower bound and dev-time oracle, not the final speed deliverable.

### Resolved Disagreements
- **"Triton may silently scalarize FP8 / can it even beat DeepGEMM?"** (Codex CORE_RISKS/TECHNICAL_GAPS).
  Resolution: added task1 capability probe and an explicit requirement to verify tensor-core
  emission (IR/PTX/NCU); the manual fp32-accumulate path is demoted to a *correctness*
  fallback only. Beating the baseline remains non-gating for Phase-1, so a correct-but-slower
  Triton kernel still satisfies the exit bar.
- **"Weight-scale per-row load is wasteful."** Resolution: folded the 128-row broadcast
  exploitation and BLOCK_N=128 alignment into task6 and the conceptual approach.
- **"Occupancy: 48 N-tiles < 148 SMs."** Resolution: added persistent/split-N scheduling as
  an explicit optimization lever (task6), and flagged that `num_stages` autotune may not help.
- **"Prior art says DeepGEMM is the safe path / no-win."** Resolution: the prior entry only
  explored DeepGEMM-wrapper variants and an *incorrect* sglang-triton path, and its own caveat
  does not rule out a custom kernel changing the device work — which is exactly C1 with the
  verified unpack. The plan proceeds with C1 as the preferred vehicle while keeping C2 as the
  guaranteed-correct floor.
- **"UE8M0 exponent edge values (byte 0/255)."** Resolution: decode via
  `bitcast_i32_to_f32(e<<23)` — e=0 yields +0.0 (matches an all-zero block), normal e∈[1,254]
  yields 2^(e-127); the verified reconstruction (matched_ratio 1.0) shows no problematic values
  in-distribution. No special-casing required; keep the bitcast form.

### Convergence Status
- Final Status: `partially_converged` (generated in `--direct` mode: Codex first-pass
  analysis was incorporated, but the iterative second-Codex convergence loop and manual review
  were intentionally skipped per direct mode; open questions are carried below as pending decisions).

## Pending User Decisions

These consolidate Codex `QUESTIONS_FOR_USER`. Per the autonomous KDA session, each carries a
recommended default (Claude Position) the implementation will follow unless overridden; they
remain `PENDING` for explicit confirmation but do not block Phase-1 (correctness-only) work.

- DEC-1: Is Phase-1 judged purely on `correct=true` for M=16 and M=32, or must the candidate
  also avoid an obviously-slow multi-kernel torch path even while correct?
  - Claude Position: Land C2 (torch) to secure correctness immediately, then supersede it with
    C1 (Triton) as the deliverable; C2 alone is acceptable only as the lower bound.
  - Codex Position: N/A - open question.
  - Tradeoff Summary: `phase1.md` requires "a path that can later be optimized (not a thin
    re-export)"; C2 is independent but weakly optimizable, so C1 is preferred if time allows.
  - Decision Status: PENDING

- DEC-2: Is a compiled CUDA/C++ extension acceptable in Phase-1, or should the first
  implementation stay Python/Triton-only?
  - Claude Position: Prefer Triton for Phase-1 (fastest to a correct independent kernel);
    treat CUDA/CUTLASS/CuTe as a Phase-2+ stretch (C3). `phase1.md` explicitly permits CUDA extensions.
  - Codex Position: N/A - open question.
  - Tradeoff Summary: CUDA/CUTLASS has a higher ceiling but higher risk/cost for a first correct build.
  - Decision Status: PENDING

- DEC-3: Are the observed input layouts (little-endian int32 packing, MN-major, 128-row
  weight-scale broadcast) guaranteed by the task contract, or only empirically observed here?
  - Claude Position: Treat them as stable (derived from sglang source + verified numerically),
    but keep the UE8M0 decode centralized so a layout change is a one-line fix; add a cheap
    shape/stride assertion in `solution.py`.
  - Codex Position: N/A - open question.
  - Tradeoff Summary: Hardcoding assumptions is fast but brittle; a guarded decode is safe.
  - Decision Status: PENDING

- DEC-4: May `solution.py` use shape-specialized dispatch for M=16 and M=32 (different
  tiling/occupancy) with an assertion/fallback for other M?
  - Claude Position: Yes — allowed; specialize for the two sweep shapes with a correct generic
    fallback so unexpected M is still correct (never masking a measured workload).
  - Codex Position: N/A - open question.
  - Tradeoff Summary: Specialization helps occupancy tradeoffs but must not silently break other M.
  - Decision Status: PENDING

- DEC-5: Should the plan target a later SGLang drop-in integration win for
  `glm52/o_proj_decode`, or is this phase scoped strictly to harness correctness?
  - Claude Position: Scope Phase-1 to harness correctness; leave drop-in integration to a later phase.
  - Codex Position: N/A - open question.
  - Tradeoff Summary: Integration adds contract work (`integrate.py`) not needed for the Phase-1 exit bar.
  - Decision Status: PENDING

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as
  "AC-", "Milestone", "Phase", "Step", "task1", or similar workflow markers. These belong to
  this plan document only.
- Use descriptive, domain-appropriate names in `solution.py` (e.g. `decode_ue8m0_scale`,
  `block_scaled_fp8_gemm`, `BLOCK_N`, `K_BLOCKS`).
- Keep the UE8M0 decode in one well-named helper so the (verified) layout assumption is
  centralized and testable.
- The candidate must do real work on every call, return a concrete CUDA bf16 tensor, and not
  rely on timer patching or lazy outputs.

## Output File Convention

This template is used to produce the main output file (`plan.md`). `alternative_plan_language`
resolved to empty (English) via merged Humanize config, so no translated variant is written.
All identifiers (`AC-*`, task IDs, file paths, API names, command flags) are language-neutral.

--- Original Design Draft Start ---

# KDA Phase-1 Draft — glm52/o_proj_decode

**Task:** GLM-5.2 Attention O-Projection, decode phase, TP1 (full 64 heads).
**Op:** FP8 blockwise GEMM `out[M,6144] = x_fp8[M,16384] @ w_fp8[6144,16384].T`.
**Target:** NVIDIA B200 (sm_100). **Baseline:** SGLang production `deep_gemm w8a8_block_fp8`.
**Phase-1 goal:** first *correct, independent* B200 implementation that replaces
`solution.py` and matches `reference.py` within tolerance on sweep shapes `[16, 32]`.
Beating the baseline is desirable but **not required** to exit Phase 1.

> Non-negotiable design constraint (from `phase1.md`): calling the exact baseline
> kernel (`w8a8_block_fp8_matmul_deepgemm`) as the only body of `run()` is **not** an
> acceptable Phase-1 deliverable. The candidate must compute the GEMM independently and
> be a path that can later be optimized.

---

## 1. Baseline behavior + validation (measured, this worktree)

### 1.1 What the op is
- `reference.py` reads axes from `definition.json` and calls
  `w8a8_block_fp8_matmul_deepgemm(x_fp8, w_fp8, x_scale, w_scale, [128,128], output_dtype=bf16)`.
- Constant axes: `K = 16384` (= local_heads·v_head = 64·256), `N = 6144` (hidden).
  Variable axis `M ∈ {16, 32}` (decode token/batch count).
- `K_scale_blocks = K // 512 = 4` (this is the *packed* int32 count, not the block count).

### 1.2 Exact input contract (empirically observed via `get_inputs`)
`run(x_fp8, x_scale, w_fp8, w_scale)` receives, after the harness `clone_args()`
(which produces contiguous clones — the M<4 TMA-padding gap is stripped, but M≥4 here so
shapes are unaffected):

| tensor    | shape        | dtype             | layout (pre-clone)         |
|-----------|--------------|-------------------|----------------------------|
| `x_fp8`   | `[M, 2048]`  | `float8_e4m3fn`   | row-major, contiguous      |
| `x_scale` | `[M, 4]`     | `int32`           | **column-major** `(1, M)`  |
| `w_fp8`   | `[6144,2048]`| `float8_e4m3fn`   | row-major, contiguous      |
| `w_scale` | `[6144, 4]`  | `int32`           | **column-major** `(1,6144)`|

Output: `[M, 6144]` `bfloat16`.

**The scales are UE8M0-packed int32, not floats.** This is the single most important
fact for an independent implementation. Decoded chain (confirmed by reading
`sglang.srt.layers.quantization.fp8_utils`):
- Weight scale is a genuine 128×128 block scale `[N//128, K//128] = [48, 16]`, then
  `transform_scale_ue8m0` **broadcasts it across the 128 rows of each N-block**
  (`index_select(-2, arange(N)//128)`) to `[6144, 16]`, then packs 4 K-block exponents
  per int32 → `[6144, 4]`, stored MN-major (column-major).
- Activation scale is per-token, per-128-K-block: logical `[M, 16]` → packed `[M, 4]`.
- Packing rule (from `_get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl`):
  each scale is a power of two; its E8M0 byte is `(fp32_bits >> 23) & 0xFF` (biased exp,
  bias 127). Four consecutive K-block bytes are packed little-endian into one int32.
  **Decode:** for row `r`, K-block `kb ∈ [0,16)`: `j = kb//4`, `byte = kb%4`,
  `e = (scale_i32[r, j] >> (8*byte)) & 0xFF`, `scale = 2^(e-127)`
  (equivalently `bitcast_i32_to_f32(e << 23)`).

### 1.3 Math the candidate must reproduce
```
sx[m,kb] = 2^(exp_x[m,kb]-127)          # per-token, per-128-K-block
sw[n,kb] = 2^(exp_w[n,kb]-127)          # per-row (broadcast within 128-N-block), per-128-K-block
out[m,n] = sum over 16 K-blocks kb:  sx[m,kb]*sw[n,kb] * ( sum_{i in block} x_fp8[m,ki]*w_fp8[n,ki] )
         -> cast to bf16
```
This is the standard blockwise-scaled FP8 GEMM "promotion" structure: accumulate the
raw FP8·FP8 dot over each 128-wide K-block, scale that partial by the rank-1 outer
product `sx[:,kb] ⊗ sw[:,kb]`, and add into an fp32 accumulator.

### 1.4 Measured baseline + roofline (this B200, CUPTI cold-L2, harness timer)
- **M=16: 9.02 µs**, **M=32: 8.72 µs** (median device-kernel span).
- Bytes moved per call ≈ **12.9 MB**, dominated by the `w_fp8` weight stream (12.58 MB).
  Activations (32–64 KB), scales (~0.1 MB), output (0.2–0.4 MB) are negligible.
- Effective bandwidth ≈ **1.43–1.51 TB/s** — only **~18–19%** of B200 HBM peak (~8 TB/s).
- Roofline floor at 8 TB/s ≈ **1.6 µs**. The 12.58 MB weight fits in the 126 MiB L2, but
  the harness flushes L2 each iteration → every call re-reads the weight from HBM, so this
  is genuinely an **HBM-bandwidth-bound weight stream**.
- **Interpretation:** deep_gemm is tuned for large M; at M=16/32 the UMMA M-tile is
  mostly idle and N=6144 tiling under-fills 148 SMs, leaving ~5× bandwidth headroom.
  Beating 9 µs is very plausible for a weight-streaming kernel (even 50% of peak BW ≈ 3 µs).

### 1.5 Correctness oracle + gate
- Oracle = live `reference.py` output. Tolerance: `max_atol=0.1`, `max_rtol=0.05`,
  `required_matched_ratio=0.999` (torch.allclose-style matched-ratio; inf/nan → fail).
- Gate exit codes (`evaluate.py`): `0` WIN (correct **and** faster on every shape),
  `1` correct-but-slower, `2` incorrect. Phase-1 success = `correct=true` on both shapes.
- Anti-reward-hack (`harness/reward_hack.py`): no monkey-patching
  `torch.cuda.Event.elapsed_time`; outputs must be real `torch.Tensor` (no lazy/proxy);
  a post-timing recheck re-verifies one fresh call against the oracle. **The candidate must
  do real work every call and not depend on `x_scale`/`w_scale` being pre-decoded.**

### 1.6 De-risking experiment already run (pure-torch reconstruction)
Unpacked the UE8M0 int32 scales as in §1.2, did `dequant → fp32 matmul → bf16`, compared
to deep_gemm:
- **M=16: max_abs 0.00195, matched_ratio 1.000000**
- **M=32: max_abs 0.00781, matched_ratio 1.000000**

→ The correctness path is *proven*: an independent dequant+GEMM reproduces the oracle
with ~15–50× margin under the atol. Remaining work is purely about doing it *fast* and
*independently* in a kernel.

---

## 2. Risks / unknowns

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| R1 | UE8M0 int32 scale layout mis-decoded (byte order, MN-major, K-block index). | ~~High~~ **Retired** | Decoded from sglang source + verified numerically (§1.6). Byte order little-endian, `kb=4j+byte`, power-of-two `2^(e-127)`. |
| R2 | Triton FP8 (`e4m3`) `tl.dot` support / correctness on sm_100 + Triton 3.6.0. | Medium | First step = isolated capability probe (§4 step 0) before committing. Fallback to manual fp32 accumulate (no `tl.dot`) since M is tiny and the op is BW-bound, not MMA-bound. |
| R3 | Column-major int32 scale loads awkward / uncoalesced in Triton. | Low | Scales are tiny (≤0.1 MB) and re-used across all K-iters of a tile; decode once per tile into registers/SMEM. Load pattern coalesces along M (x) / N (w) which are the tile dims. |
| R4 | M=16 is the MMA min; skinny-M efficiency. | Low (BW-bound) | Compute is not the bottleneck (roofline 1.6 µs vs 9 µs baseline). `BLOCK_M = M` (16 or 32, both multiples of 16) avoids padding. If `tl.dot` underperforms at M=16, fold M into the fp32 reduction. |
| R5 | Not beating baseline. | Low for Phase-1 | Phase-1 does **not** require a WIN. Correct+independent is the exit bar; perf is Phase-2/3. |
| R6 | Numerical drift vs deep_gemm's exact accumulation order. | Low | atol=0.1 is generous; §1.6 shows 0.002–0.008 actual. fp32 accumulation with per-block promotion is well within margin. |
| R7 | `run()` must accept the exact reference input signature/layout (cannot re-quantize; `get_inputs` is owned by reference). | Medium | Design consumes `(x_fp8, x_scale, w_fp8, w_scale)` verbatim; no re-quant. Handle both contiguous clone and (defensively) original strides. |
| R8 | JIT/compile overhead counted? | Low | Timer measures device-kernel span only; warmup (10 iters) triggers Triton autotune/compile before timing. Keep the kernel launch-light (single kernel, no host-side scale unpacking each call). |

---

## 3. Ranked candidate directions

### C1 — Triton FP8 blockwise-scaled GEMM, weight-streaming (PRIMARY)
- One Triton kernel; grid over N-tiles (and optionally M, but `BLOCK_M=M` suffices).
- Each program loads its `w_fp8[BLOCK_N, K]` slice **once** (the whole game is streaming
  the 12.58 MB weight at high BW), `x_fp8[M, K]` (tiny, hot in L2/regs), decodes the
  relevant UE8M0 scale columns inline (bit-shift → `exp2`), loops K in 128-wide blocks,
  accumulates `sum(x·w)` per block, scales by `sx[:,kb] ⊗ sw[:,kb]`, adds to fp32 acc,
  writes bf16.
- Levers: `BLOCK_N` (128/256), vectorized fp8 loads, `num_warps`/`num_stages` autotune,
  optional N-split for SM occupancy (148 SMs vs 48 N-tiles at BLOCK_N=128 → try smaller
  BLOCK_N or 2D grid to fill the machine).
- Pros: independent, fully optimizable toward the BW roofline, moderate effort, stays in
  the harness `.venv` (Triton 3.6.0 present). Best expected-value Phase-1 deliverable.
- Cons: R2 (fp8 dot support) — de-risked by step 0.

### C2 — Pure-torch dequant + GEMM (CORRECTNESS SCAFFOLD / SAFETY NET)
- Exactly the §1.6 reconstruction: unpack scales, dequant to bf16, `@`, cast. Independent
  of deep_gemm; guaranteed correct (already verified). Slow (materializes a 25 MB bf16
  weight + a bf16 GEMM) → will not beat baseline.
- Role: (a) landed first as a *known-correct* independent `solution.py` so Phase-1's
  correctness bar is met immediately and safely; (b) serves as the numerical oracle to
  diff C1 against during development. Not the final deliverable, but a legitimate
  independent fallback.

### C3 — CuTe-DSL / CUTLASS Blackwell FP8 blockwise GEMM (STRETCH / Phase-2+)
- `tcgen05` UMMA consuming the UE8M0 SF tensors natively (MN-major TMA-aligned), 2-SM
  cooperative, TMA weight loads. Highest performance ceiling.
- Pros: can approach HBM peak; the "right" long-term kernel.
- Cons: high implementation cost/risk for Phase-1; defer until C1 establishes a correct
  baseline and profiling justifies the complexity.

### C4 — deep_gemm retuning / lower-level primitives (REJECTED as primary)
- Re-dispatching deep_gemm with different configs is still "the baseline op" → violates
  the Phase-1 independence constraint. Not pursued. (Prior art confirms: DeepGEMM-wrapper
  bypasses and num_sms sweeps did not improve CUPTI device-kernel timing.)

**Recommendation:** land **C2** first (immediate correct independent baseline), then build
**C1** as the real Phase-1 deliverable and optimization vehicle; keep **C3** as the
Phase-2/3 stretch.

---

## 4. First concrete steps

0. **Triton FP8 capability probe** (de-risk R2, standalone — not the solution): compile a
   trivial `tl.dot` of two `float8_e4m3fn` tiles with fp32 acc on this B200/Triton 3.6.0,
   confirm numerical sanity vs a torch reference. If it fails, switch C1's inner product to
   a manual fp32 multiply-accumulate (viable because the op is BW-bound, not MMA-bound).
1. **Land C2 as `solution.py`** — the verified unpack+dequant+matmul. Run the authoritative
   gate; confirm `correct=true` on M=16 and M=32. This secures the Phase-1 exit criterion.
2. **Write a UE8M0 unpack helper** (shared, tested against §1.6 values) and a tiny
   host-side numeric oracle for kernel diffing.
3. **Implement C1 (Triton)** with `BLOCK_M=M`, `BLOCK_N=128`, K-loop of 16 blocks,
   inline scale decode, fp32 promotion. Diff against the C2 oracle at M=16/32.
4. **Autotune C1** over `BLOCK_N ∈ {64,128,256}`, `num_warps ∈ {4,8}`,
   `num_stages ∈ {2,3,4}`, and a 2D (N × N-split) grid to fill 148 SMs. Target the ~1.6 µs
   BW floor; success criterion for a WIN is `< 9 µs` (M=16) / `< 8.7 µs` (M=32).
5. **Profile with `ncu-report-skill`** if C1 stalls below expected BW; check achieved DRAM
   throughput, tail effect, and whether the weight read is the sole bottleneck. Record NCU
   artifacts under `kda/profile/`.
6. **Record** every candidate in `kda/candidates.jsonl` and every measured run in
   `kda/benchmark.csv` (scaffolds created).

---

## 5. Exact validation commands

Fast advisory smoke (per shape):
```bash
cd /home/qinhaiyan/KDA-Exp/worktrees/glm52-o_proj_decode
PY=/home/qinhaiyan/Kernel-Harness/.venv/bin/python
PYTHONPATH=testbench $PY -m harness.profile \
  testbench/tasks/glm52/o_proj_decode --shape 16
```

Authoritative gate (correctness + CUPTI cold-L2 timing; exit 0=WIN,1=slower,2=incorrect):
```bash
cd /home/qinhaiyan/KDA-Exp/worktrees/glm52-o_proj_decode
PY=/home/qinhaiyan/Kernel-Harness/.venv/bin/python
$PY testbench/bin/evaluate.py testbench/tasks/glm52/o_proj_decode --max-workloads 1   # M=16 only, fast
$PY testbench/bin/evaluate.py testbench/tasks/glm52/o_proj_decode                     # both shapes
```
(`./run.sh` in the task dir forwards to `evaluate.py`; prefer the `$PY` form so the shared
`.venv` is used.)

Only `solution.py` may be edited; never edit `reference.py`.

---

## 6. Evidence rules — promote / revise / reject

- **PROMOTE** a candidate to `solution.py` when: authoritative `evaluate.py` reports
  `correct=true` (matched_ratio ≥ 0.999) on **both** M=16 and M=32, output dtype is
  bf16, no nan/inf, and it passes the post-timing recheck (no reward-hack flag). This
  alone satisfies Phase-1 exit. Log to `kda/candidates.jsonl` + `kda/benchmark.csv`.
- **REVISE** when: correct on M=16 but not M=32 (or vice versa) — inspect scale-decode
  boundary / tiling for that M; **or** correct but slower than baseline — keep as the
  standing correct solution and iterate on tiling/occupancy/BW (Phase-2 territory).
- **REJECT** when: any shape incorrect (tolerance/nan/inf), a reward-hack flag fires, the
  body is a thin re-export of `w8a8_block_fp8_matmul_deepgemm`, or it depends on inputs
  the harness does not provide. Revert `solution.py` to the last PROMOTED candidate.
- **Baseline bar for a WIN (not required in Phase-1):** device-kernel median
  `< 9.0 µs` (M=16) and `< 8.7 µs` (M=32) on this B200; roofline aspiration ≈ 1.6 µs.

---

## 7. Summary of the plan
1. Immediately secure correctness with the verified independent pure-torch dequant+GEMM
   (C2) as `solution.py`.
2. Build the real deliverable: a Triton weight-streaming blockwise-scaled FP8 GEMM (C1)
   that decodes the UE8M0 scales inline and streams the 12.58 MB weight once, diffed
   against the C2 oracle and tuned toward the HBM roofline.
3. Keep a CuTe/CUTLASS `tcgen05` path (C3) as the Phase-2/3 stretch for closing the
   remaining gap to peak bandwidth.

The correctness risk is retired; the open question is how close a clean weight-streaming
kernel gets to the ~1.6 µs floor versus deep_gemm's ~9 µs.

--- Original Design Draft End ---
