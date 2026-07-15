# KDA Phase-1 Plan — `glm52/moe_gate_proj_prefill` (Independent Masked FP8 Grouped GEMM on B200)

> **Retarget note (2026-07-15):** Harness shapes updated to **DP=1 / TP=1 / EP=32** (commit `88d7044`). Numeric axes below were patched in-place; re-benchmark after this change.


## Goal Description

Replace `testbench/tasks/glm52/moe_gate_proj_prefill/solution.py:run()` with a
**correct, independent** B200 (sm_100) implementation of the masked FP8 grouped
GEMM `out[e] = a[e] @ b[e].T` (E=8 experts, K=6144, N=2048, `layout=1`), that
matches the live `reference.py` oracle within tolerance on every sweep shape
`M ∈ {1024, 2048, 4096}`.

The implementation must **not** be a thin re-export of the baseline kernel:
calling `deep_gemm.fp8_m_grouped_gemm_nt_masked` as the computational body of
`run()` is disallowed. The deliverable should be a genuinely independent kernel
that is later optimizable (single-launch, on-device control flow), with a
validated correctness oracle used during development.

Phase-1 exit is **correctness-only**: the authoritative `evaluate.py` must report
`correct=true` on all three shapes. Beating the baseline latency is desirable and
measured throughout, but is **not** a blocking exit condition for Phase 1.

Only `testbench/tasks/glm52/moe_gate_proj_prefill/solution.py` may be edited.
`reference.py`, `definition.json`, `workload.jsonl`, tolerances, and the harness
must not be modified.

## Acceptance Criteria

- AC-1: **Full-output correctness on every sweep shape.** The authoritative
  `evaluate.py` reports `correct=true` for M=1024, 2048, 4096 (matched_ratio ≥
  0.999 against the full `out[E,Mp,N]` tensor at `atol=0.1, rtol=0.05`), with no
  NaN/Inf in the candidate output. This gate is a **hard requirement** (from
  `task.json`).
  - Positive Tests (expected to PASS):
    - `python testbench/bin/evaluate.py testbench/tasks/glm52/moe_gate_proj_prefill`
      reports `correct=true` for all three workloads; exit code is 0 (WIN) or 1
      (correct-not-faster).
    - `python testbench/bin/evaluate.py ... --max-workloads 1` reports
      `correct=true` for M=1024.
    - Re-running the gate a second time reproduces `correct=true` on all shapes.
  - Negative Tests (expected to FAIL / be rejected):
    - A candidate that computes **all** `Mp` rows unconditionally fails at M=4096
      (mismatch on reference's skipped 128-row blocks pushes matched_ratio < 0.999).
    - A candidate that writes only rows `[0, masked_m[e])` (not full 128-row
      blocks) fails at M=1024/2048 (rows `masked_m[e]..127` of every expert differ
      from the reference's fully-computed block).
    - A candidate that leaves padded rows uninitialized / NaN trips the NaN/Inf
      sanity gate and is reported `INCORRECT`.
  - AC-1.1: **Block-granular write pattern.** For each expert `e`, the candidate
    overwrites exactly rows `[0, ceil(masked_m[e]/128)*128)` (clamped to `Mp`) of
    the provided `out` and leaves the remaining rows byte-identical to the
    incoming buffer.
    - Positive: local check — after `run()`, rows `[0, ceil(masked_m[e]/128)*128)`
      equal the manual dequant matmul and rows above equal the incoming `out`
      clone, for every expert, at M=4096 (mixed 128/256 written rows).
    - Negative: writing 256 rows for an expert whose `masked_m[e] ≤ 128` at
      M=4096 corrupts a skipped block and fails the full-tensor compare.
  - AC-1.2: **Numeric faithfulness margin.** Over the reference-written rows, the
    candidate's max absolute error stays well under `atol` (target ≤ 0.05,
    observed 0.0156 for the fp32-accumulate reference math).
    - Positive: local harness reports `max_absolute_error < 0.05` and
      `matched_ratio == 1.0` over written rows on all shapes.
    - Negative: an implementation that accumulates FP8 products in FP16/BF16
      (instead of FP32) or applies the per-128-K block scales incorrectly drives
      max error above `atol` on at least one shape.

- AC-2: **Independence from the baseline kernel.** `run()` computes the result
  with an independent kernel and does not call
  `deep_gemm.fp8_m_grouped_gemm_nt_masked` (nor `m_grouped_fp8_fp4_gemm_nt_masked`,
  its alias) as its computational body.
  - Positive Tests (expected to PASS):
    - `grep` of `solution.py` shows no call to `fp8_m_grouped_gemm_nt_masked` /
      `m_grouped_fp8_fp4_gemm_nt_masked` in the `run()` path; the GEMM is produced
      by the project's own Triton (or CUDA/CUTLASS) kernel.
    - Using `deep_gemm.utils.math` helpers for **scale unpacking only**
      (`unpack_ue8m0_from_int`) is permitted (data-layout helper, not the GEMM).
  - Negative Tests (expected to FAIL / be rejected):
    - A `run()` whose body is `deep_gemm.fp8_m_grouped_gemm_nt_masked(...)` (with
      or without stripped `.item()` syncs) is rejected as a non-independent
      deliverable even if it passes correctness.

- AC-3: **Exact scale recovery from packed UE8M0.** The candidate reconstructs
  the plain per-128-K UE8M0 scales for A and B from the packed int32 `a_s`/`b_s`
  and applies them per 128-element K block.
  - Positive Tests (expected to PASS):
    - Round-trip unit check (known plain UE8M0 scales →
      `transform_sf_into_required_layout` → candidate's inverse) is bit-exact
      (`torch.equal`).
    - A/B dequant uses `[E,Mp,K/128]` and `[E,N,K/128]` scale grids respectively
      (48 K-scales for K=6144), consistent with `a_s`/`b_s` shape `[...,12]` (12×4).
  - Negative Tests (expected to FAIL / be rejected):
    - Interpreting the 12 packed lanes as 12 K-groups (instead of 48) or using the
      wrong byte order yields incorrect output and fails AC-1.
    - Calling `unpack_ue8m0_from_int` on the non-contiguous `a_s` without
      `.contiguous()` raises (last-dim stride ≠ 1) — the candidate must handle
      contiguity defensively.

- AC-4: **No per-call host-device syncs or object-reuse caching.** Control flow
  reads `masked_m` on-device; `run()` performs no `.item()`/`.cpu()`/`.tolist()`
  reads of input tensor values in the timed path, and does not cache results
  across calls.
  - Positive Tests (expected to PASS):
    - `run()` returns a fresh, genuinely-computed `torch.Tensor` every call; the
      driver's post-timing re-check still reports `correct=true` (no `REWARD_HACK`).
    - No `.item()`/`.cpu()`/`.tolist()` on `masked_m`/`expected_m`/`layout` in the
      timed path (a single host read of shape metadata is acceptable; reading
      tensor *values* per expert is not).
  - Negative Tests (expected to FAIL / be rejected):
    - Building the launch grid from `masked_m.tolist()` reintroduces device→host
      syncs (the overhead prior art specifically removed).
    - A candidate that precomputes on call 1 and returns cached buffers on later
      calls is caught by the driver post-timing recheck and reported `REWARD_HACK`.

- AC-5: **Evidence artifacts recorded.** Candidates and runs are logged, and NCU
  profiles (if collected) are stored under the campaign directories.
  - Positive Tests (expected to PASS):
    - `kda/candidates.jsonl` gains an entry per candidate; `kda/benchmark.csv`
      records per-shape latency vs the recorded baseline.
    - Any Nsight Compute report lives under `kda/profile/`.
  - Negative Tests (expected to FAIL / be rejected):
    - A "win" claim with no corresponding `kda/benchmark.csv` row or evaluate
      output is not accepted as evidence.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
A single-launch, independent **Triton** block-scaled FP8 grouped-masked GEMM
(candidate **C1**) that: recovers/decodes the UE8M0 scales (ideally decoded
inside the kernel to stay single-launch), accumulates FP8×FP8 in FP32 with
per-128-K block scaling, consumes `masked_m` on-device to honor the 128-row block
write pattern, writes BF16 into the provided `out`, passes `evaluate.py`
`correct=true` on all shapes, and is measured against the baseline with a shape
config (`BLOCK_N`, `num_warps`, `num_stages`, optional `Mp` specialization) tuned
for the fixed constants. Verified to actually emit FP8 tensor-core codegen on
sm_100. Optionally a CUDA/CUTLASS/CuTe-DSL grouped-GEMM variant (C3) is explored
only after C1 is correct and profiled.

### Lower Bound (Minimum Acceptable Scope)
An independent, correct implementation (candidate **C2**): recover the plain
scales from `a_s`/`b_s`, dequantize A and B, compute per-expert matmuls (e.g.
`torch.bmm`) over the block-rounded `ceil(masked_m[e]/128)*128` rows, and write
BF16 into `out` while preserving skipped-block rows. This satisfies AC-1..AC-4
(it does not call the baseline GEMM) and yields the Phase-1 correctness exit even
if it does not beat baseline latency. It also serves as the in-process oracle for
developing C1.

### Allowed Choices
- Can use: Triton 3.6 (`tl.dot` FP8 with FP32 accumulate on sm_100), CUDA C++ via
  `torch.utils.cpp_extension.load_inline`, CUTLASS 4.5 / CuTe-DSL grouped FP8
  GEMM, custom helper kernels/JIT functions defined inside `solution.py`,
  `deep_gemm.utils.math.unpack_ue8m0_from_int` and other **data-layout** helpers,
  `torch.bmm`/`torch.matmul` for the oracle path, runtime inference of shapes
  (`E, Mp, N, K`) from the input tensors.
- Cannot use: `deep_gemm.fp8_m_grouped_gemm_nt_masked` /
  `m_grouped_fp8_fp4_gemm_nt_masked` (or any DeepGEMM grouped-GEMM entry point) as
  the computational body; edits to any file other than `solution.py`; per-expert
  host reads of `masked_m` values in the timed path; caching results across calls;
  timer/monkey-patch reward hacks; assuming padded rows may be left non-finite.

> **Note on Determinism:** the numerics (FP8 e4m3 operands, per-128-K UE8M0 block
> scaling, FP32 accumulation, BF16 output) and the 128-row block write pattern are
> fixed by the reference contract and validated empirically; these are not free
> choices. The free choices are the kernel technology (Triton/CUDA/CUTLASS) and
> its tiling/scheduling configuration.

## Feasibility Hints and Suggestions

> Reference only — one possible path, not prescriptive.

### Conceptual Approach

Scale recovery (validated bit-exact):
```python
from deep_gemm.utils.math import unpack_ue8m0_from_int
KB = K // 128                                   # 48 for K=6144
a_scale = unpack_ue8m0_from_int(a_s.contiguous())[..., :KB]   # [E, Mp, 48] fp32 (pow2)
b_scale = unpack_ue8m0_from_int(b_s.contiguous())[..., :KB]   # [E, N,  48] fp32 (pow2, per-row)
```

Correctness oracle / lower-bound (C2):
```python
E, Mp, N = out.shape; K = a_fp8.shape[-1]
a_deq = (a_fp8.float().view(E, Mp, KB, 128) * a_scale.unsqueeze(-1)).view(E, Mp, K)
b_deq = (b_fp8.float().view(E, N,  KB, 128) * b_scale.unsqueeze(-1)).view(E, N, K)
prod  = torch.bmm(a_deq, b_deq.transpose(1, 2)).to(torch.bfloat16)   # [E, Mp, N]
# write only block-rounded rows, preserve the rest of `out`:
for-block-granular copy of rows [0, ceil(masked_m[e]/128)*128) into out
return out
```

Primary deliverable (C1), single Triton launch, grid `(E, n_blocks, m_blocks)`:
- Skip `m_block` when `m_block*128 >= ceil(masked_m[e]/128)*128` (read `masked_m`
  from device; compute the block bound with integer ops, no host sync).
- Loop K in 128-chunks: `acc += tl.dot(a_tile_fp8, b_tile_fp8.T)  *
  a_scale[m, kb] * b_scale[n, kb]` in FP32. Decode the UE8M0 exponent byte to a
  scale inside the kernel (`scale = reinterpret((byte.to(int32) << 23))`) to keep
  the path single-launch, or pass the pre-decoded FP32 scale tensors in (two tiny
  extra kernels — acceptable but not ideal).
- Store BF16, masked on `m_row < write_rows` (block bound) and `n < N` — **not**
  on `m_row < masked_m[e]`.
- Start `BLOCK_M=128` (matches reference granularity), `BLOCK_N=128`,
  `BLOCK_K=128`; tune after correctness. DeepGEMM's SM100 config is 128/128/64.
- **Feasibility gate:** confirm the compiled kernel actually uses FP8 tensor cores
  on sm_100 (inspect PTX/SASS or measure latency consistent with tensor-core
  execution); if Triton 3.6 FP8 blockscale codegen is inadequate, pivot to a
  CUDA/CUTLASS grouped GEMM (C3).

### Relevant References
- `testbench/tasks/glm52/moe_gate_proj_prefill/reference.py` — oracle: masked
  path via `fp8_m_grouped_gemm_nt_masked`; `get_inputs` builds routing/quant offline.
- `testbench/harness/correctness.py` — full-tensor compare, matched-ratio, NaN/Inf gate.
- `testbench/harness/driver.py` — clone semantics, post-timing recheck.
- `testbench/harness/timing.py` — CUPTI cold-L2 device-kernel timing.
- `testbench/harness/reward_hack.py` — caching/lazy/timer defenses.
- `deep_gemm/utils/math.py` — `unpack_ue8m0_from_int`, `pack_ue8m0_to_int`,
  `per_token_cast_to_fp8`, `per_block_cast_to_fp8` (layout semantics).
- `deep_gemm/utils/layout.py` — MN-major TMA-aligned packed-UE8M0 transform.
- `testbench/tasks/glm52/routed_gateup_nvfp4_decode/solution.py` — sibling routed
  grouped-GEMM win (alternate-backend pattern).
- `testbench/knowledge/entries/glm52--routed_down_decode--*.json` — same baseline
  kernel; overhead-bound diagnosis; win from removing `.item()` syncs.
- KernelWiki: `kernel-grouped-gemm`, `kernel-deepgemm`, `kernel-fused-moe`
  (masked layout, UE8M0 4-per-uint32 block scaling, tcgen05, reward-hack warning).
- `kda/probe_sf.py` — the transient probe that established every empirical fact here.

## Dependencies and Sequence

### Milestones
1. **M1 — Evidence seed & harness contract lock.**
   - Phase A: record baseline latency (M=1024/2048/4096) and metrics into
     `kda/benchmark.csv`; seed `kda/candidates.jsonl`.
   - Phase B: pin the runtime contract in `solution.py` (shapes/dtypes/strides,
     `masked_m` device/int32, `expected_m` 0-dim tensor, defensive `.contiguous()`
     on scale tensors).
2. **M2 — Correctness lock via C2 (lower bound).** Depends on M1.
   - Step 1: implement scale recovery + round-trip self-check.
   - Step 2: dequant + per-expert matmul over block-rounded rows, in-place write
     preserving skipped blocks; `layout` handled defensively.
   - Step 3: pass `evaluate.py --max-workloads 1`, then the full gate → `correct=true`.
3. **M3 — Independent optimizable kernel C1 (upper bound).** Depends on M2 (oracle).
   - Step 1: Triton block-scaled FP8 grouped-masked GEMM, single launch,
     on-device `masked_m`, full-block writes.
   - Step 2: validate against C2 oracle in-process (full-tensor equality within tol).
   - Step 3: FP8 tensor-core codegen feasibility gate; pass `evaluate.py` all shapes.
   - Step 4: profile vs baseline (`harness.profile` + `ncu-report-skill` →
     `kda/profile/`); record to `kda/benchmark.csv`.
4. **M4 — Optional tuning / C3 exploration.** Depends on M3 correct.
   - Tile/warp/stage sweep; optional CUTLASS/CuTe grouped-GEMM if C1 leaves
     compute on the table. Non-blocking for Phase-1 exit.

Dependency summary: M1 → M2 → M3 → M4. C2 is a prerequisite oracle for C1. The
Phase-1 exit criterion (AC-1) is satisfiable at M2; M3 upgrades to the preferred
optimizable deliverable; M4 is purely performance.

## Task Breakdown

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Record baseline latency/metrics into `kda/benchmark.csv`; seed `kda/candidates.jsonl` | AC-5 | coding | - |
| task2 | Pin runtime contract in `solution.py` (infer shapes; defensive contiguity; device `masked_m`; no host syncs) | AC-4 | coding | task1 |
| task3 | Implement + self-check exact UE8M0 scale recovery (round-trip bit-exact) | AC-3 | coding | task2 |
| task4 | Implement C2 oracle: dequant + per-expert matmul with block-granular in-place write | AC-1, AC-1.1, AC-1.2, AC-2 | coding | task3 |
| task5 | Gate C2 via `evaluate.py` (`--max-workloads 1`, then full) → `correct=true` all shapes | AC-1 | coding | task4 |
| task6 | Implement C1: single-launch Triton block-scaled FP8 grouped-masked GEMM, on-device `masked_m`, full-block BF16 writes | AC-1, AC-1.1, AC-2, AC-4 | coding | task5 |
| task7 | Validate C1 against C2 oracle in-process (full-tensor within tol) | AC-1.1, AC-1.2 | coding | task6 |
| task8 | FP8 tensor-core codegen feasibility check for C1 on sm_100 (PTX/SASS or latency signature); decide Triton vs CUDA/CUTLASS pivot | AC-1 | analyze | task6 |
| task9 | Gate C1 via `evaluate.py` all shapes; profile vs baseline; write NCU to `kda/profile/`, results to `kda/benchmark.csv` | AC-1, AC-5 | coding | task7, task8 |
| task10 | Optional tuning (`BLOCK_N`/warps/stages/`Mp` specialization) and/or C3 CUTLASS exploration | AC-1 | coding | task9 |

## Claude-Codex Deliberation

### Agreements
- Correctness hinges on **scale interpretation** and the **128-row block write
  pattern**, not the GEMM algebra (dequant + FP32 matmul is bit-faithful within tol).
- Full 128-row blocks must be written for any block containing a valid row; skipped
  blocks must preserve the incoming `out`; computing all `Mp` rows is wrong at M=4096.
- `masked_m` must be consumed on-device; no per-expert host syncs; no result caching
  (driver post-timing recheck enforces this).
- NaN/Inf is a hard gate; padded/skipped regions must stay finite (they equal the
  finite cloned buffer).
- C2 (torch dequant + bmm) is the correctness **oracle**, likely too slow to win;
  C1 (single-launch Triton) is the intended shipped deliverable.
- Triton FP8 blockscale on sm_100 needs an explicit **codegen feasibility gate**;
  CUDA/CUTLASS is the fallback.

### Resolved Disagreements / Clarifications (Codex Analysis v1 questions resolved against the phase contract)
- **"Correctness-only vs must-win?"** → Resolved: `phase1.md` sets Phase-1 exit =
  `correct=true` on all shapes; beating baseline is desirable but non-blocking.
  Performance is tracked (AC-5, M3/M4) but does not gate exit. (See DEC-1 for the
  one residual judgment call.)
- **"Inline CUDA/C++ extensions allowed?"** → Resolved: yes; `phase1.md` explicitly
  lists CUDA C++, CuTe DSL, Triton, Python-wrapped CUDA extensions, CUTLASS. Helper
  kernels defined inside `solution.py` are permitted; watch JIT compile/cache overhead.
- **"Tensors guaranteed contiguous / exact shapes?"** → Resolved: `get_inputs` is
  fixed and `clone_args` preserves layout, so shapes/strides are stable
  (`a_fp8`/`b_fp8`/`out` contiguous; `a_s`/`b_s` MN-major non-contiguous). The
  candidate still calls `.contiguous()` defensively before unpacking scales.
- **"Rely on constants or infer dynamically?"** → Resolved: infer `E, Mp, N, K`
  from tensor shapes at runtime (robust to `Mp∈{128,256}`); it is acceptable to
  tune block sizes for the known constants `E=8, K=6144, N=2048`.
- **"Recover scales on host conflicts with avoiding syncs?"** → Clarified: unpacking
  is a device elementwise op with **no host value read** (not a sync); it only adds
  small launches. For best perf, decode the UE8M0 byte inside the GEMM kernel.
- **Scale-lane reconciliation** → Confirmed: `a_s`/`b_s` last dim 12 packs 4 UE8M0
  bytes each ⇒ 48 = K/128 K-scales (not 12). Recovered scale = `2^(byte-127)`.

### Convergence Status
- Final Status: `partially_converged` (generated in `--direct` mode: Codex
  first-pass analysis was incorporated, but the iterative second-Codex convergence
  loop was skipped by design). Human review of this plan is expected before RLCR.

## Pending User Decisions

- DEC-1: Performance ambition for the Phase-1 close.
  - Claude Position: Land C2 to satisfy the correctness exit immediately, then ship
    C1 (independent, single-launch, optimizable) as the Phase-1 deliverable, and
    only push tuning/C3 (M4) opportunistically. Treat a measured baseline win as a
    Phase-2 objective, not a Phase-1 gate.
  - Codex Position: For an optimization campaign, prefer requiring `evaluate.py`
    exit code 0 (a real win over DeepGEMM), using `harness.profile` only for
    direction; a correct-but-slower path is a checkpoint, not the deliverable.
  - Tradeoff Summary: `phase1.md` explicitly makes correctness the sole Phase-1
    exit gate and says beating baseline is not required; the overhead-bound regime
    (~140µs flat, ~42% HBM) and prior 1.24× wins suggest a win is plausible but not
    guaranteed within Phase-1 scope. Deciding whether Phase-1 must end on a measured
    win changes M3/M4 effort and whether C2 may stand as the final deliverable.
  - Decision Status: `PENDING`

## Implementation Notes

### Quantitative Metric Classification (from the draft/contract)
- **Hard requirements:** `max_atol=0.1`, `max_rtol=0.05`, `required_matched_ratio=0.999`,
  no NaN/Inf — the correctness gate (`task.json`). Must be met on every shape.
- **Advisory / directional (Phase 1):** `solution_us < baseline_us` per shape
  (authoritative WIN). Measured and recorded, but does not block Phase-1 exit.

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such
  as "AC-", "Milestone", "Phase", "Step", "task1", or similar workflow markers.
- Use descriptive, domain-appropriate names (e.g. `write_rows`, `block_m`,
  `k_scale_groups`, `dequant_scale`).
- Keep `solution.py` self-contained; if defining Triton/CUDA kernels inline, guard
  JIT compilation/caching so repeated calls do not recompile and never cache
  *results* across calls.

--- Original Design Draft Start ---

# KDA Phase-1 Draft — `glm52/moe_gate_proj_prefill`

**Target:** NVIDIA B200 (sm_100). **Goal of Phase 1:** produce the *first correct
independent* implementation of `solution.py:run()` that matches `reference.py`
within tolerance on all sweep shapes `M ∈ {1024, 2048, 4096}`. Beating the
baseline is desirable but **not** required to exit Phase 1. Calling the exact
baseline kernel (`deep_gemm.fp8_m_grouped_gemm_nt_masked`) as the only body of
`run()` is **not** an acceptable deliverable.

All empirical numbers below were measured in this worktree against the live
`reference.py` (see `kda/probe_sf.py`, an intentionally transient probe).

---

## 1. Task definition (authoritative)

Masked FP8 grouped GEMM, one GEMM per local expert:

```
out[e, :, :] = a[e] @ b[e].T          for e in [0, E)
  a[e] : [Mp, K]  (FP8 e4m3, per-token/1x128 UE8M0 scaled)
  b[e] : [N,  K]  (FP8 e4m3, per-block/128x128 UE8M0 scaled)
  out  : [E, Mp, N]  (bfloat16)
```

Constants: `E=8` (EP32-local of 256), `K=6144`, `N=2048`, `layout=1` (masked),
`n_global=256`, `topk=8`. `K` and `N` are already 128-aligned
(`Ka=6144`, `Na=2048`).

`run()` receives exactly (positional, from `definition.json:inputs`):
`a_fp8, a_s, b_fp8, b_s, out, masked_m, expected_m, m_indices, layout`.

Despite the name "prefill", **this task always drives the masked path**
(`layout==1`). `m_indices` is empty; the contiguous branch is never exercised.

### Measured input shapes / dtypes (M=1024)

| arg | shape | dtype | notes |
|-----|-------|-------|-------|
| `a_fp8` | `[8, Mp, 6144]` | `float8_e4m3fn` | contiguous |
| `a_s` | `[8, Mp, 12]` | `int32` | **packed UE8M0**, non-contiguous (MN-major, m-stride 1, pack-stride Mp). 12×4 = 48 = K/128 K-scales |
| `b_fp8` | `[8, 2048, 6144]` | `float8_e4m3fn` | contiguous |
| `b_s` | `[8, 2048, 12]` | `int32` | **packed UE8M0**, per-**row** (transform expanded 128×128 block scale to per-N-row), MN-major |
| `out` | `[8, Mp, 2048]` | `bfloat16` | pre-allocated (`torch.empty`), we write into it |
| `masked_m` | `[32]` | `int32` | valid rows per expert |
| `expected_m` | scalar | `int32` | DeepGEMM grid hint (mean load); **our kernel does not need it** |
| `m_indices`, `layout` | `[0]`, scalar | `int32` | unused / always 1 |

`Mp = align(max(masked_m), 128)`. Measured: **M=1024 → Mp=128**, **M=2048 → Mp=128**,
**M=4096 → Mp=256**. `masked_m` ranges (per shape): 1024→[17,40], 2048→[48,80],
4096→[97,158]; all 8 experts active (no empty experts on prefill).

---

## 2. Baseline behavior & how the harness judges us

### Harness mechanics (verified from `testbench/harness/*`)

- `driver.py` calls the reference's `get_inputs` **once**, then runs reference
  and candidate on **independent clones** of the same inputs (`clone_args`).
  Because both clone the *same* source `out`, any byte we/they leave untouched
  is identical between the two.
- **Correctness** (`correctness.py`): compares the **full** `out[E,Mp,N]`
  element-wise: `|x-y| ≤ atol + rtol·|y|`, aggregated to a matched-ratio.
  Gate = `matched_ratio ≥ 0.999` with `atol=0.1, rtol=0.05`. A hard **NaN/Inf
  sanity gate** fails if *either* tensor has any non-finite element.
- **Timing** (`timing.py`): CUPTI cold-L2 **device-kernel** time (span from first
  to last kernel of the iteration), fresh clone per iter, median of reps. Host
  Python is not directly timed, but every kernel it launches (including
  de-swizzle) lands inside the span — so **fewer launches / no host-device
  syncs** is the perf lever.
- **Reward-hack defenses** (`reward_hack.py` + driver post-timing recheck): the
  timed object is re-run and re-compared to the oracle afterwards; outputs must
  be exact `torch.Tensor` (no lazy/proxy); the timer identity is re-checked.
  ⇒ **We must not cache results across calls** and must genuinely compute every
  call. (KernelWiki `kernel-grouped-gemm` documents the NVFP4-hackathon
  cache-across-timing reward hack; the harness here defends against it.)

### Baseline latency (advisory `harness.profile`, warm-L2, current identity `solution.py`)

| M | median µs | useful TFLOPS | useful/padded util | roofline |
|---|-----------|---------------|--------------------|----------|
| 1024 | ~140 | 173 | 0.234 | 41.6% HBM |
| 2048 | ~137 | 370 | 0.492 | 42.4% HBM |
| 4096 | ~149 | 706 | 0.510 | 44.0% HBM |

Latency is ~flat across a 4× work increase ⇒ **launch/overhead- and
padding-bound**, not compute-bound. At M=1024 only 23% of the padded FLOPs are
useful (Mp=128 forces ≥128 rows/expert while `masked_m≈30`). Prior in-repo art
(`glm52--routed_down_decode`, same baseline kernel) won **1.24×** purely by
removing the two `.item()` device→host syncs the reference does per call
(`layout.item()`, `expected_m.item()`) — that trick is forbidden for our Phase-1
(it calls the baseline kernel), but it confirms the overhead-bound diagnosis and
tells us an independent kernel that avoids host syncs and extra launches has
real headroom.

### Write-pattern (critical for correctness — measured)

The masked kernel writes at **`block_m = 128` granularity**: for expert `e` it
computes rows `[0, ceil(masked_m[e]/128)·128)` (clamped to `Mp`); m-blocks that
contain no valid row are **skipped and retain the incoming `out` buffer**.
Verified by pre-filling `out` with a sentinel:

- M=1024, M=2048: `Mp=128` (one block) → **all 128 rows written**, values
  independent of the input buffer.
- M=4096: `Mp=256` → `rows_written ∈ {128, 256}` per expert, exactly
  `ceil(masked_m/128)·128`. Experts with `masked_m ≤ 128` leave rows `128..255`
  as the input buffer.

**Consequence for our candidate:** write into the provided `out` in-place and
overwrite **exactly** rows `[0, ceil(masked_m[e]/128)·128)` per expert; leave the
rest untouched. Skipped rows then match the reference byte-for-byte (same cloned
buffer); computed rows must match numerically. Computing *all* `Mp` rows
unconditionally would MISMATCH the reference's skipped blocks at M=4096
(≫0.1% of elements) and fail the ratio gate.

---

## 3. The independent-implementation enabler (validated)

The only real obstacle to an independent path is consuming DeepGEMM's *packed
UE8M0, MN-major, TMA-aligned* scale tensors (`a_s`, `b_s`, `int32`). We do **not**
need to call DeepGEMM's GEMM to use them:

**Scale recovery is exact.** Round-trip test (known plain UE8M0 scales →
`deep_gemm.transform_sf_into_required_layout(...)` → our inverse) is
`torch.equal` — **bit-exact**:

```python
from deep_gemm.utils.math import unpack_ue8m0_from_int
# a_s: [E, Mp, 12] int32 (packed) ; K//128 = 48 scales
a_scale = unpack_ue8m0_from_int(a_s.contiguous())[..., :K//128]   # [E, Mp, 48] fp32, pow2
b_scale = unpack_ue8m0_from_int(b_s.contiguous())[..., :K//128]   # [E, N,  48] fp32, pow2
```

`.contiguous()` is required (`unpack` needs last-dim stride 1). `b_s` is already
expanded to **per-row** `[E, N, 48]` by the transform, so A and B dequant are
symmetric. Recovered scales are exact powers of two (UE8M0), so recovery is
lossless — this is a light element-wise unpack of tiny tensors
(`E·Mp·12` + `E·N·12` int32), not "the baseline op".

**Dequant + fp32 matmul matches the reference.** With recovered scales,
`a_deq = a_fp8·a_scale`, `b_deq = b_fp8·b_scale`, `out = bmm(a_deq, b_deqᵀ)` gives
(over the rows the reference wrote):

| M | max_abs_err | matched_ratio @ (0.1, 0.05) |
|---|-------------|------------------------------|
| 1024 | 0.0156 | 1.000000 |
| 2048 | 0.0156 | 1.000000 |
| 4096 | 0.0156 | 1.000000 |

`max_abs 0.0156 ≪ atol 0.1`; the residual is bf16 output rounding, not a scheme
mismatch. This is because DeepGEMM's per-128-K-block scaling is block-diagonal in
K, so full dequant + fp32 matmul is mathematically identical up to fp32 ordering.
**The independent math is proven correct on every sweep shape.**

---

## 4. Ranked candidate directions

### C1 — Triton block-scaled FP8 grouped-masked GEMM (**primary deliverable**)

A single Triton kernel, grid `(E · n_blocks · m_blocks)`, that:
- recovers `a_scale`/`b_scale` once (host unpack, above) and passes fp32 (or int8
  exponent) scale tensors in;
- per tile loops K in 128-chunks: `acc += tl.dot(a_tile_fp8, b_tileᵀ_fp8) ·
  a_scale[m,kb] · b_scale[n,kb]` in fp32 (mirrors DeepGEMM's block-scaled accumulation);
- honors the write pattern: skip m-blocks with `m_block·128 ≥ ceil(masked_m[e]/128)·128`
  (read `masked_m` on-device — no host sync); write bf16 into `out`.
- `BLOCK_M=128` aligns naturally with the reference's block granularity;
  start `BLOCK_N=128, BLOCK_K=128` (DeepGEMM SM100 uses 128/128/64 —
  a later tuning axis).

Why primary: genuinely independent (no DeepGEMM GEMM call), a real B200 kernel we
can tune (tile sizes, num_stages/warps, persistent scheduling, later TMA/`tcgen05`
via CuTe), one launch, and Triton 3.6 supports fp8 `tl.dot` with fp32 accumulate
on sm_100. Prior art: SGLang `w8a8_block_fp8_matmul` / DeepSeek block-fp8 Triton;
KernelWiki `kernel-grouped-gemm`, `kernel-deepgemm`, `kernel-fused-moe`.

### C2 — De-swizzle + dequant + `torch.bmm` (fp32→bf16) (**correctness oracle / immediate fallback**)

The Section-3 path, wrapped as `run()`, writing only the masked block rows into
`out`. Guaranteed-correct (already validated), fully independent of DeepGEMM's
GEMM. Roles: (a) land a green Phase-1 pass immediately and de-risk the harness
contract end-to-end; (b) serve as the in-process oracle to debug C1; (c) prove
the write-pattern / scale-recovery code in isolation. Likely **slower** than
baseline (materializes ~1.6 GB fp32 `b_deq` + a cuBLAS bmm over full `Mp`), which
is acceptable for Phase 1. Keep the dequant in bf16 and restrict bmm to
`ceil(masked_m/128)·128` rows to limit waste.

### C3 — CUTLASS 4.5 / CuTe-DSL Blackwell grouped FP8 GEMM w/ blockscale (**stretch / Phase 2**)

`KernelPtrArrayTmaWarpSpecialized...Sm100` grouped GEMM with native UE8M0 block
scaling and `tcgen05.mma`/TMEM — highest ceiling, highest effort, and requires
converting DeepGEMM's SF layout to CUTLASS's SFA/SFB atom layout. Defer until C1
is correct and profiled; revisit if C1 leaves compute on the table.

**Recommended order:** C2 (lock correctness + validate scale/write code) → C1
(port the same math into a tunable single-launch kernel; this becomes the
shipped `solution.py`) → C3 only if warranted later.

---

## 5. Risks & unknowns

1. **`block_m` assumption (medium).** Correctness of the write-pattern depends on
   the reference writing at 128-row granularity. Verified for all three sweep
   shapes; the sweep is fixed. *Mitigation:* keep `BLOCK_M=128`; add a
   guard/self-check that recomputes `rows_written` vs `ceil(masked_m/128)·128`
   during development. If a shape ever disagreed, fall back to reading the
   reference's write mask.
2. **Host-device syncs (medium, perf).** Building a per-expert grid may tempt a
   `masked_m.tolist()` host sync. Prefer reading `masked_m`/counts on-device in
   the kernel; if a host count is unavoidable, do it once (not per expert), never
   `.item()` in a loop (the reference's own syncs are the overhead we're beating).
3. **FP8 numerics drift (low).** Validated at max_abs 0.0156 with fp32 accumulate.
   Triton `tl.dot` fp8 accumulation is fp32; keep per-128-K block scaling exactly
   as DeepGEMM to stay ≪ atol. Watch bf16 store rounding only.
4. **Reward-hack tripwires (low, but hard-fail).** No caching across calls; return
   a real `torch.Tensor`; never patch the timer. driver.py re-checks post-timing.
5. **Padded-row NaN/Inf (low).** The sanity gate fails on any non-finite. Our
   in-place write leaves skipped rows = the (finite) cloned buffer and the
   reference does the same, so both are finite; never write NaN into padding.
6. **Memory (low).** C2's fp32 `b_deq` ≈ 1.6 GB — fine on B200 (192 GB) but avoid
   in C1 (stream fp8 tiles instead).
7. **`layout==0` path (low).** Never exercised here; assert/branch defensively so
   an unexpected contiguous workload errors loudly rather than silently mis-computes.

---

## 6. First concrete steps

1. **Write `kda/candidates.jsonl` + `kda/benchmark.csv` seeds** with the measured
   baseline (Section 2) as the denominator reference.
2. **Implement C2** in `solution.py`: recover scales (Section 3), dequant to bf16,
   `bmm` per expert over `ceil(masked_m/128)·128` rows, write into `out`, return
   `out`. Handle `layout` defensively.
3. **Gate C2**: run the authoritative evaluate (`--max-workloads 1`, then full).
   Expect `correct=true` on all shapes.
4. **Implement C1** (Triton) using C2 as the in-process oracle; match block-128
   write pattern; single launch; on-device `masked_m`.
5. **Gate + profile C1**: authoritative evaluate for correctness; `harness.profile`
   + `ncu-report-skill` for latency vs the Section-2 baseline. Record NCU under
   `kda/profile/`.
6. Iterate tile config (`BLOCK_N/BLOCK_K`, `num_warps`, `num_stages`) only after
   C1 is correct.

---

## 7. Exact validation commands

```bash
cd /home/qinhaiyan/KDA-Exp/worktrees/glm52-moe_gate_proj_prefill
PY=/home/qinhaiyan/Kernel-Harness/.venv/bin/python

# Fast advisory smoke (per shape)
PYTHONPATH=testbench $PY -m harness.profile \
  testbench/tasks/glm52/moe_gate_proj_prefill --shape 1024   # also 2048, 4096

# Authoritative gate (CUPTI cold-L2 + correctness). Exit: 0=WIN, 1=correct-not-faster, 2=incorrect
$PY testbench/bin/evaluate.py testbench/tasks/glm52/moe_gate_proj_prefill --max-workloads 1
$PY testbench/bin/evaluate.py testbench/tasks/glm52/moe_gate_proj_prefill
```

Edit only `testbench/tasks/glm52/moe_gate_proj_prefill/solution.py`. Never edit
`reference.py`.

---

## 8. Evidence rules — promote / revise / reject

- **Promote** a candidate when the authoritative `evaluate.py` reports
  `correct=true` on **all** shapes `{1024,2048,4096}` (matched_ratio ≥ 0.999,
  no NaN/Inf), reproduced across ≥2 runs. This alone satisfies the Phase-1 exit
  criterion. Record in `kda/candidates.jsonl` + `kda/benchmark.csv`.
- **Prefer/keep** C1 over C2 once C1 is correct — C1 is the optimizable, single-
  launch, independent kernel intended as the deliverable; C2 remains only as the
  oracle. (A pass that is *only* a re-export of the baseline kernel is rejected by
  the Phase-1 contract regardless of speed.)
- **Revise** (don't discard) on: correctness failure isolated to padded/skipped
  rows (→ fix write-pattern rounding), a single-shape tolerance miss (→ inspect
  fp8 accumulation / bf16 store), or a latency regression vs baseline (Phase-1
  still passes; log for Phase 2).
- **Reject** on: non-finite outputs, reward-hack detection (`REWARD_HACK`),
  `RUNTIME_ERROR`, or reliance on calling `deep_gemm.fp8_m_grouped_gemm_nt_masked`
  as the computational body.
- **Performance (advisory in Phase 1):** an authoritative WIN needs
  `solution_us < baseline_us` on **every** shape (cold-L2). Track geomean and
  min-conservative speedup vs the Section-2 baseline; do not block Phase-1 exit
  on it.

--- Original Design Draft End ---
