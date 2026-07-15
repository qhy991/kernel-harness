# GLM-5.2 `o_proj_prefill` — First Correct Independent B200 FP8 Block-Scaled GEMM

> **Retarget note (2026-07-15):** Harness shapes updated to **DP=1 / TP=1 / EP=32** (commit `88d7044`). Numeric axes below were patched in-place; re-benchmark after this change.


## Goal Description

Produce the first **correct, independent** B200 (sm_100 / Blackwell) implementation of
the GLM-5.2 attention O-projection for `testbench/tasks/glm52/o_proj_prefill/solution.py`.

The op is an FP8 blockwise-scaled GEMM:
`out[M, 6144] = x_fp8[M, 16384] @ w_fp8[6144, 16384].T`, bf16 output, over the prefill
sweep `M ∈ {1024, 2048, 4096}` (K=16384=64·256 local heads·v_head, N=6144=hidden).

Inputs are pre-quantized (offline, untimed) exactly as the SGLang/DeepGEMM production path
produces them: `float8_e4m3fn` data plus **UE8M0-packed int32 block scales** (1×128
activation groups, 128×128 weight blocks), stored column-major / TMA-aligned. `run()`
must consume those tensors and reproduce the reference's block-scaled accumulation.

Phase-1 success = authoritative `evaluate.py` reports `correct=true` on **all three
shapes** (matched-ratio tolerance) with a **genuinely independent** kernel body. Beating
the DeepGEMM baseline latency is desirable but **not required** to exit Phase 1; the
chosen design must nonetheless be optimizable (not a thin re-export of the baseline op).

Authoritative edit target: `testbench/tasks/glm52/o_proj_prefill/solution.py` (only its
`run()` function is used). `reference.py` must never be edited.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for
deterministic verification. Correctness tolerances (`max_atol=0.1`, `max_rtol=0.05`,
`required_matched_ratio=0.999`) are **hard requirements fixed by the harness/task**
(`task.json` / `workload.jsonl`), not tunable preferences. Latency-beats-baseline is an
**optimization trend** (desirable, non-gating for Phase 1).

- AC-1: Correctness on the full sweep. Under the authoritative gate, `solution.py`'s
  `run()` output matches `reference.py` on every shape `M ∈ {1024, 2048, 4096}`:
  elementwise `|cand − ref| ≤ 0.1 + 0.05·|ref|` (both cast to fp32) for ≥ 99.9% of
  elements, no NaN/Inf.
  - Positive Tests (expected to PASS):
    - `PY testbench/bin/evaluate.py testbench/tasks/glm52/o_proj_prefill --max-workloads 1` reports the M=1024 workload correct.
    - `PY testbench/bin/evaluate.py testbench/tasks/glm52/o_proj_prefill` (full sweep) reports `correct=true` and exits `0` or `1`.
    - `PYTHONPATH=testbench PY -m harness.profile testbench/tasks/glm52/o_proj_prefill --shape 2048` shows a matched result (advisory smoke).
  - Negative Tests (expected to FAIL):
    - A kernel that scales only once per full K (instead of per 128-group) drops below matched_ratio 0.999 → INCORRECT (exit 2).
    - A kernel that reads the wrong UE8M0 byte for a K-group produces systematic error → INCORRECT.
    - Any NaN/Inf in the output → INCORRECT.

- AC-2: Independence of the kernel body. The committed `run()` computes the GEMM with an
  author-written kernel, not the baseline op or a thin wrapper of it.
  - Positive Tests (expected to PASS):
    - `run()` contains an author-written GEMM (e.g. a `@triton.jit` block-scaled kernel) as the compute body.
    - Source review confirms no import/call of `w8a8_block_fp8_matmul_deepgemm` and no bare `deep_gemm.fp8_gemm_nt` / `deep_gemm` grouped-GEMM entrypoint as the compute path.
  - Negative Tests (expected to FAIL):
    - `run()` whose body is `return w8a8_block_fp8_matmul_deepgemm(...)` → rejected (thin re-export).
    - `run()` whose body is `return deep_gemm.fp8_gemm_nt(...)` → rejected (thin re-export).
    - Delegating the multiply to any SGLang/DeepGEMM production GEMM helper → rejected.

- AC-3: Interface & harness-contract compliance.
  - AC-3.1: Signature is `def run(x_fp8, x_scale, w_fp8, w_scale)` (names + order matching `definition.json["inputs"]`).
    - Positive: harness calls `run()` positionally and (in the alias probe) by keyword using those exact names, both succeed.
    - Negative: renamed/reordered params that break the keyword alias probe → rejected.
  - AC-3.2: Output is a **fresh, contiguous** `torch.bfloat16` tensor of shape `[M, 6144]`, with `M = x_fp8.shape[0]` (never the raw workload M).
    - Positive: returned tensor is `bfloat16`, `[x_fp8.shape[0], 6144]`, `is_contiguous()`, allocated fresh (not aliasing any input).
    - Negative: returning a view/alias of an input, wrong dtype-that-blows-tolerance, or wrong shape → rejected.
  - AC-3.3: `run()` is pure/stateless — no input mutation, no cross-call caching, correct under per-iteration `clone_args`.
    - Positive: repeated calls on freshly cloned inputs yield identical results; post-timing recheck passes.
    - Negative: output cached by shape/data-pointer, or reliance on input object identity → reward-hack rejection (exit 2).

- AC-4: Stride-safe UE8M0 scale handling. Scale decode/indexing works on the actual
  runtime tensor layout (column-major / F-contiguous, possibly re-laid-out by `clone`),
  not on an assumed contiguous layout.
  - Positive Tests (expected to PASS):
    - Decoded activation multiplier for group `g`, row `m` equals `2^((x_scale[m, g//4] >> (8·(g%4)) & 0xFF) − 127)` and matches an independent reference decode.
    - Weight multiplier uses `w_scale[n, g//4]` directly (full-N layout) and equals the value for `w_scale[(n//128)*128, g//4]` (repetition verified once).
  - Negative Tests (expected to FAIL):
    - Decoding via `int32.view(torch.uint8)` on an F-contiguous scale tensor (byte order/stride wrong) → mismatch.
    - Indexing weight scale as `[n//128, ...]` against the actual `[N, ...]` layout → mismatch.

- AC-5: Numerically-faithful block-scaled accumulation. The K reduction is split into
  128-wide groups; each group's FP8 partial product is scaled by `s_a·s_b` and summed in
  an fp32 accumulator; bf16 conversion happens only at store.
  - Positive Tests (expected to PASS):
    - With `BLOCK_K = 128`, accumulator dtype fp32, per-group scaling → matches reference within tolerance on all shapes.
    - A standalone dequant→`torch.matmul` oracle (debug only) agrees with the kernel's per-group contributions.
  - Negative Tests (expected to FAIL):
    - Accumulating several K-groups before applying a single scale → error when group scales differ.
    - bf16 (not fp32) accumulation across all of K → tolerance/matched_ratio failure risk.

- AC-6: No timed-path recompilation / warmup-safe. Kernels are defined at module scope and
  specialized via constexprs so first-call JIT compile does not land inside the measured
  region and does not present as instability.
  - Positive Tests (expected to PASS):
    - `@triton.jit` kernel(s) defined at import time; harness warmup (10 iters) absorbs any compile; timed iterations are compile-free.
    - Post-timing recheck (fresh clone) still passes correctness.
  - Negative Tests (expected to FAIL):
    - Defining/compiling a kernel inside `run()` per call such that timed iterations recompile → instability / degraded measurement.

- AC-7: Campaign bookkeeping recorded (per the Phase-1 task contract). Candidate entries in
  `kda/candidates.jsonl`, runs in `kda/benchmark.csv`, any NCU artifacts under
  `kda/profile/`.
  - Positive Tests (expected to PASS):
    - After an evaluation, a candidate row exists in `kda/candidates.jsonl` and a run row in `kda/benchmark.csv`.
  - Negative Tests (expected to FAIL):
    - Committing a passing candidate with no corresponding bookkeeping entry → incomplete deliverable.

- AC-8 (non-gating, trend): Performance measured vs baseline. Latency captured on all
  shapes; a WIN (exit 0) requires `solution_us < baseline_us` on every shape, but exit 1
  (correct, slower) still satisfies Phase 1.
  - Positive Tests (expected to PASS):
    - Per-shape solution/baseline latency recorded in `kda/benchmark.csv`.
    - Exit code `0` (faster everywhere) OR `1` (correct, not faster) — both acceptable for Phase-1 exit.
  - Negative Tests (expected to FAIL):
    - Treating a slower-but-correct result as a Phase-1 failure → incorrect gate interpretation.
    - Claiming a WIN without per-shape `solution_us < baseline_us` evidence → unsupported.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
A single fused author-written Triton block-scaled FP8 GEMM in `solution.py` that unpacks
the UE8M0 int32 scales in-kernel (stride-aware), accumulates 128-wide K-groups in fp32
with per-group `s_a·s_b` scaling, stores bf16, is autotuned over a small tile-config set,
and is correct on all three shapes — with candidate/benchmark bookkeeping and an optional
NCU profile recorded. Reaching or approaching DeepGEMM latency is welcome here but is not
required and must not compromise correctness or independence.

### Lower Bound (Minimum Acceptable Scope)
A two-stage but still independent implementation: a small author-written prologue decodes
the UE8M0 int32 scales to compact fp32 scale tensors (stride-safe), followed by an
author-written Triton block-scaled GEMM (`BLOCK_K=128`, fp32 accumulate, per-group
scaling, bf16 store) that is correct on all three shapes. Slower than the baseline is
acceptable (exit 1). Bookkeeping entries recorded.

### Allowed Choices
- Can use: author-written Triton (`tl.dot` on `float8e4nv`, or `tl.dot_scaled`) kernels;
  hand-written CUDA C++ / CuTe-DSL / CUTLASS `tcgen05` block-scaled GEMM; an fp32
  in-kernel accumulator; a small torch/Triton scale-decode prologue; autotuning; edge
  masking or deliberate specialization to the aligned sweep shapes; reading DeepGEMM/
  SGLang source for **understanding** the layout.
- Cannot use: `w8a8_block_fp8_matmul_deepgemm` or a bare `deep_gemm.fp8_gemm_nt` /
  DeepGEMM grouped-GEMM entrypoint as the compute body; any SGLang/DeepGEMM production
  GEMM helper as the multiply; editing `reference.py`, `workload.jsonl`, harness/evaluator
  code; copying contest release kernels into this workspace; hardcoded machine paths in
  committed `solution.py`; persistent state or workload-specific output caching.

> **Note on Deterministic Designs**: The *math* is fully determined (block-scaled FP8
> GEMM with the exact UE8M0 decode and per-128-group accumulation described above) — this
> is fixed and not a choice. The *kernel realization* (Triton vs CuTe/CUTLASS, fused vs
> two-stage, tile sizes) is an open engineering choice within the bounds above.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual
> suggestions, not prescriptive requirements.

### Conceptual Approach
Reproduce, in fp32:
```
out[m,n] = Σ_{g=0..15}  s_a[m,g] · s_b[n,g] · ( Σ_{k∈group g} A_fp8[m,k] · B_fp8[n,k] )
group g  = K-slice [128g, 128g+128)
s_a[m,g] = 2^( ((x_scale[m, g//4] >> (8·(g%4))) & 0xFF) − 127 )   # UE8M0, little-endian
s_b[n,g] = 2^( ((w_scale[n, g//4] >> (8·(g%4))) & 0xFF) − 127 )   # full-N layout, per-row
```
Triton sketch (module-level kernel, `BLOCK_K = 128`):
```
acc = zeros([BLOCK_M, BLOCK_N], fp32)
for g in range(K // 128):                      # 16 groups
    a = load A_fp8 tile [BLOCK_M, 128]          # float8e4nv (verify E4M3 semantics)
    b = load B_fp8 tile [BLOCK_N, 128]
    p = tl.dot(a, b.T)                          # fp32 partial (or upcast→bf16 fallback)
    sa = decode(x_scale group g)               # [BLOCK_M]
    sb = decode(w_scale group g)               # [BLOCK_N]
    acc += p * sa[:, None] * sb[None, :]
store acc.to(bf16)
```
Bring-up ladder: (0) scratch dequant→`torch.matmul` oracle to validate the UE8M0 decode
and per-group math (debug only, NOT the deliverable, and not proof of final correctness
vs DeepGEMM); (1) two-stage Triton (decode prologue + block-scaled GEMM), validate M=1024
then full sweep; (2) fuse the decode in-kernel (the deliverable); (3) optional autotune /
measure / NCU. Prototype a `tl.dot_scaled` variant early to compare against manual
dot+scale — but keep manual dot+scale as the robust correctness primary.

### Relevant References
- `testbench/tasks/glm52/o_proj_prefill/reference.py` — oracle + baseline (do not edit); shows the exact `get_inputs` quant/requant recipe.
- `testbench/harness/{driver,correctness,timing,inputs,reward_hack}.py` — how `run()` is loaded (source exec, only `run` used), compared (matched-ratio, fp32), timed (CUPTI cold-L2, `clone_args` each iter), and guarded (lazy/alias/monkey-patch).
- SGLang `sglang/srt/layers/quantization/fp8_kernel.py` — classic Triton `w8a8_block_fp8_matmul` scale-indexing pattern (fp32 per-block scales) to mirror; `sglang_per_token_group_quant_fp8` layout.
- DeepGEMM `deep_gemm/utils/{math,layout}.py` — `pack_ue8m0_to_int` / `unpack_ue8m0_from_int` confirm byte order (study only; author the unpack in the committed kernel).
- KernelWiki `kernel-fp8-block-scale-gemm`, `kernel-deepgemm` — SM100 `tcgen05.mma...block_scale` design (Phase-2 perf target); UE8M0 = 4 scales/int32; fp32-accumulate, scales at MMA boundary.
- Prior art PRs: CUTLASS PR-2139 (Blackwell blockwise/groupwise GEMM), vLLM PR-14383 (Blackwell FP8 blockwise), vLLM PR-38083 (E8M0 accuracy context — not a divergence source here since inputs/scales are shared).

## Dependencies and Sequence

### Milestones
1. Layout & math validation (no committed kernel):
   - Phase A: Confirm UE8M0 decode + per-128-group accumulation reproduce `reference.run` within tolerance in a scratch dequant→`torch.matmul` oracle (M=1024). Verify weight-scale per-row repetition and Triton `float8e4nv` ↔ torch `float8_e4m3fn` semantics.
2. Independent GEMM bring-up (two-stage → fused):
   - Step 1: Two-stage Triton (author-written decode prologue + author-written block-scaled GEMM); pass M=1024 via `--max-workloads 1`, then full sweep.
   - Step 2: Fuse UE8M0 unpack into the kernel; re-validate full sweep. This is the committed deliverable.
3. Hardening & measurement (non-gating):
   - Step 1: Confirm stateless/no-alias/no-recompile guards; edge-mask or document shape specialization.
   - Step 2: Record latency vs baseline in `kda/benchmark.csv`; if a bottleneck is unclear, profile with `ncu-report-skill` (artifacts under `kda/profile/`); optionally prototype `tl.dot_scaled` / plan a CuTe/CUTLASS successor for Phase 2–3.

Dependencies: Milestone 2 depends on the decode/math confirmation in Milestone 1;
Milestone 3 depends on a correct kernel from Milestone 2. Correctness (AC-1..AC-6) gates
Phase 1; performance (AC-8) is layered on afterward and never blocks the Phase-1 exit.

## Task Breakdown

Each task includes exactly one routing tag (`coding` = implemented by Claude;
`analyze` = executed via Codex `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|--------------------------|------------|
| task1 | Scratch oracle: decode UE8M0 → fp32, dequant, `torch.matmul`, diff vs `reference.run` (M=1024); verify weight-scale per-row repetition and `float8e4nv`↔`float8_e4m3fn` semantics | AC-4, AC-5 | coding | - |
| task2 | Implement two-stage independent Triton path in `solution.py`: author-written stride-safe scale-decode prologue + author-written block-scaled GEMM (`BLOCK_K=128`, fp32 accumulate, bf16 store) | AC-1, AC-2, AC-3, AC-5 | coding | task1 |
| task3 | Fuse UE8M0 unpack into the kernel (drop prologue); module-level `@triton.jit`; re-validate full sweep | AC-1, AC-2, AC-6 | coding | task2 |
| task4 | Harden: assert fresh contiguous bf16 output, no input mutation/alias, no per-call recompile; edge-mask or document shape specialization | AC-3, AC-6 | coding | task3 |
| task5 | Record candidate/run bookkeeping (`kda/candidates.jsonl`, `kda/benchmark.csv`); capture per-shape latency vs baseline | AC-7, AC-8 | coding | task3 |
| task6 | Independent numeric-divergence review: does manual dot+scale plausibly hold matched_ratio ≥ 0.999 over 25M elements at M=4096 vs tcgen05 accumulation, or is `tl.dot_scaled` needed? | AC-1, AC-5 | analyze | task2 |
| task7 | Optional NCU profile + Phase-2 successor sketch (`tl.dot_scaled` / CuTe / CUTLASS `tcgen05` block-scale) if pursuing baseline-competitive latency | AC-8 | analyze | task5 |

## Claude-Codex Deliberation

### Agreements
- The independent boundary is strict: no `w8a8_block_fp8_matmul_deepgemm`, no bare
  `deep_gemm.fp8_gemm_nt`, no SGLang/DeepGEMM production GEMM helper as the multiply.
- Correctness math must use `BLOCK_K = 128`, an fp32 accumulator, and per-128-group
  `s_a·s_b` scaling before accumulation; bf16 only at store.
- Scale decode must be stride-safe (int32 bit-shift/mask), not `int32.view(uint8)` on an
  F-contiguous tensor; derive `M` from `x_fp8.shape[0]`; return a fresh contiguous bf16
  tensor; keep `run()` pure and kernels module-level (no timed-path recompile).
- Verify Triton `float8e4nv` ↔ torch `float8_e4m3fn` semantics rather than assuming; keep
  an in-kernel bf16-upcast fallback if `tl.dot` cannot consume FP8 on this build.
- Use the dequant→`torch.matmul` oracle only for debugging scale indexing, not as proof
  of final correctness against DeepGEMM.

### Resolved Disagreements
- Repo-convention conflicts (Codex flagged hardcoded `.venv` path, `kda/*` artifacts,
  `integration_status`/`check_env`/`knowledge.py`, `<model>/<task>` command form): Codex's
  sandbox blocked reading the repo guide, so it inferred a generic Kernel-Harness
  contract. For **this worktree** the authoritative contract is `kda/prompts/phase1.md` +
  `kda/AGENT_PROMPT.md`, which **explicitly mandate** the `kda/candidates.jsonl` /
  `kda/benchmark.csv` / `kda/profile/` bookkeeping and the exact
  `PY=/home/qinhaiyan/Kernel-Harness/.venv/bin/python … testbench/bin/evaluate.py <task-dir>`
  commands, and define Phase-1 success as `evaluate.py correct=true` (exit ≤ 1).
  Resolution: keep those commands/paths in the plan's validation section (they are the
  contract), and forbid hardcoded machine paths only inside committed `solution.py`.
  Generic hygiene steps (`check_env.py`, `integration_status.py`) are folded into an
  optional pre-flight, not a gate (see DEC-1).
- `tl.dot_scaled` as stretch vs primary: Codex argued it may be the only path that matches
  hardware block-scaling. Resolution: keep manual dot+scale (fully controllable, clearly
  independent) as the correctness primary, but prototype `tl.dot_scaled` early (task6) and
  adopt it if manual dot+scale cannot hold matched_ratio ≥ 0.999.
- Two-stage vs fused deliverable: Codex asked whether a decode prologue is allowed.
  Resolution: allowed (the contract only forbids the exact baseline op as the whole body);
  two-stage is the lower bound, fused is the upper bound — both independent.

### Convergence Status
- Final Status: `partially_converged` (generated in `--direct` mode; the Phase-5
  Claude↔Codex convergence loop is intentionally skipped, so human review is expected
  before the RLCR loop). One Codex first-pass analysis was incorporated (0 convergence
  rounds by design).

## Pending User Decisions

- DEC-1: Run generic Kernel-Harness pre-flight (`check_env.py`, `integration_status.py glm52/o_proj_prefill`) and confirm the drop-in vs fused-only integration contract before implementation?
  - Claude Position: Not required for the Phase-1 correctness gate, which `phase1.md`
    defines solely via `evaluate.py`. Treat as optional hygiene (an `analyze`/pre-flight
    step), so it does not block correctness work.
  - Codex Position: Recommends running these before editing to confirm the contract and
    avoid inferring drop-in status.
  - Tradeoff Summary: Running them costs little and de-risks integration assumptions;
    skipping them is safe for a pure Phase-1 correctness deliverable since the gate is
    self-contained. Low impact either way.
  - Decision Status: `PENDING`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as
  "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead (e.g. `K_GROUP = 128`,
  `decode_ue8m0_scale`, `block_scaled_fp8_gemm`).
- No hardcoded machine paths, evaluator/timing assumptions, persistent state, or
  workload-specific output caching in committed `solution.py`.

--- Original Design Draft Start ---

# Implementation Plan Draft — `glm52/o_proj_prefill`

**Task:** GLM-5.2 Attention O-Projection, prefill, TP1 (full 64 heads). FP8 blockwise GEMM.
**Op:** `out[M, 6144] = x_fp8[M, 16384] @ w_fp8[6144, 16384].T` (bf16 output).
**Target:** NVIDIA B200 (sm_100 / Blackwell). Phase 1 = first **correct independent**
implementation; performance measured but not gating.

---

## 1. Baseline behavior + validation

### 1.1 What the baseline does
`reference.py` is both the correctness oracle and the latency denominator. It calls
SGLang's production path `w8a8_block_fp8_matmul_deepgemm(x_fp8, w_fp8, x_scale,
w_scale, [128,128], bf16)`, which dispatches to `deep_gemm.fp8_gemm_nt` — a Blackwell
`tcgen05.mma` block-scaled FP8 GEMM with TMEM accumulation. Weights are pre-quantized
offline in `get_inputs` (untimed); `run()` executes only the GEMM.

Numerically it computes, in fp32 with per-128-K-group rescaling:
```
out[m,n] = Σ_{g=0..15}  s_a[m,g] · s_b[n,g] · ( Σ_{k∈group g} A_fp8[m,k] · B_fp8[n,k] )
```
where `group g` is the 128-wide K-slice `[128g, 128g+128)`, and `s_a`, `s_b` are the
per-group power-of-two dequant multipliers.

### 1.2 Exact input contract (measured on-device, all shapes)
`run()` is called **positionally** in `definition.json["inputs"]` order. Parameter
names should match the keys (harness alias-probe safety), so:
`def run(x_fp8, x_scale, w_fp8, w_scale)`.

| Tensor | Shape | Dtype | Layout / strides | Notes |
|---|---|---|---|---|
| `x_fp8` | `[M, 2048]` | `float8_e4m3fn` | row-major contiguous `(2048,1)` | activations |
| `x_scale` | `[M, 4]` | `int32` | **column-major** `stride=(1, align(M,4))` (TMA-aligned) | ue8m0-packed activation scales |
| `w_fp8` | `[6144, 2048]` | `float8_e4m3fn` | row-major contiguous `(2048,1)` | weights |
| `w_scale` | `[6144, 4]` | `int32` | column-major `stride=(1, 6144)` (TMA-aligned) | ue8m0-packed weight scales, **repeated per row within each 128-row N-block** |
| **output** | `[M, 6144]` | `bfloat16` | fresh contiguous | `M = x_fp8.shape[0]` (already TMA-aligned; ∈{1024,2048,4096}) |

`packed_K = ceil_div(ceil_div(K,128), 4) = ceil_div(16,4) = 4` int32 columns; each int32
packs **4 consecutive K-groups**.

### 1.3 UE8M0 scale decode (confirmed by probe + DeepGEMM source)
Each int32 holds 4 little-endian e8m0 exponent bytes. For output K-group `g` (0..15):
```
byte  = (scale_int32[row, g // 4] >> (8 * (g % 4))) & 0xFF     # e in [1,254]
mult  = 2.0 ** (byte - 127)                                     # pure power-of-2 dequant factor
```
- Activation: `s_a[m,g]` from `x_scale[m, g//4]`.
- Weight: `s_b[n,g]` from `w_scale[n, g//4]` (already expanded to full N — no `n//128` needed).
- Divisor used at quant time is `448.0` (fp8 e4m3 max); the stored value **is** the
  multiply-back factor, so there is no inverse to apply. Probe check: byte `120 →
  2^-7 = 0.0078125`, byte `115 → 2^-12`.

**Critical correctness insight:** both the baseline and our kernel consume the *same*
pre-quantized `x_fp8`/`w_fp8` and the *same* ue8m0 scales. E8M0 "power-of-2 ceiling"
rounding (a known Blackwell accuracy topic, e.g. vLLM PR-38083) is already baked into
the shared inputs, so it is **not** a divergence source between us. We only need to
reproduce the group-scaled fp32 accumulation — tolerance is very achievable.

### 1.4 Harness gate (how correctness/latency are judged)
- Inputs always come from **reference's** `get_inputs`; `solution.py` needs only `run()`.
- Per-iteration `clone_args` (plain `.clone()` each timing iter) → `run()` must be
  **stateless**, must not mutate/return inputs, and must not rely on over-aligned
  padding. Our M is TMA-aligned so `x_scale`/`w_scale` survive clone as F-contiguous;
  we still treat scale strides as opaque (decode with stride-agnostic ops).
- Correctness = matched-ratio: `|cand−ref| ≤ atol + rtol·|ref|` elementwise (both cast
  to fp32), fraction passing must be `≥ 0.999`. Tolerance: `atol=0.1, rtol=0.05`. Any
  nan/inf → INCORRECT.
- Exit codes: `0`=WIN (correct **and** faster on every shape), `1`=correct but not
  faster, `2`=incorrect/incomplete. **Phase-1 success = exit ≤ 1 (correct on all shapes).**

---

## 2. Ranked candidate directions

### #1 (PRIMARY) — Author-written Triton block-scaled FP8 GEMM
A hand-tiled GEMM: `tl.dot` on `float8e4nv` (e4m3) tiles accumulating in fp32, with
per-128-K-group rescaling by `s_a·s_b`. This is genuinely independent (our own tiling,
not the baseline op) and directly optimizable. Confirmed feasible: Triton **3.6.0** on
**sm_100 (cc 10.0)**, `tl.float8e4nv` and `tl.dot_scaled` both present.

- **Scale handling — two sub-variants:**
  - **1a (bring-up):** decode ue8m0 → fp32 `s_a[M,16]`, `s_b[N,16]` in `run()` with a
    couple of stride-safe torch ops (`view(uint8)` → `ldexp`), then a standard
    block-scaled Triton GEMM indexing `s_a[m, k//128]`, `s_b[n, k//128]`. Mirrors the
    proven scale-indexing pattern of SGLang's classic Triton `w8a8_block_fp8_matmul`
    (which uses fp32 per-block scales). Robust; one extra tiny launch.
  - **1b (fused deliverable):** unpack ue8m0 int32 inside the kernel (shift/mask/`exp2`),
    removing the prologue op.
- **Stretch:** `tl.dot_scaled` consuming the packed scales natively (MX-style), closer
  to the hardware block-scale path.
- Pros: fastest route to correctness; readable; a real optimization surface (tiling,
  autotune, `num_warps`, pipelining). Cons: unlikely to beat the tcgen05 baseline at
  large M (fine for Phase 1).

### #2 (LATER / Phase 2-3) — CuTe-DSL / CUTLASS `tcgen05` block-scaled FP8 GEMM
The Blackwell-native perf ceiling: `tcgen05.mma.kind::mxf8f6f4.block_scale` with UE8M0
scales applied inside the MMA and TMEM fp32 accumulation — exactly what DeepGEMM does.
Prior art: CUTLASS PR-2139 (Blackwell blockwise/groupwise GEMM), vLLM PR-14383 (Blackwell
FP8 blockwise), CUTLASS SM100 schedules. **Out of Phase-1 scope** per the prompt (Phase 1
prioritizes a clean correct design), recorded as the intended optimization end-state.

### #3 (SCAFFOLD ONLY — not a deliverable) — direct `deep_gemm.fp8_gemm_nt`
The low-level primitive under the baseline wrapper. Calling it as the body of `run()` is
a **thin re-export of the baseline op and is NOT an acceptable Phase-1 deliverable**
(prompt §"Phase 1 Goal"). Use it **only** as an independent correctness oracle / latency
sanity check during bring-up, never committed as the solution.

---

## 3. First concrete steps (bring-up ladder)

Each step is validated before proceeding; artifacts recorded in `kda/candidates.jsonl`.

- **Step A — scale-decode oracle (scratch, not committed):** in a scratch script, decode
  ue8m0 → fp32, dequant `A`,`B` to bf16 with per-group scales, `torch.matmul`, and compare
  to `reference.run` under the task tolerance on M=1024. Proves the decode + math model in
  isolation (no kernel yet). Also cross-check against a one-off `deep_gemm.fp8_gemm_nt` call.
- **Step B — Triton variant 1a:** implement `run()` = {decode scales to fp32 in torch} +
  {author's block-scaled Triton GEMM}. Return fresh contiguous bf16 `[M,6144]`.
  Validate M=1024 first, then the full sweep.
- **Step C — fuse (variant 1b):** move ue8m0 unpack into the kernel; drop the prologue.
  Re-validate full sweep. **This is the Phase-1 deliverable in `solution.py`.**
- **Step D — measure/tune (optional, non-gating):** autotune `BLOCK_M/N/K`, `num_warps`,
  `num_stages`; record latency vs baseline in `kda/benchmark.csv`; if a bottleneck is
  unclear, profile with `ncu-report-skill`, artifacts under `kda/profile/`.

Implementation notes carried into the plan:
- Derive `M` from `x_fp8.shape[0]` (never the raw workload M).
- Treat `x_scale`/`w_scale` strides as opaque — decode via the tensor's own layout; do
  not assume C-contiguity (they are column-major / F-contiguous).
- If `tl.dot` rejects fp8 e4m3 on this build, upcast tiles to bf16 in-kernel (correct,
  slightly slower) as a fallback.
- Keep `run()` pure/stateless (post-timing recheck + reward-hack guards will catch state).

---

## 4. Exact validation commands

Run from this worktree root; use the shared Kernel-Harness venv.
```bash
cd /home/qinhaiyan/KDA-Exp/worktrees/glm52-o_proj_prefill
PY=/home/qinhaiyan/Kernel-Harness/.venv/bin/python

# Fast advisory smoke (single shape)
PYTHONPATH=testbench $PY -m harness.profile \
  testbench/tasks/glm52/o_proj_prefill --shape 1024

# Authoritative gate — quick (one workload) then full sweep
$PY testbench/bin/evaluate.py testbench/tasks/glm52/o_proj_prefill --max-workloads 1
$PY testbench/bin/evaluate.py testbench/tasks/glm52/o_proj_prefill
```
Exit codes: `0`=WIN, `1`=correct-but-not-faster, `2`=incorrect. Sweep shapes
`M ∈ {1024, 2048, 4096}`. Edit only `testbench/tasks/glm52/o_proj_prefill/solution.py`;
never edit `reference.py`.

---

## 5. Evidence rules — promote / revise / reject

**Promote** a candidate (accept as Phase-1 deliverable) when:
- Authoritative `evaluate.py` (full sweep) reports **`correct=true` on all three shapes**
  (matched_ratio ≥ 0.999, no nan/inf), and
- `run()` is genuinely independent (own GEMM body; not the baseline op / not
  `deep_gemm.fp8_gemm_nt` as the sole body), and
- The design has a clear optimization path (tiling/autotune or a CUTLASS/CuTe successor).

**Revise** (iterate, don't discard the direction) when:
- Correct on some shapes but not others, or matched_ratio just under 0.999 → almost
  certainly a scale byte-order / group-index / M-derivation bug. Debug against the Step-A
  oracle by diffing per-group contributions; inspect `s_a`/`s_b` decode; verify tile-K vs
  group-K (128) alignment (K-group pointer advances every `128/BLOCK_K` inner iters).
- Correct but slower than baseline → still a valid Phase-1 exit; log latency and defer to
  Step D / Phase 2 (CUTLASS/CuTe) for the speed work.

**Reject** a direction when:
- It cannot reach all-shape correctness within bounded effort, **or**
- It requires editing `reference.py`, copying contest release kernels, or using the exact
  baseline op (`w8a8_block_fp8_matmul_deepgemm`) / a bare `deep_gemm.fp8_gemm_nt` as the
  `run()` body.

**Bookkeeping (per task contract):** candidates → `kda/candidates.jsonl`, runs →
`kda/benchmark.csv`, NCU reports → `kda/profile/`.

---

## 6. Risks & unknowns

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| R1 | UE8M0 byte-order / group mapping wrong | Low (confirmed by probe + `pack_ue8m0_to_int` source) | Step-A oracle validates decode before any kernel |
| R2 | Column-major / F-contig scale strides mishandled after `clone_args` | Med | Decode scales with stride-agnostic torch ops; derive M from `x_fp8.shape[0]`; never assume C-contig |
| R3 | Triton fp8 `tl.dot` unsupported on this build | Low (Triton 3.6.0, sm_100, `float8e4nv`/`dot_scaled` present) | In-kernel bf16 upcast fallback |
| R4 | Numeric mismatch (accumulation order) | Low | fp32 per-group accumulate; atol=0.1 generous; matched_ratio is a sharp signal for systematic errors |
| R5 | Slower than tcgen05 baseline at large M | High | Acceptable (exit 1 passes Phase 1); perf via CUTLASS/CuTe in Phase 2-3 |
| R6 | Reward-hack / statefulness guards trip | Low | Keep `run()` pure; return fresh contiguous bf16; no input mutation/return |
| R7 | Accidentally shipping a baseline re-export | Low | Explicit rule #5-reject; deep_gemm call is scaffold-only |

--- Original Design Draft End ---
