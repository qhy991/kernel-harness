# Implementation Plan Draft — `glm52/o_proj_prefill`

> **Retarget note (2026-07-15):** Harness shapes updated to **DP=1 / TP=1 / EP=32** (commit `88d7044`). Numeric axes below were patched in-place; re-benchmark after this change.


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
