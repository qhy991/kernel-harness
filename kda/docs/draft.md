# KDA Phase-1 Draft — `glm52/moe_gate_proj_prefill`

> **Retarget note (2026-07-15):** Harness shapes updated to **DP=1 / TP=1 / EP=32** (commit `88d7044`). Numeric axes below were patched in-place; re-benchmark after this change.


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
