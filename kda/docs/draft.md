# KDA Phase-1 Draft — glm52/o_proj_decode

> **Retarget note (2026-07-15):** Harness shapes updated to **DP=1 / TP=1 / EP=32** (commit `88d7044`). Numeric axes below were patched in-place; re-benchmark after this change.


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
  the Phase-1 independence constraint. Not pursued.

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
