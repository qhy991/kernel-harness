# Phase-1 Implementation Draft — `glm52/moe_gate_proj_decode`

> **Retarget note (2026-07-15):** Harness shapes updated to **DP=1 / TP=1 / EP=32** (commit `88d7044`). Numeric axes below were patched in-place; re-benchmark after this change.


Target machine: **NVIDIA B200 (sm_100)**. Env: `torch 2.11.0+cu130`, `deep_gemm 0.1.4`,
`triton 3.6.0`, CUPTI present. Authoritative edit target:
`testbench/tasks/glm52/moe_gate_proj_decode/solution.py` (do **not** edit `reference.py`).

Goal of Phase 1: a **first correct, independent** B200 implementation that matches
`reference.py` within tolerance on both sweep shapes `M ∈ {16, 32}`, and that is a real
kernel we can later optimize — **not** a thin re-export of `deep_gemm.fp8_m_grouped_gemm_nt_masked`.

---

## 1. Baseline behavior (what we must reproduce)

The op is a **decode-path masked grouped FP8 GEMM** for GLM-5.2 MoE Gate Projection,
EP32-local `E=8`:

```
out[e, :, :] = a[e] @ b[e].T        for each expert e with masked_m[e] > 0
   a[e]  : [Mp, K]  fp8_e4m3   (rows = padded token slots)
   b[e]  : [N,  K]  fp8_e4m3   (expert weights, "NT" => b is transposed inside)
   out[e]: [Mp, N]  bfloat16
   K = 6144, N = 2048, E = 32
```

`run()` today just calls `deep_gemm.fp8_m_grouped_gemm_nt_masked((a_fp8,a_s),(b_fp8,b_s), out, masked_m, expected_m)`.
`solution.py` is currently **byte-identical to `reference.py`** — this is the forbidden re-export
and is the thing we replace.

### Harness contract (verified by reading `testbench/harness/*.py`)

- `driver.py:186` builds inputs from the **reference's** `get_inputs`; the candidate's own
  `get_inputs` is **ignored**. Both `ref_run` and `cand_run` are called positionally with
  **independent clones** of the *same* input list. → Our `run()` must consume the inputs
  exactly as DeepGEMM produces them, including scale-factor layout.
- Positional signature (definition order): `run(a_fp8, a_s, b_fp8, b_s, out, masked_m,
  expected_m, m_indices, layout)`. For this task `layout==1` (masked) always; `m_indices`
  is empty and unused.
- Correctness (`correctness.py`): fp32 compare, `|c-r| <= atol + rtol*|r|`,
  **matched_ratio ≥ 0.999**, `atol=0.1`, `rtol=0.05`, plus a NaN/Inf gate. The compared
  tensor is `run()`'s **return value** (`out`), full `[E, Mp, N]`.
- Timing (`timing.py`): CUPTI **cold-L2** (L2 flushed each iter), median device-kernel ms,
  args re-cloned per iter. Every kernel we launch inside `run()` counts in the timed span
  (min-start → max-end). Not a Phase-1 gate, but shapes the later design.
- Reward-hack guard (`reward_hack.py`): rejects monkey-patched `elapsed_time` and
  non-`torch.Tensor` (lazy/proxy) outputs; **re-runs** one fresh call after timing and
  re-checks it against the oracle. → No cross-iteration caching / stateful shortcuts;
  every call must honestly recompute. An honest kernel is unaffected.

### Empirically measured input tensors (probe on this B200)

| name         | shape            | dtype        | meaning |
|--------------|------------------|--------------|---------|
| `a_fp8`      | `[8, 128, 6144]`| `float8_e4m3`| activations `[E, Mp, K]` |
| `a_s`        | `[8, 128, 12]`  | `int32`      | UE8M0 scales, **4 K-block exponents packed per int32** (48 K-blocks → 12 words) |
| `b_fp8`      | `[8, 2048, 6144]`| `float8_e4m3`| weights `[E, N, K]` |
| `b_s`        | `[8, 2048, 12]` | `int32`      | UE8M0 per-(128×128)-block scale, **expanded per N-row**, 4 packed per int32 |
| `out`        | `[8, 128, 2048]`| `bfloat16`   | output `[E, Mp, N]` (uninitialized `torch.empty`) |
| `masked_m`   | `[32]`           | `int32`      | valid rows per expert (e.g. M16: `[1,1,0,2,…]`, sum≈14; M32: sum≈31) |
| `expected_m` | scalar           | `int32`      | DeepGEMM perf hint (2 / 3); irrelevant to correctness |
| `m_indices`  | `[0]`            | `int32`      | unused for masked layout |
| `layout`     | scalar           | `int32`      | `== 1` (masked) |

`Mp = align(max_count, 128) = 128` for both shapes (max count ≤ 3). So there is **exactly
one 128-row M-block per expert**, always. Active experts: **12 (M16) / 21 (M32)**.

---

## 2. Two decisive facts that de-risk correctness (both empirically proven)

### Fact A — Masked write is BLOCK_M=128 granular (per-expert on/off)

Probe: sentinel the `out` clone, run the reference, diff. Result on both shapes:

```
written_rows_per_expert == ceil(masked_m/128)*128     (TRUE)
written_rows_per_expert == masked_m  (row-granular)   (FALSE)
```

Because `Mp=128` and `masked_m ≤ 3 < 128`, every **active** expert (`masked_m>0`) has its
**full 128 rows written** with real GEMM values (the padded rows are ordinary `randn`
activations run through the GEMM), and every **inactive** expert (`masked_m==0`) is **left
untouched** (retains the input-clone bytes, which are identical between ref and candidate).

**Consequence:** our masking collapses to *"process expert `e` iff `masked_m[e] > 0`, and
compute/write the entire `[128, N]` tile."* No partial-row predication is needed, and
`expected_m` / `masked_m` values (beyond the `>0` test) do **not** affect correctness. This
is a major simplification — padded rows are not special-cased, they are just more GEMM rows
sharing the same inputs, so a full-tile compute reproduces the oracle bit-for-tolerance.

> Guard: if a future shape ever produced `masked_m[e] > 128` (would need `M`≳128 with heavy
> skew), `Mp` grows and multiple M-blocks appear. For the fixed sweep `{16,32}` this cannot
> happen, but the kernel should still gate rows on `Mp` generically.

### Fact B — Plain scales are cleanly recoverable; full recipe validated

`a_s`/`b_s` are UE8M0 exponents packed 4-per-`int32` along K, logically `[E, mn, K/128]`
(`mn = Mp` for a, `N` for b — B's 128×128 block scale is materialized per N-row). Unpack:

```python
def unpack_scale(s_int32):                 # [..., 12] int32 -> [..., 48] fp32
    b = s_int32.contiguous().view(torch.uint8).to(torch.int32)   # 4 bytes / word, K-order
    return (b << 23).view(torch.float32)   # ue8m0 byte E -> 2^(E-127)
```

Probe #2 (torch round-trip): dequantize `a_real = a_fp8 * scale_a`, `b_real = b_fp8 *
scale_b` (scale broadcast over its 128-wide K-block), then `out_e = (a_real @ b_real.T)`
cast to bf16, per active expert. Compared to the DeepGEMM reference:

```
M=16: max_abs=0.0156  max_rel=0.0078  worst_expert_bad_frac=0.00000
M=32: max_abs=0.0156  max_rel=0.0078  worst_expert_bad_frac=0.00000   (need bad_frac ≤ 0.001)
```

Zero out-of-tolerance elements, `max_abs` (0.0156) is ~6× under `atol=0.1`. The block-scaled
GEMM math is identical to full-dequant-then-matmul (scales are constant within each 128-K
block), so **the entire independent numerical recipe is proven correct.** The only remaining
work is packaging it as an efficient, honest B200 kernel.

---

## 3. Baseline performance (advisory — Phase 1 does not gate on speed)

`harness.profile --shape 16` (warm-L2 advisory): **median ≈ 88.9 µs**, roofline
**5227 GB/s = 65.3% HBM peak, memory-bound**, `active_experts=12`,
`useful_vs_padded_util=0.0034`, `useful_tflops≈4.0`.

Interpretation: the op is **weight-bandwidth-bound**. Cost ≈ streaming each active expert's
weights `b[e]` (`N·K` fp8 = 2048·6144 = 12.6 MB) once: ~8·12.6 MB ≈ 101 MB (M16). `a` and
`out` are tiny. Compute is ~0.3% "useful" (only ≤3 of 128 rows matter) but that is *free*
because we are bandwidth-bound, so cutting padded compute will **not** cut latency. The real
lever (Phase 2+, not now) is pushing HBM utilization from 65% → 90%+ on the weight stream
(TMA multicast/pipelining, tcgen05, 2-SM). Independent target for the eventual WIN gate:
`solution_us < baseline_us` per shape under CUPTI cold-L2.

---

## 4. Prior art (KernelWiki, cutoff 2026-04-27)

- `wiki/kernels/grouped-gemm.md` (`kernel-grouped-gemm`): the canonical MoE grouped-GEMM
  page. Confirms **Layout 2 = masked (decode + CUDA-graph)** with `A:[E,M_max,K]`,
  `B:[E,N,K]`, `C:[E,M_max,N]` — exactly our shapes. Documents static (precomputed
  tile→expert map) vs **dynamic/persistent (atomic tile counter, CLC)** scheduling, and the
  caveats we hit: *thin-GEMM inefficiency at small M*, *masked layout wastes compute on
  padding*, *TMA 128-B alignment*. Also documents the GPU-Mode P4 **reward hack** (cross-call
  result caching) — a reminder to stay honest; our harness's `reward_hack.py` would catch it.
- `wiki/kernels/deepgemm.md` (`kernel-deepgemm`): DeepGEMM FP8 fine-grained (128×128) block
  scaling — the baseline family and math we are matching.
- `wiki/kernels/fp8-block-scale-gemm.md`, `wiki/kernels/fused-moe.md`: FP8 block-scale GEMM
  and FP8 block-scale MoE routing + dual GEMM structure.
- PRs for later optimization reference: `pr-sglang-13731`/`14640` (MXFP8 grouped GEMM on
  B200), `pr-flashinfer-1086` (accelerate Blackwell grouped GEMM), `pr-flashinfer-2503` /
  `pr-sglang-9199` / `pr-vllm-25990` (`grouped_gemm_nt_masked` masked-decode path),
  `pr-sglang-5432` (DeepGEMM group_gemm_masked as decode GEMM), `pr-sglang-16622` (FP8 MoE
  NaN fix on Blackwell — numerical-stability caution).

Triton feasibility (triton 3.6.0, sm_100): we do **not** need `tl.dot_scaled`. Since scales
are constant per 128-K block, a standard K-loop with `BLOCK_K=128` — load fp8 tile, cast,
`tl.dot` with fp32 accumulate, multiply the block partial by the per-row × per-N-block scalar
scale, accumulate — reproduces the block-scaled GEMM. `tl.dot` on fp8-e4m3 (or bf16) is
supported on Blackwell in this Triton version.

---

## 5. Ranked candidate directions

### C1 — Triton masked grouped FP8 GEMM *(primary; recommended first landing)*
One Triton kernel; grid `(E, ceil(N/BLOCK_N))` (or persistent). Per program: early-exit if
`masked_m[e]==0`; else loop K in `BLOCK_K=128` steps, `tl.dot(a_fp8_kb, b_fp8_kb)` (fp32
acc), scale each K-block partial by `sa[m,kb] ⊗ sb[nblk,kb]` (outer product of per-row and
per-N-block UE8M0 scalars), accumulate, cast to bf16, store the full `[128, BLOCK_N]` tile.
Consumes `a_s`/`b_s` directly via logical indexing + unpack (no host-side reshuffle).
- **Pros:** fully independent, self-contained, portable, clearly optimizable (tiling, warp
  spec, `num_stages`, persistent scheduling later); matches the proven recipe exactly.
- **Cons:** thin M (128 padded rows, ≤3 real) → low tensor-core utilization; unlikely to
  beat DeepGEMM on latency in Phase 1 (fine — speed is not the gate).
- **Risk:** low. Numerics validated; scale unpack validated; masking validated.

### C2 — Torch/CuTe-free reference-recipe fallback *(safety net / correctness oracle)*
The exact Probe-#2 recipe in a few lines of torch (`unpack_scale`, per-active-expert
`(a_real @ b_real.T).to(bf16)`, skip `masked_m==0`). Independent of DeepGEMM's GEMM (uses
`torch.matmul`), guaranteed correct (already measured `bad_frac=0.0`).
- **Use:** land this **first** to bank a green `correct=true`, then swap the matmul for the
  C1 Triton kernel. Also serves as the local numerical oracle while iterating C1.
- **Cons:** slow (bf16 dequant materializes full weights); acceptable for Phase 1 exit only,
  not a good long-term body — but it is not a re-export of the baseline op, so it satisfies
  the Phase-1 "independent" rule while C1 matures.

### C3 — Custom scale prologue + `deep_gemm.bf16_m_grouped_gemm_nt_masked` *(hybrid)*
Our own Triton/torch dequant kernel (fp8+UE8M0 → bf16 `a`,`b`), then DeepGEMM's **bf16**
masked grouped GEMM.
- **Pros:** offloads the matmul to a tuned Blackwell kernel; dequant is genuinely ours.
- **Cons:** materializing bf16 weights **doubles** the HBM traffic (fp8→bf16), so it will be
  *slower* than the fp8 baseline and is a poor optimization base; also still leans on a
  DeepGEMM kernel. De-prioritized.

### C4 — Custom CUDA/CuTe blockscaled grouped GEMM (tcgen05 + TMA) *(Phase 2+ only)*
Hand-written sm_100 kernel consuming the native SF layout. Highest ceiling, highest effort;
out of scope for a first correct landing.

**Plan:** ship **C2** to secure `correct=true`, immediately develop **C1** as the real
Phase-1 deliverable (independent + optimizable), keep C2 as the in-repo oracle. C3/C4 are
explicitly Phase-2 material.

---

## 6. First concrete steps

1. **Oracle harness (local):** save Probe #2 as `kda/tools/oracle.py` (unpack + per-expert
   `torch.matmul` recipe) to compare any candidate `out` against DeepGEMM per shape offline
   (fast, no CUPTI).
2. **Land C2** into `solution.py`: `run()` = unpack scales → per active expert
   `(a_real @ b_real.T).to(bf16)` into `out`; leave `masked_m==0` experts untouched; return
   `out`. Run the authoritative gate → expect `correct=true` on M16 & M32.
3. **Develop C1** Triton kernel behind the same `run()` interface; validate against the C2
   oracle offline first (bit-tolerance), then via the authoritative gate.
4. Record every attempt in `kda/candidates.jsonl` + `kda/benchmark.csv`; keep any NCU
   captures under `kda/profile/`.
5. Only after a correct C1 lands, consider perf (persistent scheduling, `BLOCK_N`/stages
   sweep, HBM-utilization). Phase-2 territory.

## 7. Exact validation commands

```bash
cd /home/qinhaiyan/KDA-Exp/worktrees/glm52-moe_gate_proj_decode
PY=/home/qinhaiyan/Kernel-Harness/.venv/bin/python

# Fast advisory smoke (per shape)
PYTHONPATH=testbench $PY -m harness.profile \
  testbench/tasks/glm52/moe_gate_proj_decode --shape 16
PYTHONPATH=testbench $PY -m harness.profile \
  testbench/tasks/glm52/moe_gate_proj_decode --shape 32

# Authoritative gate (correctness + CUPTI cold-L2 latency)
$PY testbench/bin/evaluate.py testbench/tasks/glm52/moe_gate_proj_decode --max-workloads 1
$PY testbench/bin/evaluate.py testbench/tasks/glm52/moe_gate_proj_decode
# exit 0 = WIN (correct & faster on every shape), 1 = correct but not faster, 2 = incorrect
```

Offline oracle check while iterating (no harness): compare candidate `out` to the Probe-#2
recipe per active expert; require max_abs well under 0.1 and zero out-of-tolerance elements.

## 8. Evidence rules — promote / revise / reject

- **Promote** a candidate when: authoritative `evaluate.py` reports `correct=true` on **both**
  M16 & M32 (matched_ratio ≥ 0.999, no NaN/Inf), and the offline oracle shows zero
  out-of-tolerance elements. Record latency (advisory) for tracking. This is Phase-1 exit.
- **Revise** when: correct but suspicious/fragile — e.g. numeric margin thin (`max_abs`
  approaching 0.1), padded/inactive-expert regions differ from the input clone, or latency
  regressed badly vs C2. Fix scale unpack / masking / accumulation and re-measure.
- **Reject** when: any shape `INCORRECT`/`REWARD_HACK`/`RUNTIME_ERROR`; matched_ratio < 0.999
  (usually means padded rows or inactive experts were mishandled — the 99.6%/34% padded
  fraction makes this the dominant failure mode); NaN/Inf; or the body is a re-export of
  `fp8_m_grouped_gemm_nt_masked` (violates the Phase-1 "independent" rule).

## 9. Key risks / unknowns (with mitigation)

- **R1 Padded-row / inactive-expert matching** *(highest impact)* — padded rows are 99.6%
  (M16) / 34% (M32) of elements; mishandling tanks matched_ratio. *Mitigated:* Fact A proves
  the rule (write full 128 rows for active experts, touch nothing for `masked_m==0`); both
  ref and candidate start from identical `out` clones.
- **R2 Scale layout drift** — if a future `deep_gemm` changed SF packing, `unpack_scale`
  would break. *Mitigated:* pinned to 0.1.4; round-trip oracle catches it instantly; unpack
  is a 2-line, testable transform. Keep the oracle in-repo.
- **R3 Triton fp8 `tl.dot` numerics on sm_100** — accumulation/rounding could differ from
  DeepGEMM. *Mitigated:* fp32 accumulate + per-128-K-block scaling mirrors DeepGEMM; validate
  against oracle (0.0156 headroom under 0.1 atol is generous).
- **R4 `Mp>128` regime** — not reachable for `{16,32}` but would need multi-M-block logic.
  *Mitigated:* write the tile loop generically over `Mp`; document the assumption.
- **R5 Timed-region honesty** — any caching across iterations is caught by `reward_hack.py`'s
  post-timing re-check. *Mitigated:* stateless kernel, recompute every call.
- **R6 fp8 `tl.dot` operand layout (NT)** — `b` is `[N,K]` and we need `a[M,K] @ b[N,K].T`;
  ensure the Triton dot contracts on K with `b` transposed via strides, not a materialized
  transpose. *Mitigated:* standard NT `tl.dot` pattern; validate against oracle.
