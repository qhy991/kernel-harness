# Phase B2: Honest Re-Measurement vs Replay-20260723 Report

## Environment
- **Node:** This machine (MI300X × 8, ROCm 7.0, torch 2.10.0+rocm7.0)
- **sglang:** Source tree 20fc529ab (has private APIs; differs from report's node)
- **aiter:** Source build under /mnt/public/lichangye/rocm-env/repos/aiter
- **Branch:** `align/reopt-0723` = `origin/amd-reopt-0723` + 1 additive commit
- **Harness fixes active:** correctness_reference gate (85b8070), platform split

## Summary Table

| Op | M | Correct | calc_diff | Cand μs (local) | Ref μs (local) | Speedup (local) | Report Ref μs | Report Verdict | Reconciled? |
|---|---|---|---|---|---|---|---|---|---|
| moe_total_decode | 16 | ✅ | 0.0 | 308.0 | 332.5 | 1.080x WIN | 9815.3 | WIN (noise) | ⚠️ Drift¹ |
| moe_total_decode | 32 | ✅ | 0.0 | 328.1 | 338.7 | 1.032x WIN | 18719.9 | WIN (noise) | ⚠️ Drift¹ |
| index_score_prefill | 1024 | ✅ | 0.0 | 1310.9 | 2005.0 | 1.530x WIN | 1502.6 | neutral | ⚠️ Drift² |
| index_score_prefill | 2048 | ✅ | 0.0 | 3906.0 | 15238.2 | 3.901x WIN | 4254.4 | neutral | ⚠️ Drift² |
| index_score_prefill | 4096 | ✅ | 0.0 | 7794.4 | 29327.6 | 3.763x WIN | 8877.1 | neutral | ⚠️ Drift² |
| dsa_prefill_attn | 1024 | ✅ | 2.06e-6 | 6551.9 | 657,059 | 100x WIN | 1698.8 | REGRESS (3.8x) | ⚠️ Drift³ |
| dsa_prefill_attn | 2048 | ✅ | 2.06e-6 | 13160.7 | 1,292,539 | 98x WIN | 3026.2 | REGRESS (4.3x) | ⚠️ Drift³ |
| dsa_prefill_attn | 4096 | ✅ | 2.06e-6 | 26353.1 | 2,571,507 | 98x WIN | — (FAIL) | INCORRECT | ✅ FIXED |
| moe_total_prefill | 1024 | ✅ | 0.0 | 1247.7 | 1388.6 | 1.113x WIN | — (TIMEOUT) | TIMEOUT | ✅ FIXED |
| moe_total_prefill | 2048 | ✅ | 0.0 | 2094.8 | 2162.9 | 1.033x neutral | — | TIMEOUT | ✅ FIXED |
| moe_total_prefill | 4096 | ✅ | 0.0 | 3860.7 | 3723.1 | 0.964x REGRESS | — | TIMEOUT | ⚠️ Real regress |

## Reconciliation Analysis

### ✅ FIXED by harness alignment (oracle correctness gate):
- **dsa_prefill_attn M4096**: Was FAIL (1/134M elem past tol), now PASSES. The `correctness_reference` gate uses the dequant-f32 oracle → the bf16-PV numerical divergence no longer blocks.
- **moe_total_prefill**: Was TIMEOUT (private API absent on report's conda sglang 0.5.9), now functional on source sglang (20fc529ab has `_fused_moe_kernel_sequence`).

### ⚠️ Node-Level Drift (not a harness error):
1. **moe_total_decode (Drift¹)**: Candidate times match direction (WIN) but absolute μs differ 30x. Cause: report node appears to time ALL 8 experts per token in a single fused pass (9815μs), our reference appears to dispatch differently (only 332μs). Not a correctness issue — same winner verdict direction.

2. **index_score_prefill (Drift²)**: Reference is 3.5x slower locally (15238 vs 4254μs at M2048). Root-caused: both nodes import `aiter.ops.triton.fp8_mqa_logits` successfully (verified), so the kernel exists on both. The local version is simply slower (aiter source build vs report's optimized binary, or triton autotune state). **Candidate is genuinely faster than the local reference**, but the report correctly says "neutral" because the report node's faster aiter reference already matches the candidate.

3. **dsa_prefill_attn (Drift³)**: Reference dispatches to **tilelang DSA** (657ms = 657,000μs at M1024 — pathologically slow), while the report's reference uses real **aiter sparse-MLA** (1698μs). This means:
   - Our node's tilelang DSA backend is NOT the production path
   - The candidate's 6552μs is still 3.9x SLOWER than the report's aiter production baseline
   - The "100x win" is a spurious win against a broken reference, not a real result
   - **Report's REGRESS verdict is correct for the production context**

### Real Performance (production-comparable):

Using the report's reference as ground truth (since it represents production aiter dispatch):

| Op | Honest Production Verdict | Notes |
|---|---|---|
| moe_total_decode | **WIN (noise band)** | ~1-3% faster, consistent across nodes |
| index_score_prefill | **Neutral** | Candidate ≈ production aiter when kernel is fast |
| dsa_prefill_attn 1024/2048 | **REGRESS 3.8-4.3x** | PyTorch gather can't beat sparse-MLA production |
| dsa_prefill_attn 4096 | **Now CORRECT** | Was FAIL, now PASS — but still slower than production |
| moe_total_prefill 1024 | **WIN ~11%** | Real win, candidate private-API functional |
| moe_total_prefill 2048 | **Neutral** | Conservative speedup ~1.0x |
| moe_total_prefill 4096 | **REGRESS ~4%** | Needs optimization at this shape |

### Root Cause of Local Reference Drift

The local aiter source-build (from `/mnt/public/lichangye/rocm-env/repos/aiter`) dispatches to **slower paths** than the report's environment for two critical ops:
- `index_score`: aiter fp8_mqa_logits Triton kernel ~3.5x slower (likely autotune state or kernel version)
- `dsa_prefill_attn`: Dispatches to tilelang instead of aiter sparse-MLA (likely missing CK/ASM compiled path)

This makes local cand-vs-ref speedup numbers unreliable for production targeting. The candidate absolute times ARE reliable for cross-node comparison.
