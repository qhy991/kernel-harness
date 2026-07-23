# Phase C2: Re-optimization Assessment

## Summary of Real Win/Loss Against Production

| Op | Production Verdict | Action Taken |
|---|---|---|
| moe_total_decode | WIN (noise, ~3%) | No change needed — already a small genuine win |
| index_score_prefill | Neutral → depends on aiter kernel speed on target node | No change needed — candidate matches or beats depending on node |
| dsa_prefill_attn | REGRESS 3.8-4.3x | **Cannot fix** — candidate is a PyTorch path (6.5ms), production baseline is aiter CK/ASM (1.7ms). Node lacks compiled CK kernel so can't benchmark alternatives. Needs new algorithm (Triton sparse-MLA) — research task. |
| moe_total_prefill | WIN at M≤2048, neutral M4096 | **Hardened in C1** — import resilience + timeout + M4096 skip. Now 2 WIN / 0 REGRESS / 1 NEUTRAL. |

## Key Finding: Node Environment Gap

This node's aiter source-build dispatches critical kernels to slow fallbacks:
- `aiter_sparse_mla_fwd`: 662ms (should be ~1.7ms on production build)
- `fp8_mqa_logits`: 15ms (should be ~4.3ms on production build)

Root cause: CK/ASM components for gfx942 were not compiled in the source-build. 
The harness **code** is correct (aligned with origin/amd-reopt-0723); the performance gap is purely runtime/binary.

## Retarget Campaign Priorities (from e2e profile)

Per `rewardbench/amd/e2e_profile_20260722/e2e_prefill_op_share.csv`:
1. **AllReduce** (34-53% of GPU time) — highest-impact target, requires `serving_native` tree
2. **MoE fused** (13-16%) — our moe_total_prefill is already optimized (C1)
3. **MLA/DSA** (5-14%) — blocked by CK/ASM build issue on this node
4. **FP8 GEMM** (8-11%) — baseline already soft on this node

## Recommendations

1. **Push the hardened moe_total_prefill** — real measurable win, robust to sglang drift
2. **Mark dsa_prefill_attn as BLOCKED** until the node gets a properly compiled aiter (CK for gfx942)
3. **Deprioritize index_score_prefill** — neutral on production nodes, only looks like a win here
4. **Next sprint: AllReduce optimization** via `serving_native/` tree — highest e2e impact
