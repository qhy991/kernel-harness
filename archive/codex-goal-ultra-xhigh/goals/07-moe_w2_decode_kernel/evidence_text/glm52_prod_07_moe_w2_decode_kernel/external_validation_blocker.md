# External production-validation blocker

Date: 2026-07-22

## Blocking facts

This host has four physical NVIDIA B200 GPUs. Production acceptance requires
one node with eight ranks at TP8/DP8/EP8. A four-rank TP4/DP4/EP4 run is a
separate diagnostic lane and cannot satisfy, weaken, or be relabeled as that
gate.

The pinned `sgl-deep-gemm==0.1.4.post1` grouped-masked API also lacks
`enable_overlap`, `max_block_n`, and `signal`. Current Blackwell routed W2
source disables down-GEMM/combine two-stream overlap and is measurable without
those arguments, but this package cannot certify a future overlap-enabled
branch.

No rank count, scheduling hint, mask source, or overlap result is relabeled:

- local M16/M32 buckets are never divided by world size;
- plan `expected_m=4/8` and current-source-derived `expected_m=5/9` remain
  separately named;
- deterministic leaf masks are not called live EP8 router observations; and
- TP4/DP4/EP4 remains diagnostic, never TP8/DP8/EP8 evidence.

## Production evidence unavailable here

| Required evidence | Status |
|---|---|
| Live EP8 `packed_recv_count` / `masked_m` for M16 and M32 | BLOCKED: eight ranks unavailable |
| Exact eight-rank recipe, SM allocation, stream, signal, overlap args, graph state, and selected config | BLOCKED: eight ranks unavailable; overlap-enabled post1 API also unavailable |
| Three uncontended all-stock full-region baselines using maximum rank latency | BLOCKED: eight ranks unavailable |
| Candidate-vs-stock TP8/DP8/EP8 full-region correctness and paired latency | BLOCKED: eight ranks unavailable |
| Production graph/overlap replay across dispatch, W13, quant, W2, and combine | BLOCKED: eight ranks unavailable |
| Eight-rank SGLang end-to-end decode comparison | BLOCKED: eight ranks unavailable |

## Useful local evidence retained separately

| Available lane | Result and authority |
|---|---|
| Exact single-B200 packed-ABI leaf portfolio | complete; [`paired_alignment_summary.json`](paired_alignment_summary.json) |
| Stock and BM16 Nsys/NCU profiles | complete raw matrices; [`validation.md`](validation.md) |
| ptxas/PTX/SASS and JIT identity | complete; [`profile/moe-w2-alignment16/analysis/jit_inventory.json`](../../profile/moe-w2-alignment16/analysis/jit_inventory.json) |
| Repeated single-GPU leaf CUDA Graph replay | strict PASS; [`leaf_validation_summary.json`](leaf_validation_summary.json) |
| Edge-mask active-row correctness | strict PASS for 12 cases; [`leaf_validation_summary.json`](leaf_validation_summary.json) |
| Four-rank TP4/DP4/EP4 lane | strict PASS for [stock-only eager/no-overlap diagnostic](tp4_diagnostic/tp4_20260722T185932Z_1629770_10420/summary.json); never TP8 acceptance |

Alignment 16 improves paired leaf p50 by 6.2--8.7% on the four named workloads,
but its selector is process-global. The local graph/edge evidence proves only
the single-GPU leaf contract. The TP4 attempt contains no candidate and its
region check is structural/repeatability validation without an independent
math oracle. Its logs additionally record default 20-SM DeepEP communication,
failed IBGDA initialization, and NCCL rank-based device inference, so its
latency is a fallback-environment diagnostic rather than tuned communication.
None of these results authorizes production enablement.

Static call mapping and ABI evidence are in
[`reachability.md`](reachability.md). Package/build identity is in
[`stock_deep_gemm_provenance.json`](stock_deep_gemm_provenance.json), and the
stock DeepEP diagnostic dependency is recorded in
[`deepep_overlay_provenance.json`](deepep_overlay_provenance.json).

## Required external lane

Before any bucket is promoted, an eight-B200 environment must:

1. launch the exact current SGLang configuration at TP8/DP8/EP8 with
   `SGLANG_GLM52_OPT=0` and record immutable source, package, model, flags,
   topology, clocks, import, and cache identities;
2. capture live M16 and M32 dispatch outputs, including `masked_m`, plan versus
   current-source scheduling-hint identity, recipes, streams, graph state, SM
   reservation, overlap/signal arguments, return contract, and selected
   DeepGEMM config;
3. collect at least three uncontended stock baselines for the W2 leaf, the full
   dispatch -> W13 -> SwiGLU+quant -> W2 -> combine region, and SGLang decode,
   using maximum latency across ranks;
4. test only a locally justified, fail-closed candidate and preserve exact
   tensor/output, stream, recipe, signal, overlap, graph, and fallback
   semantics; and
5. require a >=3% paired-p50 leaf win plus non-regressing full-region and
   end-to-end results before enabling that exact bucket.

Until those steps pass, every goal-07 bucket remains on stock. The local goal
therefore closes as an evidence-backed no-replacement result, not as an
eight-rank acceptance claim.
