# Goal: source-tune DeepGEMM for small-M attention O projection

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Use source-level DeepGEMM or CuTe changes to improve the production packed
O-projection GEMM at local M16 and/or M32, with a reproducible variant portfolio
and a static fallback to stock SGLang.

## Production target

`linear_attn_o_decode_m16/m32`, N6144, K16384, BF16 output, packed UE8M0 scales,
SM100.  The baseline is the exact DeepGEMM build imported by SGLang, not the frozen
synthetic reference.

## Work

- [ ] Lock stock and experimental library commits, independent JIT caches, import
  paths, compile flags, and selected kernel configs.  Verify reference and candidate
  never resolve to the same unintended binary.
- [ ] Record three paired production baselines and profile stock M16/M32.  Quantify
  weight bandwidth, waves, CTA count, TMA stalls, issue activity, occupancy,
  registers, spills, epilogue cost, and launch floor.
- [ ] Create evidence-backed source variants for small M.  Investigate only measured
  levers such as tile/cluster shape, N partition, persistent scheduling, SM clamp,
  pipeline stages, scale loads, and epilogue stores.
- [ ] For every variant record expected instruction/resource change, ptxas output,
  relevant SASS, NCU delta, paired latency, correctness, and rejection reason.
- [ ] Select independently by M bucket with a static device-side-safe oracle.  Do
  not add a runtime tuner or host read to decode.
- [ ] Integrate the winning overlay into SGLang with a pinned manifest and stock
  fallback.  Validate CUDA Graph replay, attention layer, and end-to-end decode.

## Completion

Finish with a shared-definition production win or a no-replacement disposition
that identifies the physical/library floor.  An arbitrary 40% HBM target is a
stretch metric, not a reason to enable a slower production variant.

## Deliverables

Provide fork/build manifest, variant table, NCU/ptxas/SASS artifacts, source diff,
paired M16/M32 oracle table, correctness, graph/layer/e2e validation, and rollback
instructions.

