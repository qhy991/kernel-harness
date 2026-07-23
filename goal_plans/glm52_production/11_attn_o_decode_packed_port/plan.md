# Goal: port an O-projection candidate to the native packed ABI

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Convert the strongest historical O-projection decode candidate into a true
production-native implementation, then retain only M16 and/or M32 buckets that
beat stock packed DeepGEMM.  Do not carry forward any per-call float32 scale pack
whose only purpose was to bridge the synthetic task.

## Production target

- `linear_attn_o_decode_m16`
- `linear_attn_o_decode_m32`
- packed int32 UE8M0 inputs already produced by the caller
- current SGLang output allocation, dtype, stream, and CUDA Graph contract

## Work

- [ ] Inventory the candidate and label every operation as intrinsic production
  work, synthetic-ABI adapter, allocation, launch, or observation overhead.
- [ ] Remove adapters and accept the production packed tensors directly.  Verify
  that the resulting core is materially different from stock DeepGEMM; if it is
  identical, measure and document the expected no-op before making more changes.
- [ ] Establish alternating paired baselines and capture nsys component timelines
  for stock and ported paths at M16/M32.
- [ ] Optimize any measured residual overhead in SGLang or an isolated kernel
  library.  Permitted work includes allocation removal, launch fusion, static
  descriptor reuse that does not cache input data, PDL, and shape-specific library
  configuration.
- [ ] If device code changes, collect NCU plus ptxas/SASS evidence tied to the
  hypothesis.  Do not hand-roll the main FP8 GEMM without a measured library-kernel
  limitation and a credible lower bound.
- [ ] Validate byte/layout semantics, numerical correctness, graph replay, layer
  timing, and end-to-end decode.  Enable buckets independently.

## Completion

Complete with at least one deployable bucket or with a no-replacement result that
shows the packed port converges to the stock path and cannot cover its added
overhead.  A gain against f32-scale DeepGEMM is not accepted.

## Deliverables

Include adapter-removal diff, path equivalence analysis, paired results, nsys/NCU
artifacts, build provenance, graph/layer/e2e results, and per-bucket policy.

