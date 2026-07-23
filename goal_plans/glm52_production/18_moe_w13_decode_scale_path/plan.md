# Goal: minimize fused W13 decode scale and launch overhead

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Optimize preparation and launch around the fused production W13 decode GEMM for
M16/M32 while retaining packed input scales and the DeepEP low-latency pipeline.
The old separate gate-projection task is not a replacement boundary.

## Production target

| Bucket | Compute | Communication context |
|---|---|---|
| M16 | `moe_w13_grouped_decode_m16` | low-latency dispatch/combine M16 on eight ranks |
| M32 | `moe_w13_grouped_decode_m32` | low-latency dispatch/combine M32 on eight ranks |

Live geometry is E32 local, slab1024, expected M4/8, K6144, fused N4096.  Inputs
come from DeepEP and may already have packed scales.

## Work

- [ ] Trace the exact W13 inputs after DeepEP and prove whether any scale cast,
  layout conversion, allocation, SM reconfiguration, or launch gap remains.  Do not
  add a float32 adapter when production already supplies packed UE8M0.
- [ ] Baseline compute, eight-rank communication, full MoE region, and SGLang decode
  for both buckets.  Capture streams and gaps with nsys.
- [ ] Decompose preparation, grouped GEMM, W13-to-SwiGLU handoff, and any signal or
  wait cost.  Profile the binding component with NCU.
- [ ] Optimize measured overhead in SGLang or an isolated DeepGEMM/DeepEP source
  checkout.  Legal options include removing redundant transforms, allocation reuse
  without data caching, launch fusion, PDL, or per-bucket config.
- [ ] Preserve fused W13 output order, recipes, masks, DeepEP handles, streams,
  graph capture, and every-call dependence on current inputs.
- [ ] Enable a bucket only if the full dispatch-to-combine region and end-to-end
  decode improve.  Keep the other bucket stock when needed.

## Completion

Complete with a safe production win or an evidenced no-replacement result.  A
synthetic pack win or isolated W13 gain that removes overlap is rejected.

## Deliverables

Provide ABI/overhead decomposition, timelines, source diff/build provenance,
paired bucket table, overlap and correctness tests, full-region/e2e results, and
policy.

