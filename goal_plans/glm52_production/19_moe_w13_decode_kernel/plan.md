# Goal: source-tune the fused W13 decode grouped GEMM

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Optimize the production packed W13 device kernel for real EP8 decode geometry,
rather than optimizing independent N2048 gate or up projections.

## Production target

- `moe_w13_grouped_decode_m16`: E32, slab1024, expected M4, K6144, N4096.
- `moe_w13_grouped_decode_m32`: E32, slab1024, expected M8, K6144, N4096.
- Input is produced by low-latency DeepEP dispatch; output feeds fused
  SwiGLU+quant and W2.

## Work

- [ ] Capture actual `masked_m`, selected DeepGEMM config, SM count, recipes,
  kernel/JIT identity, and streams for both buckets.
- [ ] Establish paired W13 and full eight-rank MoE-region baselines.
- [ ] NCU the stock kernel for weight traffic, expert/tail scheduling, CTA waves,
  tensor activity, TMA stalls, eligible warps, registers, spills, and output stores.
- [ ] Build evidence-backed source variants in an isolated DeepGEMM or CuTe tree.
  Consider tiny-row expert scheduling, fused-N tile geometry, persistent/CLC work
  distribution, SM partition, TMA pipeline, and epilogue only where metrics support
  them.
- [ ] Record ptxas/SASS effects for every resource or instruction hypothesis and
  retain a negative-result ledger.  Do not claim a win by omitting one half of W13
  or padded/masked correctness.
- [ ] Implement static M16/M32 selection and preserve recipes, signals, overlap
  return values, and graph replay.  Unsupported modes fall back immediately.
- [ ] Validate W13, SwiGLU+quant, W2, DeepEP dispatch/combine, and SGLang e2e.

## Completion

Complete with a shared-definition production win or a no-replacement result.  A
40% HBM score on either old separate projection is not a production gate.

## Deliverables

Include real-shape capture, fork/build manifest, NCU/ptxas/SASS artifacts, variant
ledger, paired oracle table, correctness/overlap tests, full-region/e2e results,
and rollback instructions.

