# Goal: remove production W2 decode pack and launch overhead safely

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Reduce overhead around the production packed W2 grouped GEMM at decode M16/M32
without adding an online scale adapter and without weakening DeepEP/TBO overlap.
The former E=8 synthetic task is only a component-development tool.

## Production target

| Bucket | Compute workload | Required communication context |
|---|---|---|
| M16 | `moe_w2_grouped_decode_m16` | `deepep_ll_dispatch_decode_m16`, `deepep_ll_combine_decode_m16` |
| M32 | `moe_w2_grouped_decode_m32` | `deepep_ll_dispatch_decode_m32`, `deepep_ll_combine_decode_m32` |

The live EP8 shape uses 32 local experts, expert slab 1024, expected M 4/8, K2048,
N6144, and packed UE8M0 scales.  The complete region is dispatch -> W13 ->
SwiGLU+quant -> W2 -> combine.

## Work

- [ ] Prove whether any scale pack, allocation, layout transform, or launch gap is
  actually present in the production W2 path.  If scales already arrive packed,
  do not port a synthetic float32 packing kernel; pivot to the measured overhead.
- [ ] Baseline the two W2 workloads and the eight-rank low-latency DeepEP dispatch
  and combine tasks.  Capture an nsys timeline of the full MoE region with overlap.
- [ ] Decompose W2 into preparation, allocation, SwiGLU+quant handoff, grouped GEMM,
  signals, waits, and return-contract cost.  Measure floors instead of inferring
  them from synthetic timings.
- [ ] Optimize launch/allocation/layout handling in SGLang or the reached library.
  Source edits may fuse legal preparation, reuse allocation without caching input
  data, improve PDL, or reduce gaps.  Preserve current tensors on every invocation.
- [ ] Pass through `overlap_args`, recipes, `max_block_n`, signal behavior, and the
  overlap return value.  A candidate that works only when overlap is absent cannot
  be promoted to the production bucket.
- [ ] Confirm paired microbench gains, graph replay, full eight-rank MoE-region
  improvement, and SGLang end-to-end improvement.  Use EP4 only as a diagnostic.

## Completion

Complete with a safe production bucket win or an evidenced no-replacement result.
An isolated W2 improvement accompanied by a slower DeepEP-overlapped region is a
rejection, not a partial deployment success.

## Deliverables

Include overhead/floor table, eight-rank timelines, profiler results, source diff,
ABI/overlap tests, paired M16/M32 numbers, full-region numbers, end-to-end result,
and exact bucket fallback policy.

