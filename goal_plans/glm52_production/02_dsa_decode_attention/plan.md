# Goal: optimize production DSA flashmla_kv decode attention

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Optimize the complete GLM-5.2 DSA decode attention path shown by the production
trace: `flashmla_kv`, whose hot kernels are `flash_fwd_splitkv_mla*` plus combine.
Do not substitute either `flash_mla_sparse_fwd` or the current no-flag SM100
TRT-LLM default for this explicitly selected serving lane.

## Production target

| Bucket | Harness workload |
|---|---|
| local M16 | add `dsa_flashmla_kv_decode_m16` |
| local M32 | add `dsa_flashmla_kv_decode_m32` |

The concrete call chain is `DSAAttentionBackend._forward_flashmla_kv` to
`sgl_kernel.flash_mla.flash_mla_with_kvcache`.  Preserve paged KV layout,
FP8-KV behavior, sparse top-k indices, scheduler metadata, split count, softmax
scaling, output semantics, and CUDA Graph replay.

## Work

- [ ] Trace both buckets from `--dsa-decode-backend flashmla_kv` through the
  SGLang branch to split-KV and combine GPU kernels.  Record FlashMLA source/build
  identity, scheduler metadata, split count, and compiled variant.
- [ ] Establish three paired stock baselines in `serving_native`, then capture the
  containing DSA region and full decode baseline.
- [ ] Profile the fused kernel.  Quantify sparse gather efficiency, L2/HBM traffic,
  tensor-pipe activity, softmax/reduction cost, active warps, register pressure,
  spills, tail waves, and launch gaps.
- [ ] Rank source changes by measured recoverable time.  Changes may be made in
  SGLang, sgl-kernel, FlashMLA, CUTLASS/CuTe, or a narrowly scoped
  custom kernel while keeping an isolated build and stock fallback.
- [ ] Maintain separate variants for M16 and M32 when their best launch/tile
  configurations differ.  Dispatch from device-resident shape metadata or static
  graph bucket knowledge without host synchronization.
- [ ] Run exact correctness, repeated paired timing, graph replay, DSA-region, and
  complete SGLang decode tests before enabling a bucket.

## Completion

At least one production bucket must pass the shared production-win definition;
otherwise finish with a profiler-backed no-replacement disposition.  Results from
the synthetic DSA task are supporting evidence only and must not be compared as the
production baseline.

## Deliverables

Include code/backend mapping, generated source or binary identity, profiler files,
experiment ledger, source changes, per-bucket oracle, graph result, containing
region speedup, and end-to-end throughput/latency comparison.
