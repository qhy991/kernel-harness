# Goal: optimize production DeepEP dispatch and combine

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Optimize DeepEP itself as a serving bottleneck, independently of W13/W2 GEMM
optimization.  Cover the complete communication pairs reached through SGLang's
`DeepEPDispatcher` and `DeepEPBuffer` facade:

- Decode: `low_latency_dispatch` and `low_latency_combine`.
- Prefill/extend: `get_dispatch_layout`, normal `dispatch`, and normal `combine`.

Preserve the dispatch-produced handle and every recipe, signal, event, hook,
stream, FP8/UE8M0, token-count, expert-alignment, and overlap contract.  Dispatch
and combine must be assessed individually and as a pair around the MoE compute
region.

## Fixed workloads

Production EP8/DP8:

- `deepep_ll_dispatch_decode_m16` / `deepep_ll_combine_decode_m16`.
- `deepep_ll_dispatch_decode_m32` / `deepep_ll_combine_decode_m32`.
- `deepep_normal_dispatch_prefill` / `deepep_normal_combine_prefill`, local
  M=4096.

Four-card diagnostics:

- `ep4_deepep_ll_dispatch_decode_m16/m32` and matching combine workloads.
- `ep4_deepep_normal_dispatch_prefill` and matching combine workload, local
  M=8192.

The EP4 expert partition differs from EP8 and is never a production substitute.

## Authorized optimization

Changes may be made in SGLang's DeepEP integration, a pinned DeepEP source
checkout, its CUDA kernels, configuration search, buffer sizing/registration,
signal protocol, QP/channel selection, SM allocation, stream/event scheduling,
TBO overlap, FP8 packing integration, or a narrowly scoped fused operation.  Keep
the stock DeepEP package and configuration as reference and fallback; never edit
an installed wheel without preserving a reproducible source patch and build.

## Required work

- [ ] Record SGLang and DeepEP source/build identity, import resolution, topology,
  NVLink/RDMA mode, local experts, buffer sizes, QPs/channels, configuration,
  stream/event graph, and concrete CUDA kernels for all four operation types.
- [ ] Audit the serving-native setup against SGLang's real AUTO-mode buffer
  creation and dispatcher ABI.  Verify that combine uses the exact handle emitted
  by the paired dispatch and that correctness observes only valid received tokens.
- [ ] Capture paired per-rank and rank-max baselines plus Nsight Systems traces.
  Attribute layout calculation, FP8/scale packing, signal kernels, dispatch/combine
  transport, synchronization, hooks, launch gaps, and overlap with W13/W2.
- [ ] Search DeepEP configurations separately for decode M16, decode M32, and
  prefill.  Then make one measured source or integration change at a time.  Avoid
  improving an isolated kernel by reducing actual compute/communication overlap.
- [ ] Validate token routing, top-k ids/weights, packed scale semantics, received
  counts, combine output, graph replay, async hooks/events, repeated calls, and the
  full `dispatch -> W13 -> SwiGLU+quant -> W2 -> combine` region.
- [ ] Build a static phase x M x topology oracle.  A winning dispatch may be
  enabled while combine remains stock, or vice versa, only if their shared handle
  and configuration contracts remain compatible and the paired region wins.

## Completion and deliverables

Use EP4 to find and reject candidates, but require EP8 paired measurements before
production promotion.  On this host, leave EP8 fallback active and provide exact
eight-rank rerun commands when external validation is the remaining gate.  Commit
source/build provenance, configs, raw rank-wise results, profiler artifacts,
correctness and graph tests, full-region results, attempt ledger, enable/fallback
policy, and final disposition to the isolated branches; do not push remote state.
