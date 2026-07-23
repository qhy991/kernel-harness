# Goal: trace and optimize the SGLang TP AllReduce path

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Determine which GLM-5.2 serving regions actually call AllReduce, then optimize the
runtime-selected implementation reached through
`GroupCoordinator.all_reduce`.  Do not assume that a standalone NCCL AllReduce is
the production path: current SGLang may select custom all-reduce, quick all-reduce,
PyMSCCL++, torch symmetric memory, out-of-place PyNCCL, in-place PyNCCL, or c10d
according to topology, graph state, message size, and configuration.

If the current production trace contains only AllGather at the suspected site,
record that reachability result and do not substitute AllReduce under the same
operator name.  Continue optimizing only the independently proven AllReduce
callers.

## Fixed diagnostic shapes

Use the existing four-card workloads:

- `tp4_allreduce_decode_m16`: BF16 `[16, 6144]` per rank.
- `tp4_allreduce_decode_m32`: BF16 `[32, 6144]` per rank.
- `tp4_allreduce_prefill`: BF16 `[8192, 6144]` per rank.

If a TP8 production trace proves the same AllReduce ABI is active, add explicitly
named TP8 workloads without overwriting these TP4 diagnostics.  Decode local M
remains 16 and 32 at either world size.

## Authorized optimization

Changes may be made in SGLang backend selection and thresholds, custom/quick
all-reduce CUDA code, PyNCCL, PyMSCCL++, torch symmetric-memory integration,
CUDA-graph buffer registration, stream/event scheduling, a pinned communication
library, or a narrowly gated custom kernel.  Preserve in-place versus out-of-place
semantics, graph capture, allocator ownership, rank ordering, and stock fallback.

## Required work

- [ ] Obtain code and runtime reachability evidence for every material AllReduce
  caller.  Record the selected communicator, decision predicates, message bytes,
  topology, graph mode, stream, output aliasing, and following consumer.
- [ ] Validate that the serving-native workload reproduces the chosen production
  ABI.  A raw torch.distributed candidate is a comparison, not automatically the
  production reference.
- [ ] Capture paired rank-max latency and Nsight Systems traces for M16, M32, and
  prefill.  Attribute host gaps, registration, barriers, NCCL/custom kernels, and
  overlap separately.
- [ ] Compare eligible SGLang backends and thresholds before writing a new kernel.
  For a source-level hypothesis, profile memory transactions, synchronization,
  occupancy, tail behavior, and NVLink utilization and inspect generated code when
  relevant.
- [ ] Test exact reduction values, repeated destructive-input restoration outside
  the timed region, in/out-place aliasing, CUDA Graph replay, streams, and the full
  producer-AllReduce-consumer region.
- [ ] Implement a static message-size/topology/graph oracle only for demonstrated
  wins.  No host synchronization or per-call environment mutation is allowed.

## Completion and deliverables

Four-card results are an independent TP4 diagnostic result.  They may land behind
a TP4-only gate but cannot establish a TP8 production win.  Preserve exact TP8
validation commands and leave TP8 stock behavior active until tested.  Commit
reachability evidence, source/backend identities, raw paired results, profiler
artifacts, tests, dispatch table, attempt ledger, and final disposition to the
isolated branches; do not push remote state.
