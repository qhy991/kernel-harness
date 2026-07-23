# Goal: optimize the production SGLang DP AllGather path

Read `../COMMON_RULES.md` first and treat it as part of this goal.

## Outcome

Optimize the AllGather operation observed as a serving bottleneck, starting from
the exact SGLang call reached by GLM-5.2 rather than a standalone NCCL example:

`tensor_model_parallel_all_gather` or its live caller ->
`GroupCoordinator.all_gather_into_tensor` -> the runtime-selected PyNCCL/NCCL or
torch.distributed backend.

Prove the actual caller, group, backend, stream, graph mode, tensor ownership, and
consumer before changing code.  Preserve rank order and the caller-provided output
buffer.

## Fixed workloads

Production TP8/DP8 acceptance uses:

- `dp_allgather_decode_m16`: BF16 `[16, 6144]` per rank.
- `dp_allgather_decode_m32`: BF16 `[32, 6144]` per rank.
- `dp_allgather_prefill`: BF16 `[4096, 6144]` per rank.

The existing four-card diagnostics are:

- `tp4_allgather_decode_m16` and `tp4_allgather_decode_m32`.
- `tp4_allgather_prefill`.

Do not change local M16/M32 for DP8.  TP4 timings may select candidates and expose
regressions but cannot promote a TP8 implementation.

## Authorized optimization

Changes may be made in SGLang distributed dispatch, PyNCCL integration, CUDA graph
handling, stream/event scheduling, registered or symmetric buffers, a pinned NCCL
or communication-library source checkout, or a topology- and shape-gated custom
NVLink kernel.  Consumer fusion or overlap is allowed only when the exact serving
data dependency supports it.  Keep stock NCCL/PyNCCL available as fallback.

## Required work

- [ ] Trace the live production call and record group membership, topology,
  transport/backend, NCCL algorithm/protocol, input/output addresses, stream,
  CUDA Graph behavior, and immediate consumer for decode and prefill.
- [ ] Audit the existing serving-native reference against the live ABI.  Fix only
  production-suite code or add a separately named workload when a real mismatch is
  demonstrated; do not alter the frozen synthetic harness.
- [ ] Capture paired per-rank and rank-max baselines.  Use Nsight Systems to
  separate launch/event gaps, synchronization, communication time, and useful
  overlap.  Record NVLink traffic and NCCL kernel identity.
- [ ] Compare only graph-safe, ABI-compatible candidates: PyNCCL versus c10d where
  reachable, registered/symmetric buffers, algorithm/protocol thresholds,
  persistent/custom small-message kernels, launch reduction, and producer/consumer
  overlap.  Do not time setup, communicator creation, or a new packing copy as a
  hidden adapter.
- [ ] Validate exact gathered rank order and values, repeated graph replay, stream
  ordering, odd/tail cases within the fixed buckets, and the containing SGLang
  region.  Record each negative result.
- [ ] Build a topology x M x backend oracle with no device-to-host dispatch.  A
  candidate winning only on four ranks must remain disabled in the TP8 lane.

## Completion and deliverables

Promote only after an uncontended eight-rank paired win of at least 3% plus a
containing-region and end-to-end improvement.  On this four-GPU host, finish the
diagnostic candidate, profiler evidence, source diff, exact eight-rank rerun
commands, and fallback gate without claiming production promotion.  Commit raw
results, topology/backend identities, tests, graph evidence, attempt ledger, and
the final disposition to the isolated task branches; do not push remote state.
