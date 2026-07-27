# TP4 DP AllGather profiler analysis

Disposition boundary: this is a four-rank diagnostic trace, not TP8 production
evidence. Profiler timing is not benchmark timing.

## Valid trace

`profiler/tp4/20260722T154605Z-tp4_allgather_decode_m16-reference/` is a complete,
post-run-guarded stock M16 capture from Kernel-Harness `1205f9a` and clean SGLang
`f93f8867b4bc124c9809c9110ec7361ed11b6b4a`. It used physical GPUs 0-3, stable
all-pairs `NV18` topology, `SGLANG_GLM52_OPT=0`, NCCL 2.28.9, one warmup, and
three timed graph replays. The `.nsys-rep` is the canonical raw trace; the exported
SQLite is stored losslessly as `.sqlite.zst` and can be regenerated from the report.

Nsight event tracing inflated the rank-max timing to about 4.16 ms versus the
unprofiled ~0.09-0.10 ms baseline. Those profiled durations are rejected for every
performance claim.

## Collective identity and transport

NCCL TRACE records `count=98304`, BF16 datatype 9, four ranks, and a 196,608-byte
local payload. Auto selection is:

`AllGather: 196608 Bytes -> Algo RING proto LL channel{Lo..Hi}={0..31}`

Nsight independently records
`ncclDevKernel_AllGather_RING_LL(...4096)` with grid 32, block 512, 96 registers per
thread, and 40 CUDA-graph-node launches among 60 whole-process AllGather launches.
The remaining launches belong to communicator/correctness/capture setup and must
not be attributed to the three benchmark samples. Small 4-byte `RING_LL` AllReduce
kernels are rank barriers from the harness, not the target collective.

For the twelve timed per-rank graph launches, the host-side `cudaGraphLaunch` API
duration is 24.697-37.479 us (mean 30.866 us). CUPTI event tracing heavily distorts
subsequent queue and kernel durations, so the report does not interpret those as
the unprofiled launch gap. The timed graph (graph id 2) contains the RING/LL
AllGather operation. A separate off-timing ordering graph (graph id 5) captures
producer copy, AllGather, and dependent consumer copy; three exact replay checks
validate that ordering. The ordering graph is correctness evidence, not part of
the timed samples, and it does not establish a live model consumer or useful
overlap.

## NVLink utilization and logical traffic

The `gb10x` counters are integer throughput-percent samples, not byte counters.
Correlating them only with the three timed AllGather kernel windows gives:

| GPU | samples per metric | active request/response mean range | max |
|---|---:|---:|---:|
| 0 | 115 | 0.0348%-0.0435% | 1% |
| 1 | 114 | 0.0263%-0.0351% | 1% |
| 2 | 113 | 0.0265% | 1% |
| 3 | 0 | not sampled; traced kernel windows were ~14 us | N/A |

User-data response fields stayed zero in these coarse samples. Request/user-data
fields are not summed because Nsight exposes separate protocol categories that can
overlap semantically. With no timed-window sample on GPU3 and severe profiler
perturbation, the defensible conclusion is only that the observed samples are
consistent with low NVLink utilization for this tiny TP4 M16 exchange.

From the ABI, not a hardware counter: each rank contributes 196,608 logical bytes
and receives a 786,432-byte output. A four-rank ring sends and receives
`3 * 196608 = 589824` logical send bytes per rank and the same receive volume
(2,359,296 aggregate logical send bytes). Protocol and physical wire bytes were
not measured and are not inferred.

## Optimization consequence

This trace supports a small-message launch/coordination diagnosis, not a bandwidth
kernel rewrite. The stock library already selects RING/LL across all 32 channels.
No device code was changed, so Nsight Compute, PTX, SASS, and ptxas evidence are not
applicable. A custom kernel remains unjustified unless a graph-safe candidate first
shows a reproducible paired gain. Grouped broadcasts were the final configuration
experiment, but no clean post-hardening session entered the oracle; stock PyNCCL is
the final enabled path.

The whole-process `collective-report.json` totals deliberately retain setup and
correctness traffic. Kernel-window values above came from the exported SQLite by
joining `CUPTI_ACTIVITY_KIND_KERNEL` AllGather graph nodes with `GPU_METRICS` and
`TARGET_INFO_GPU_METRICS` on device-specific `typeId` and kernel timestamps.
