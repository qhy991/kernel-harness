# GLM-5.2 production FlashMLA-KV decode report

**Disposition: NO REPLACEMENT.** The exact M16/M32 production-ABI path is
reachable and correct, and the isolated `max_num_splits=32` M16 combine
specialization has the intended code-generation effect, but it does not clear
the repeated 3% paired gate. No bucket is enabled. Stock FlashMLA remains active
for M16, M32, every other ABI, and every topology.

The authoritative scheduler-corrected campaign is
[`flex-20260723T160729Z`](campaigns/flex-20260723T160729Z/REPORT.md). It ran all
paired reference/candidate measurements and profiler collection in one
`with_flexible_gpu.sh` lease and observed zero candidate sessions above 1.03
(0/12). The earlier flat GPU-3 artifacts are retained unchanged as historical
evidence collected under the superseded scheduling instruction; they are not
used as the current acceptance authority.

Repeated containing-backend performance, complete GLM-5.2 server, and
TP8/DP8/EP8 acceptance cannot run on this host because the checkpoint directory
is empty and only four physical GPUs exist. Those gates remain external
requirements and are not weakened or relabelled; see
[external validation blockers](analysis/external_validation_blockers.md).

## Exact enable and fallback policy

| Bucket / case | Candidate enabled | Runtime policy |
|---|---:|---|
| `dsa_flashmla_kv_decode_m16` | no | stock `sgl_kernel::fwd_kvcache_mla` |
| `dsa_flashmla_kv_decode_m32` | no | stock `sgl_kernel::fwd_kvcache_mla` |
| other M, ABI, dtype, topology, or graph mode | no | stock FlashMLA |

There is no production dispatch change, host readback, timed adapter, extra
allocation, or copied/packed tensor. The attempted FlashMLA source is preserved
on an isolated branch and in [the exact source patch](analysis/flashmla_d18ff63.patch);
the installed package was never modified in place.

## Reachability and production ABI

The forced route is:

```text
--dsa-decode-backend flashmla_kv
  DeepseekSparseAttnBackend._forward_flashmla_kv
  -> sgl_kernel.flash_mla.flash_mla_with_kvcache
  -> torch.ops.sgl_kernel.fwd_kvcache_mla
  -> flash_fwd_splitkv_mla_fp8_sparse_kernel*
  -> flash_fwd_mla_combine_kernel*
```

`DeepseekSparseAttnBackend` is the current class name; the plan's
`DSAAttentionBackend` is a generic spelling. The static mapping and call-site
proof are in [source reachability](analysis/source-reachability.md). Nsight
Systems then observed both named kernels, in that order, on the same stream.

The two new serving-native workloads preserve the live representation:

| Field | M16 | M32 |
|---|---:|---:|
| local M (not DP-divided) | 16 | 32 |
| Q | `[M,1,64,576]` BF16 | same |
| paged KV | `[2049,64,1,656]` FP8 | same |
| sparse physical slots | `[M,1,2048]` int32 | same |
| cache length / allocation | 2048 / 8192 | same |
| page size / reserved page | 64 / page 0 | same |
| scheduler metadata | `[148,8]` | `[148,8]` |
| splits per request | 8 | 4 |
| softmax scale | 0.0625 | 0.0625 |
| output | `[M,1,64,512]` BF16 | same |

SGLang's cache quantizer constructs the 656-byte token representation: 512 FP8
latent bytes, 16 scale bytes, and 128 bytes for BF16 RoPE. Setup, quantization,
and scheduler construction are outside the measured region. The reference call
itself includes output/LSE allocation and both device kernels. Both buckets map
to SM100 head64 V32 at
`csrc/sm100/decode/head64/instantiations/v32.cu`.

## Dependency and build provenance

| Component | Identity |
|---|---|
| SGLang source | base `f93f8867b4bc124c9809c9110ec7361ed11b6b4a`; exact-ABI tests `d9fb72325cc73bb4b6f3ad96664513eb632e2459` |
| installed SGLang / sgl-kernel | `0.5.15` / `0.4.4` |
| installed FlashMLA extension | SHA-256 `d8d97150bd86381c73406603cb7d6b682767535e0526053f04e3acefadb13316`, build ID `134e5d51dded901e97231718aceec9ed96a0b398` |
| FlashMLA / CUTLASS base | `sgl-project/FlashMLA` `05e26647fe840b8baedae486c2d86d5ce4efeb7c` / `NVIDIA/cutlass` `147f5673d0c1c3dcf66f78d677fd647e4a020219` |
| stock rebuild control | FlashMLA `0657fffdfd1c981517647e043e4ef30ffdc1480f`; extension SHA-256 `b1afc29425c79cf00ad9687636474bfb7ffc098d81c5013ad1f3ade1966342f9` |
| candidate rebuild | FlashMLA `d18ff63a73dc6519432f59acb9f04365ce14bb10`; extension SHA-256 `9665dec00cb8caa4a8b5fc42bd40f9e8320d890e1a77705f0428630010539ccb` |
| PyTorch / CUDA compiler | `2.11.0+cu130` / nvcc `13.2.78` |
| target | NVIDIA B200, SM100, 148 SMs |

The final [stock](analysis/build_stock_pybind_tensor.json) and
[candidate](analysis/build_combine32_m16_tensor.json) manifests record the exact
build commands, environments, wheel hashes, source status, artifact paths, and
CUTLASS pin. The [final precommit dependency identity](analysis/dependency_identity_final_precommit.json)
captures import resolution and all three repository states. Candidate patch SHA-256 is
`b7e2b6740e6cf6f491f309e2957d637b716903e3ae09d877f190b3aa35e51268`.
Earlier `build_stock_control.json`, `build_stock_pybind.json`, and
`build_combine32_m16.json` are preserved ABI bring-up artifacts, not performance
evidence; only the final `*_tensor` control/candidate pair was timed.
The scheduler-corrected campaign pinned Kernel-Harness
`8c18448d9b76d2d648bec6da1e47587c59b26e73` and ran on physical GPU 1, PCI
`00000000:06:00.0`, UUID
`GPU-5b9be10b-5bfc-b658-9b31-f7ae8516dc54`, exposed as logical GPU 0 for one
uninterrupted lease. Four P0 snapshots record dynamic SM clocks
645/780/757/682 MHz and the fixed 3996 MHz memory clock; Nsight Compute reports
1.95 GHz during active replay. The wrapper line and its SHA-256, environment,
tool versions, repository status, and all source pins are in the campaign's
[device snapshot](campaigns/flex-20260723T160729Z/analysis/device_start.json).
The upstream dependency is `https://github.com/sgl-project/FlashMLA.git`; the
candidate is a separate local source checkout at the commit above.

## Correctness and execution semantics

- Exact production-symbol M16 and M32 outputs passed the harness tolerance and
  dtype/shape checks for installed stock, rebuilt control, and candidate.
- Interspersing 128 `-1` sparse indices/request matches the corresponding
  valid-only top-k result at max absolute difference `3.0517578125e-05` for
  both buckets.
- A nondefault current CUDA stream matches eager output exactly.
- Direct native CUDA Graph capture succeeded; three unchanged replays match
  eager exactly, and a mutated input changes output by `0.00229645` (M16) and
  `0.00231171` (M32) while matching mutated eager output.
- Every paired graph artifact confirms distinct output allocations, identical
  stock/candidate mutations, and successful numerical comparison.
- The SGLang exact fixture covers independent raw attention correctness and the
  metadata lifecycle for both buckets, FP8 KV, 64 heads, 8192-token prefixes
  with page-rounded decode capacity, affine/interleaved top-k, exact model
  scale, and required fused top-k. Its two targeted tests passed in 23.608
  seconds; see
  [fresh test output](campaigns/flex-20260723T160729Z/logs/sglang_exact_tests.txt).

The SGLang metadata-lifecycle fixture is intentionally not claimed as a
captured full-backend replay. The direct native workload supplies real CUDA
Graph capture of the core production symbol. A complete captured backend/server
replay remains part of the unavailable external promotion gate.

## Paired performance result

Each session alternates reference/candidate order for 100 pairs after 100
warmups on the same locked physical B200. The acceptance statistic is the median
of the 100 pair-wise speedups, independently for each session and bucket.

| Mode | Bucket | Candidate session speedups | Sessions at or above 1.03 |
|---|---:|---|---:|
| eager | M16 | `1.024761`, `1.014102`, `1.016634` | 0/3 |
| eager | M32 | `1.004522`, `1.021972`, `0.998688` | 0/3 |
| CUDA Graph | M16 | `0.989357`, `0.990007`, `0.989124` | 0/3 |
| CUDA Graph | M32 | `0.992105`, `0.985662`, `0.991270` | 0/3 |

The same-source compiler/build control also missed all sessions; its eager
session speedups were `1.007414/1.014577/1.013117` at M16 and
`1.007534/1.017690/0.995031` at M32, while graph sessions were
`0.991351/0.991282/0.993822` and `0.989923/0.989446/0.993995`.
All 24 fresh raw paired artifacts, independently recomputed medians, pair
distributions, graph checks, scheduler identity, and context-only stock
baselines are retained in the campaign
[paired summary](campaigns/flex-20260723T160729Z/analysis/paired_measurements_summary.md)
and [machine-readable summary](campaigns/flex-20260723T160729Z/analysis/paired_measurements_summary.json).
The earlier flat summary remains historical and is not overwritten.

## Nsight Systems: complete two-kernel region

| Build / bucket | main | combine | PDL overlap | chain span | launch gap |
|---|---:|---:|---:|---:|---:|
| stock M16 | 17.504 us | 12.480 us | 4.096 us | 25.888 us | 0 |
| stock M32 | 25.088 us | 9.824 us | 4.000 us | 30.912 us | 0 |
| candidate M16 | 17.536 us | 12.224 us | 4.224 us | 25.536 us | 0 |

The main launch is one 384-thread block on each of 148 SMs. Stock M16 combine
is 128 blocks (`16 x 1 x 8`) and M32 is 256 blocks (`32 x 1 x 8`). Main and
combine use stream 7 and overlap through programmatic dependent launch, so
there is no host launch gap to remove. The candidate's single-trace chain delta
is only 1.36% and is profiling context, not an acceptance result. Fresh parsed
chain records are
[stock M16](campaigns/flex-20260723T160729Z/analysis/nsys_chain_stock_m16.json),
[stock M32](campaigns/flex-20260723T160729Z/analysis/nsys_chain_stock_m32.json),
and [candidate M16](campaigns/flex-20260723T160729Z/analysis/nsys_chain_combine32_m16.json).

## Nsight Compute: six-dimension diagnosis

Nsight Compute 2026.1.1 used full metric collection plus version-compatible
`SourceCounters` collection on the pinned source rebuilds. Replay-instrumented
kernel durations below are not mixed with Nsight Systems or paired acceptance
latencies.

1. **Launch geometry and tail.** Main M16/M32 each use exactly 148 blocks, one
   wave across 148 SMs. M16 combine has only 128 blocks (`0.173` waves/SM), so
   20 SMs are idle and the tool estimates a 13.5% grid-underfill opportunity.
2. **Occupancy and resources.** Main uses 168 registers/thread and 232,656 bytes
   total per-block launch shared allocation, limiting it to one block and 18.75%
   theoretical occupancy (18.45% achieved M16, 18.56% M32). Combine uses 48
   registers and achieves only 11.29% stock / 12.28% candidate occupancy because
   the short underfilled grid and latency, not its theoretical 62.5% ceiling,
   dominate. No local load/store spills were measured.
3. **Compute utilization.** Main M16/M32 reach only 14.21%/20.07% SM throughput
   and 8.75%/13.49% tensor-pipe activity. Combine has no tensor activity and
   reaches 4.35% stock / 3.62% candidate SM throughput. Compute throughput is
   not the binding roof.
4. **Memory hierarchy and access efficiency.** Main reads 23.04 MB (M16) or
   46.01 MB (M32), reaching 1.014/1.534 TB/s and 13.24%/20.04% DRAM peak; L2 hit
   rate is only 14.80%/14.96%. Source rules report roughly 17.6-17.7 useful
   bytes per 32-byte global sector and multi-way shared-memory conflicts.
   Stock combine reads 16.818 MB at 1.564 TB/s with a 0.505% L2 hit rate; the
   candidate still reads 16.817 MB at 1.541 TB/s and 0.470% L2 hit. It removes
   no sparse gather.
5. **Warp issue and stalls.** Main M16 exposes only 0.209 eligible
   warps/scheduler/cycle and 18.30% issue activity; its 1,054 PC samples include
   373 barrier and 375 long-scoreboard stalls. M32 improves to 0.289 eligible
   warps and 23.93% issue. Stock combine has 271/390 long-scoreboard samples
   (69.5%); candidate has 298/401 (74.3%). The shared-memory reduction does not
   hide the dominant gather latency.
6. **Source and executable code.** M16 main hotspots map to cross-warp barriers,
   mbarrier waits, and sparse/shared address work. Both builds contain the same
   decoded device SASS and resource inventory. The candidate merely dispatches
   the existing BF16 combine bound-32 body instead of bound-160: static ptxas
   shared memory falls 5 KiB to 1 KiB (cuobjdump 6 KiB to 2 KiB), and the
   selected static body has 280 rather than 344 SASS records. Main code is
   unchanged.

The fresh complete metrics are in
`campaigns/flex-20260723T160729Z/analysis/metrics_all_*.json`; compact values,
source views, rule output, and stall exports are alongside them.
The full [ptxas/SASS audit](analysis/ptxas_sass_inventory.md) distinguishes
decoded executable identity from differing fatbin metadata. PM sampling had
only 7-27 active samples for these very short kernels and lacked throughput
series, so it is preserved as qualitative context only.

## Attempt and decision

The candidate added an exact, fail-closed SM100 M16 predicate and passed
`max_num_splits=32` to the existing combine dispatch while leaving the main
kernel, scheduler, metadata, temporary buffers, stream, PDL behavior, M32, and
all unsupported cases unchanged. This was the lowest-risk source hypothesis:
M16 uses eight actual splits/request, yet stock dispatch selected a bound of 160
from `num_sm_parts=148`.

The source change delivered exactly the predicted resource/code-size reduction,
but the combine remains a 128-block, 16.8-MB latency-dominated gather and every
fresh graph session regressed. The experiment is rejected and rolled back
operationally by enabling nothing. The detailed hypothesis, distributions,
profiler delta, risk, and rollback point are preserved in the
[experiment ledger](analysis/experiment_ledger.md).

## Validation and artifact map

The final source state passed:

- `python3 serving_native/selftest.py` (41 workloads);
- the two exact SGLang FlashMLA-KV tests (23.608 s);
- the fresh paired-summary raw-artifact/graph/scheduler consistency check;
- `python3 testbench/bin/verify_harness.py` (24 task structural checks,
  knowledge lint/index/distill checks, sync and diff checks); its historical
  missing `runs/index.jsonl` pointer remains advisory and unrelated;
- Python compilation and `git diff --check` on authored source/docs (raw
  profiler exports retain the tools' column-padding whitespace);
- the B200 environment/import check under the required GPU lock.

Key artifacts:

- workload and callable mapping: [source reachability](analysis/source-reachability.md);
- fresh isolated import/GPU check:
  [environment output](campaigns/flex-20260723T160729Z/logs/check_env.txt);
- exact runtime semantics:
  [M16](campaigns/flex-20260723T160729Z/analysis/runtime_stock_m16.json) and
  [M32](campaigns/flex-20260723T160729Z/analysis/runtime_stock_m32.json);
- all current acceptance timing:
  [paired summary](campaigns/flex-20260723T160729Z/analysis/paired_measurements_summary.md);
- fresh profiler binaries:
  [`campaign reports`](campaigns/flex-20260723T160729Z/reports/);
- source/build delta: [patch](analysis/flashmla_d18ff63.patch) and final manifests;
- raw final compiler logs: [stock](reports/build_stock_pybind_tensor.log) and
  [candidate](reports/build_combine32_m16_tensor.log);
- static device-code audit: [ptxas/SASS inventory](analysis/ptxas_sass_inventory.md);
- unavailable gates: [external validation blockers](analysis/external_validation_blockers.md).

No `testbench/tasks/glm52` oracle or generated task file was changed, no result
is presented as an eight-rank or complete-model result, and no remote state was
modified.
