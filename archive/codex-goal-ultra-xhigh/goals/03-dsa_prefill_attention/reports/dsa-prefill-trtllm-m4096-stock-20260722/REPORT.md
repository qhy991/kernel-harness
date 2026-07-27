# GLM-5.2 DSA prefill: stock TRT-LLM kernel profile

## Result

The reached rank-local prefill leaf is one persistent Blackwell FMHA kernel. It
accounts for 96.0% of GPU kernel time and has a 0.832327 ms Nsight Systems mean
over 25 launches. The kernel is not HBM-bandwidth-bound: Nsight Compute reports
69.32% SM throughput, 62.61% tensor-pipe activity, only 6.66% aggregate DRAM
throughput, and a 94.46% L2 hit rate. Its clearest limiter is dependency latency
inside the sparse gather/producer-consumer pipeline: 60.64% of source-counter
samples are long-scoreboard stalls, while schedulers have no eligible warp in
59.17% of cycles.

This is a diagnosis, not a replacement result. The selected kernel is a vendor
AOT FlashInfer/TRT-LLM binary, so the export contains real SASS and per-PC
sampling but no CUDA/C++ source-line mapping.

## Profiled path

- Workload: `dsa_trtllm_prefill_m4096_ctx32768`.
- Stock call: `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` with
  `backend="trtllm-gen"`; `enable_pdl` is not overridden.
- Rank-local ABI: FP8 query/KV, M=4096, 64 query heads, QK width 576, value width
  512, page size 64, context 32768, and sparse top-k 2048.
- Launch: grid 4096, block 512, 128 registers/thread, 220672 bytes shared
  memory/block, and 27.676 waves/SM on 148 SMs.
- Kernel:
  `fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvDenseStaticTokenSparseP1VarSeqQ64Kv128PersistentKeepsAbForGen`.

The harness and launcher pin the isolated SGLang checkout and physical GPU 3,
exposed as logical GPU 0. The measured stream and full ABI are emitted by
[`harness/profile_dsa_prefill.py`](harness/profile_dsa_prefill.py); collection
artifacts are retained under `reports/`.

## Nsight Systems timing

The whole-process report contains 25 launches of the target kernel. The target
dominates the trace, so setup/helper kernels do not explain the leaf latency.

| Metric | Value |
| --- | ---: |
| Target share of GPU kernel time | 96.0% |
| Launches | 25 |
| Mean | 832326.5 ns |
| Median | 831647.0 ns |
| Minimum | 830431 ns |
| Maximum | 838047 ns |
| Standard deviation | 1800.2 ns |

The 0.22% standard deviation and narrow 7.6 us min-to-max range show a stable
kernel-duration plateau. The PM-sampling timeline likewise shows long-scoreboard
pressure throughout the active interval, not only at startup or tail.

## Six-dimension diagnosis

### 1. Compute utilization

- SM throughput is 69.3217%; tensor-pipe activity is 62.6120% of elapsed cycles.
- TMEM is the highest-utilized pipeline at 69.3%.
- Issue slots are busy 40.06%; only 0.507 warps/scheduler are eligible on average.

Tensor work is substantial, but issue starvation prevents the kernel from
turning its active warps into sustained issue throughput.

### 2. Memory behavior

- Maximum memory-hierarchy throughput is 56.2842%, driven by L2 rather than HBM.
- DRAM read/write utilization is 3.1630%/3.4956%; aggregate DRAM throughput in
  the Nsight Compute summary is 6.66%.
- DRAM reads are 203534336 bytes, or 242.663 GB/s during the profiled launch.
- L1 and L2 sector hit rates are 63.1884% and 94.4579%, respectively.

Nsight flags 44% excessive global-load sectors and 4.6-way shared-load bank
conflicts. Those are useful optimization targets, but the global pattern is also
consistent with the workload's deliberately sparse gather. Nsight's local
speedup estimates are heuristic and must not be added together.

### 3. Latency and stalls

- Long-scoreboard: 5.8863 cycles per issued instruction and 33094/54575 sampled
  stalls (60.6395%).
- Short-scoreboard: 1.0310 cycles/issue and 10.7467% of samples.
- Wait: 0.9852 cycles/issue and 9.9240% of samples.
- No eligible warp: 59.17% of scheduler cycles.

The hottest PCs are dependency-polling branches and `NANOSLEEP.SYNCS` sites.
Together with the visible `UTMALDG.2D.GATHER4`, `SYNCS`, and tensor-core SASS,
this identifies the dominant opportunity as producer/consumer dependency latency
around sparse loads, rather than raw DRAM bandwidth.

### 4. Occupancy and resource pressure

- Theoretical occupancy is 25.0%; achieved occupancy is 24.8159%.
- Registers and shared memory each limit residency to one block/SM.
- Local instructions include 328272 loads and 23488 stores in the collected
  metric set.

The kernel reaches its resource-limited occupancy, so launch tuning alone cannot
recover the missing issue slots. Reducing live state or shared-memory footprint
could expose more latency, but only if it does not damage the tensor/gather
pipeline.

### 5. Launch and timeline

- One persistent FMHA launch is the only material leaf; ancillary kernels are 4%
  of whole-process GPU kernel time.
- 4096 blocks over 148 SMs supply 27.676 waves/SM, so there is ample grid-level
  work and no evidence of a short-grid tail.
- The flat PM-sampling plateau and stable multi-launch timing reject launch
  overhead and end-of-grid imbalance as primary causes.

### 6. Instruction and SASS evidence

The source-counter export contains 4888 actual SASS rows. Source-line mapping is
unavailable because this is a precompiled vendor kernel, but instruction-level
mapping is present. In the sampled-PC inventory, notable static opcodes include
124 `UTMALDG`, 26 `UTCQMMA`, 77 `SYNCS`, 20 `LDTM`, 14 `STTM`, 12 `LDL`, and 89
branches. Hot PCs are preserved in
[`analysis/sass-hotspots-stock-m4096.txt`](analysis/sass-hotspots-stock-m4096.txt).

## Optimization implication

The high-confidence next target is the vendor kernel's sparse-gather scheduling:
reduce dependency distance/polling and shared/local-memory friction while
preserving tensor/TMEM overlap. SGLang's Python call site cannot directly change
that AOT SASS. A launch-policy experiment (`enable_pdl=False`) was therefore used
as the narrowest source-level seam; its paired profile is analyzed separately and
does not clear the 3% threshold.

## Artifact index and caveats

- Full metric replay: [`reports/full-stock-m4096.ncu-rep`](reports/full-stock-m4096.ncu-rep).
- Source/SASS replay: [`reports/source-stock-m4096.ncu-rep`](reports/source-stock-m4096.ncu-rep).
- Nsight Systems trace: [`reports/nsys-stock-m4096-fullapp.nsys-rep`](reports/nsys-stock-m4096-fullapp.nsys-rep).
- Key metrics: [`analysis/metrics_key_stock-m4096.json`](analysis/metrics_key_stock-m4096.json).
- Nsight Compute rule text: [`analysis/details-stock-m4096.txt`](analysis/details-stock-m4096.txt).
- PM-sampling timeline: [`analysis/pm_timeline_plots.txt`](analysis/pm_timeline_plots.txt).
- Full SASS/source-counter export: [`analysis/source-stock-m4096.txt`](analysis/source-stock-m4096.txt).

Nsight Compute warns that Work ID/Cluster Launch Control was enabled, so metrics
derived from launched cluster/block/warp/thread counts may be affected. The
profiler serializes/replays work and is not itself a benchmark; timing conclusions
use the multi-launch Nsight Systems distribution. The Systems trace is a
whole-application capture because the attempted NVTX-triggered collection yielded
no report; the target kernel is identified explicitly by its exact symbol.
