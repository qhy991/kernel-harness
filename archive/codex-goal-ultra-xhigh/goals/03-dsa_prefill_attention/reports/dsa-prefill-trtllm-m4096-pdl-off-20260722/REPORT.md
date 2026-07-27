# GLM-5.2 DSA prefill: `enable_pdl=False` profile

## Result

Disabling programmatic dependent launch does not produce a material improvement.
Nsight Systems measures 0.832172 ms mean over 25 target launches, only 0.0186%
faster than stock, while the single-launch Nsight Compute replay is 0.2938%
slower. Both changes are far below the 3% optimization threshold and ordinary
run-to-run noise. After relocation addresses and one Nsight fixed-column overflow
are normalized, all 4888 exported SASS instruction rows are identical to stock.

The trial is therefore rejected as a production optimization. Its bottleneck is
the same as stock: sparse-gather dependency latency inside the vendor AOT
producer/consumer pipeline.

## Profiled path

- Workload: `dsa_trtllm_prefill_m4096_ctx32768`.
- Trial call: the exact stock FlashInfer/TRT-LLM API and arguments, with only
  `enable_pdl=False` added.
- Rank-local ABI: FP8 query/KV, M=4096, 64 query heads, QK width 576, value width
  512, page size 64, context 32768, and sparse top-k 2048.
- Launch: grid 4096, block 512, 128 registers/thread, 220672 bytes shared
  memory/block, and 27.676 waves/SM on 148 SMs.
- Kernel:
  `fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvDenseStaticTokenSparseP1VarSeqQ64Kv128PersistentKeepsAbForGen`.

The measured candidate is frozen beside the harness in
[`harness/dsa_prefill_pdl_off_measured.py`](harness/dsa_prefill_pdl_off_measured.py).
The launcher pins the isolated SGLang checkout and physical GPU 3, exposed as
logical GPU 0.

## Nsight Systems timing

| Metric | `enable_pdl=False` | Stock | Trial delta |
| --- | ---: | ---: | ---: |
| Target share of GPU kernel time | 96.0% | 96.0% | 0.0 points |
| Launches | 25 | 25 | 0 |
| Mean | 832171.7 ns | 832326.5 ns | -154.8 ns (-0.0186%) |
| Median | 831775.0 ns | 831647.0 ns | +128.0 ns (+0.0154%) |
| Minimum | 830079 ns | 830431 ns | -352 ns |
| Maximum | 837727 ns | 838047 ns | -320 ns |
| Standard deviation | 1499.2 ns | 1800.2 ns | -301.0 ns |

The mean and median move in opposite directions by tiny amounts. This trace does
not establish a PDL effect.

## Six-dimension diagnosis

### 1. Compute utilization

- SM throughput is 69.3439%, versus 69.3217% stock.
- Tensor-pipe activity is 62.6328% of elapsed cycles, versus 62.6120% stock.
- Issue slots are busy 40.07%; TMEM remains the highest-utilized pipeline at
  69.3%.

These changes are hundredths of a percentage point and do not indicate a new
compute path.

### 2. Memory behavior

- Maximum memory-hierarchy throughput is 56.1378%, versus 56.2842% stock.
- DRAM read/write utilization is 3.1637%/3.4785%; aggregate DRAM throughput is
  6.64%.
- DRAM reads are 204176128 bytes, or 242.715 GB/s during the profiled launch.
- L1/L2 sector hit rates are 63.1812%/94.4745%.

The hierarchy behavior is indistinguishable from stock. Nsight again flags 44%
excessive global-load sectors and shared-load bank conflicts; the sparse gather
explains why a coalescing warning alone is not proof of a fixable HBM bottleneck.

### 3. Latency and stalls

- Long-scoreboard: 5.8907 cycles/issue and 33047/54565 source-counter samples
  (60.5645%).
- Short-scoreboard: 1.0298 cycles/issue and 10.6094% of samples.
- Wait: 0.9849 cycles/issue and 10.1549% of samples.
- No eligible warp: 59.18% of scheduler cycles.

The same dependency-polling branches and `NANOSLEEP.SYNCS` sites remain hot. The
PM timeline shows the same long-scoreboard plateau across the active interval.

### 4. Occupancy and resource pressure

- Theoretical occupancy is 25.0%; achieved occupancy is 24.8341%.
- Registers/thread (128), shared memory/block (220672 bytes), local loads
  (328272), and local stores (23488) are exactly the same as stock.
- Registers and shared memory still cap residency at one block/SM.

PDL selection does not alter device resource use or expose additional latency.

### 5. Launch and timeline

- The identical target symbol still accounts for 96.0% of whole-process GPU
  kernel time.
- Grid, block, resource limits, and waves/SM are unchanged.
- The 25-launch distribution is narrow and overlaps stock completely; there is
  no launch/tail signature attributable to the flag.

### 6. Instruction and SASS evidence

Both source-counter exports contain 4888 instruction rows. The offline comparison
removes process-specific absolute relocation addresses and the numeric sample
field attached to one overflowing `UTCQMMA ... !UPT` display column. The resulting
instruction sequences have the same SHA-256 and compare exactly equal. The export
has real per-PC SASS but no CUDA/C++ source-line mapping because the kernel is a
vendor AOT binary.

## Decision

Do not promote `enable_pdl=False`. The target's Nsight Systems mean improves by
0.0186%, the Nsight Compute replay regresses by 0.2938%, no device code changes,
and the stall/resource signatures are the same. Stock launch behavior must remain
the fallback and production default.

## Artifact index and caveats

- Frozen measured candidate: [`harness/dsa_prefill_pdl_off_measured.py`](harness/dsa_prefill_pdl_off_measured.py).
- Full metric replay: [`reports/full-pdl-off-m4096.ncu-rep`](reports/full-pdl-off-m4096.ncu-rep).
- Source/SASS replay: [`reports/source-pdl-off-m4096.ncu-rep`](reports/source-pdl-off-m4096.ncu-rep).
- Nsight Systems trace: [`reports/nsys-pdl-off-m4096-fullapp.nsys-rep`](reports/nsys-pdl-off-m4096-fullapp.nsys-rep).
- Key metrics: [`analysis/metrics_key_pdl-off-m4096.json`](analysis/metrics_key_pdl-off-m4096.json).
- Nsight Compute rule text: [`analysis/details-pdl-off-m4096.txt`](analysis/details-pdl-off-m4096.txt).
- PM-sampling timeline: [`analysis/pm_timeline_plots.txt`](analysis/pm_timeline_plots.txt).
- Full SASS/source-counter export: [`analysis/source-pdl-off-m4096.txt`](analysis/source-pdl-off-m4096.txt).
- Cross-profile comparison: [`../dsa-prefill-trtllm-m4096-pdl-compare-20260722/REPORT.md`](../dsa-prefill-trtllm-m4096-pdl-compare-20260722/REPORT.md).

Nsight Compute warns that Work ID/Cluster Launch Control was enabled, so metrics
derived from launched cluster/block/warp/thread counts may be affected. Its replay
is diagnostic, not benchmark timing. The Systems trace is a whole-application
capture; the exact kernel symbol separates the target from helper kernels.
