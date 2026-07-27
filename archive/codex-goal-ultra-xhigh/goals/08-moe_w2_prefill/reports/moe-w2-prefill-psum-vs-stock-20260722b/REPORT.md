# MoE W2 prefill: stock contiguous layout versus PSUM

## Scope and verdict

This report compares the stock row-wise contiguous DeepGEMM specialization
(`GemmType=1`) with the library-native PSUM specialization (`GemmType=5`) for
the single-B200 `moe_w2_grouped_prefill_m4096` production-ABI replay. The
workload has 32 local experts, 32982 valid rows in 35200 aligned rows,
K=2048, N=6144, FP8 E4M3 tensors, packed `int32` UE8M0 scales, and BF16 output.
It is a provisional replay of one EP8 rank's router contract, not a live EP8
capture.

PSUM is a real kernel-level improvement. Nsight Compute measures
331.296 -> 313.312 us (1.057400x), and the five matching Nsight Systems
launches have 332.159 -> 311.967 us kernel medians (1.064725x). The locked
10-repeat component gate independently measures a 1.064482x paired median.
The much larger 1.257373x CUDA-event ratio inside the one-repeat profiled
runner is rejected as a performance result: Nsight Systems shows that PSUM
host-side launch preparation overlaps the preceding untimed output-poison fill
before the start marker completes, whereas stock launch preparation occurs
after its start marker.

This is component-development evidence only. It neither includes DeepEP,
W13, SwiGLU+quant, combine, nor a live TP8/DP8/EP8 server.

## Collection contract

- GPU: physical B200 0, UUID
  `GPU-30b619de-87f2-1862-0d07-a595da8fe417`, one visible logical GPU.
- Code under test: Kernel-Harness `83e0352b7b9b0f0e5899c0d23e44bfe86ba917fd`,
  SGLang `07802235f03e14cfca9d709bc91802d36ca1d9c6`.
- Stack: Torch 2.11.0+cu130, CUDA 13.0, Nsight Compute 2026.1.1.0,
  Nsight Systems 2025.6.3.541.
- Runtime controls: PDL true, 148 SMs, tensor-core-util 100, eager stream 0.
- DeepGEMM JIT: fresh cache with `DG_JIT_WITH_LINEINFO=1`; the preserved NVCC
  command uses `sm_100f`, `-O3`, and `-lineinfo`.
- NCU full reports: stock 47 replay passes and PSUM 46; source reports 12
  passes each. They are separate single observations, not paired statistics.
- The analysis initially generated in `...20260722a` failed its strict
  optional base-utilization PM assertion. This corrected `...20260722b`
  analysis records that limitation explicitly and validates successfully.

## 1. Latency and speed of light

| Metric | Stock | PSUM | Delta |
|---|---:|---:|---:|
| NCU kernel duration | 331.296 us | 313.312 us | -17.984 us (-5.428%, 1.057400x) |
| Nsys kernel median, 5 launches | 332.159 us | 311.967 us | 1.064725x |
| Nsys kernel mean, 5 launches | 332.2356 us | 312.2548 us | 1.063989x |
| Nsys single timed launch | 330.974 us | 314.143 us | 1.053578x |
| SM throughput | 75.3585% | 77.7957% | +2.4372 pp |
| tensor-pipe active, elapsed | 70.2915% | 71.4089% | +1.1173 pp |
| compute-memory throughput | 64.0568% | 67.8852% | +3.8284 pp |
| DRAM active cycles | 40.3918% | 35.4432% | -4.9485 pp |
| issue active | 13.3661% | 16.7003% | +3.3342 pp |
| eligible warps/cycle | 0.147006 | 0.183957 | +25.136% |

NCU classifies stock as more compute-utilized than memory-utilized and PSUM as
balanced. The kernel is not HBM-bandwidth-bound: tensor/SM activity is near
75-78%, while DRAM active cycles are 35-40%.

The Blackwell path uses UTCQMMA. The curated HMMA metric being zero is not
evidence that tensor cores are unused.

## 2. Memory, TMA, TMEM, and cache

| Metric | Stock | PSUM | Relative delta |
|---|---:|---:|---:|
| DRAM read bytes | 624018944 | 476493568 | -23.641% |
| DRAM write bytes | 402508032 | 375352576 | -6.747% |
| total DRAM bytes | 1026526976 | 851846144 | -17.017% |
| global-load SASS instructions | 264000 | 31006 | -88.255% |
| global-load L1 lookup misses | 13200 | 592 | -95.515% |
| L1 sector hit rate | 94.9734% | 97.8571% | +2.884 pp |
| L2 sector hit rate | 71.0397% | 75.7977% | +4.758 pp |
| input TMA bytes | 5244518400 | 5244518400 | unchanged |
| TMA load instructions | 528000 | 528000 | unchanged |
| output TMA bytes | 432537600 | 408551424 | -5.545% |
| TMA store instructions | 211200 | 199488 | -5.545% |
| TMEM load instructions | 1689600 | 1595904 | -5.545% |
| UTCQMMA work | 885837004800 | 836713316352 | -5.545% |

The counters show unchanged input-TMA bytes and instruction counts; PSUM does
not reduce that counted input tile feeding. The matched 5.545% reductions in
UTCQMMA, TMEM loads, and output TMA
stores support reduced tail-validity/epilogue work inside the same 128-row tile
count. The 88.3% reduction in scalar/global loads is consistent with replacing
row-wise group lookup by one cumulative endpoint per expert. These are
correlations across two specializations; they do not by themselves prove a
single exclusive cause.

PSUM's cost is more shared-memory conflict work: shared-load conflicts rise
130340 -> 231629 and shared-store conflicts rise 4326307 -> 4584853. NCU's
coalescing and conflict rule estimates overlap with other opportunities and
must not be added into a total projected speedup.

## 3. Occupancy and resources

Both kernels launch 148 blocks of 256 threads: one CTA and eight warps per B200
SM, one wave total. Both allocate 214828 bytes of shared memory per block
(213804 dynamic plus 1024 static), which limits residency to one block/SM.

| Resource | Stock | PSUM |
|---|---:|---:|
| registers/thread | 38 | 50 |
| achieved active warps | 12.4807% | 12.5373% |
| register occupancy limit | 6 blocks | 4 blocks |
| shared-memory occupancy limit | 1 block | 1 block |
| cubin stack/local bytes | 0 / 0 | 0 / 0 |
| local-load/store instructions | 0 / 0 | 0 / 0 |

The register increase causes no residency loss and no spill. Improving
residency would require a large shared-memory reduction, not register tuning.

| Per-SM active cycles | Average | Maximum | Minimum |
|---|---:|---:|---:|
| Stock | 509525.736 | 513397 (+0.760%) | 505094 (-0.870%) |
| PSUM | 471274.845 | 477005 (+1.216%) | 459918 (-2.410%) |

The small spread gives no evidence of a few-SM straggler dominating this
isolated kernel. Its input-work distribution is also recorded: the 32 fixture
expert counts have min/max 975/1104, mean 1030.6875, median 1034, and population
standard deviation 29.2804; alignment produces 13 slabs of 1024 and 19 of 1152
rows. This is deterministic replay input, not a checkpoint-derived expert
distribution.

## 4. Scheduler stalls and synchronization

| Average other stalled warps per issue-active cycle | Stock | PSUM |
|---|---:|---:|
| long scoreboard | 6.311964 | 4.069615 |
| barrier | 4.669483 | 3.892127 |
| wait | 1.319395 | 1.376340 |
| short scoreboard | 0.636799 | 0.583856 |
| MIO throttle | 0.015264 | 0.087352 |
| branch resolving | 0.477490 | 0.543570 |
| no instruction | 0.296306 | 0.164026 |

Average warp latency per issued instruction improves 14.9461 -> 11.9979
cycles. Source PC sampling likewise shifts away from long scoreboard: its
share falls from 41.176% to 33.711%. Residual barrier/wait work, branch work,
and MIO throttling become proportionally more visible.

The largest mapped stock hotspot is `cutlass/arch/barrier.h:424`, an
`mbarrier.try_wait.parity` loop: 5942 total samples/5511 long-scoreboard versus
3983/3602 under PSUM. The stock-only top hotspot at
`cute/numeric/math.hpp:51` (433 total/364 long-scoreboard) is consistent with
row-wise group lookup. PSUM adds more activity around `bar.sync` and some TMA
wait sites, so the result is a trade rather than removal of synchronization.

## 5. Instruction and source evidence

The lineinfo reports resolve the selected stock and PSUM symbols to distinct
DeepGEMM template specializations (`GemmType=1` and `GemmType=5`). Source/SASS
mapping includes `UTCQMMA.2CTA`, `UTMALDG`, `UTMASTG`, `LDTM`,
`SYNCS.PHASECHK`, and `UCGABAR_WAIT` instructions at real positive source
lines. The PSUM kernel executes more total warp instructions
(39982759 -> 46338261, +15.896%) and branches
(2917832 -> 4244015, +45.451%), but its shorter dependency waits and reduced
tail/output work more than offset that control overhead.

`cuobjdump --dump-resource-usage` confirms `REG:38` versus `REG:50`, with
`STACK:0`, `LOCAL:0`, and `SHARED:1024` for both selected cubins. Exact source
lines differ across specializations; opcode and aggregate-counter agreement is
used before interpreting line-level movement.

## NCU rule-engine estimates

NCU's rule estimates are local, overlapping opportunities; they must not be
summed or treated as predicted end-to-end speedup.

| Rule (reported estimate) | Stock | PSUM |
|---|---:|---:|
| L2 compression | 22.96% | 18.54% |
| global-load coalescing | 46.61% | 50.13% |
| shared-load conflicts | 12.75% | 20.72% |
| shared-store conflicts | 28.92% | 32.85% |
| scheduler issue (`Est. Local Speedup`) | 24.64% | 22.20% |
| long scoreboard | 24.64% | 22.20% |
| barrier | 24.64% | 22.20% |
| divergence/predication | 28.91% | 23.86% |
| occupancy | 24.64% | 22.20% |

The coalescing estimate is easy to overread: PSUM's remaining scalar loads use
only 4/32 bytes per sector, but absolute global-load instructions are already
88.3% lower, the load L1 hit rate is 98.09%, and the request rate is tiny. The
shared-conflict, branch, and MIO estimates describe the cost PSUM trades for
lower dependency waiting; they are not evidence that reverting to the stock
layout would help.

## 6. PM-sampling tail

WarpStates PM instance series are available. Long-scoreboard active samples
fall 328 -> 309 and their mean falls 1317.42 -> 1037.13; the final-quarter to
first-quarter ratio falls 0.654 -> 0.627. Wait rises 273.06 -> 351.04 mean,
while MIO rises 3.165 -> 22.628. Dominant stall series decay toward completion
rather than showing a worse late tail under PSUM.

The four requested base-utilization PM series have no instances through this
NCU Python API for either report. `analysis/pm_series_status.txt` records this
explicitly. Therefore the PM evidence supports only a WarpStates-tail statement,
not a time-resolved SM-, tensor-, L1-, or DRAM-utilization claim. Aggregate
utilization remains available from the full NCU reports.

## Nsight Systems timing caveat

The trace contains exactly five stock and five PSUM DeepGEMM launches:
correctness, three warmups, and one timed pair. The profiled runner JSON reports
405.216008 versus 322.272003 us (1.257373x), but its timed-kernel durations are
330.974 versus 314.143 us. CUPTI's completion-marker interval decomposes as:

| Interval | Stock | PSUM |
|---|---:|---:|
| start marker to kernel | 61.312 us | 2.464 us |
| kernel | 330.974 us | 314.143 us |
| kernel to end marker | 6.848 us | 5.632 us |
| total marker span | 399.134 us | 322.239 us |

The PSUM host launch API begins before its start marker completes because the
preceding untimed poison-fill is still in flight; the device kernel remains
serialized after the marker. Stock host launch preparation occurs after its
marker. The unprofiled cache-prime result, 337.983996 versus 316.383988 us
(1.068271x),
agrees with the component and kernel ratios. The 1.257373x profiled event ratio
is profiler/order/queue-state contamination and is not cited as a gate result.
No other kernel, memcpy, or memset falls inside either timed event window.

## Raw artifacts and hashes

| Artifact | SHA256 |
|---|---|
| `../moe-w2-prefill-paired-nsys-20260722a/reports/paired.nsys-rep` | `161f331fc268315fa36434c5f8a5cded5befabd1acf7fe21a04df2b014567a0c` |
| `../moe-w2-prefill-paired-nsys-20260722a/reports/paired.sqlite` | `cfe74bba11a463dea3c11d155508f8c79dc896e15915d904c9e76cb7fa4c7004` |
| `../moe-w2-prefill-stock-ncu-20260722a/reports/stock_full.ncu-rep` | `b24c336323e0fefb53750e08aac581194b2eb00514d7ae8721cbdd3c820ae639` |
| `../moe-w2-prefill-stock-ncu-20260722a/reports/stock_source.ncu-rep` | `75b9a1cf62a5073aa254a37259d2b317bb42f4bbb72665c373642b50d3796c87` |
| `../moe-w2-prefill-psum-ncu-20260722a/reports/psum_full.ncu-rep` | `8e2c271c1830d9d5d3636bd26434f64bedfdb8ca07358d42daccea5c17258932` |
| `../moe-w2-prefill-psum-ncu-20260722a/reports/psum_source.ncu-rep` | `5a906be089637442654518db2f567d4064bab9783d8c370462550f8b03c91fbd` |

The machine-readable metric archive, curated comparison, source hotspots,
resource dump, PM plots/status, collection logs, runner JSONs, and helper hashes
are preserved next to this report and in the three collection directories.

## Ranked next directions

1. Retain the real `ep_scatter` endpoints and test the checkpoint-derived EP8
   region first. The component gain is already above noise; production adapter,
   output-gap, stream, graph, and overlap costs are now the binding evidence gap.
2. If the region preserves the gain, reduce residual PSUM synchronization around
   `mbarrier.try_wait`, `cluster_wait`, and `bar.sync` while holding the exact
   UTCQMMA/TMA work and valid-row output fixed. Barrier and long scoreboard
   remain the two largest stall classes.
3. Target PSUM's added branch/shared/MIO overhead without restoring row-wise
   lookup. Branch instructions rise 45.45%, shared conflicts worsen, and MIO
   stalls rise; any change must beat the same paired component oracle and retain
   source-mapped correctness.
4. Treat higher occupancy as lower priority. Shared memory, not registers,
   fixes one CTA/SM; obtaining a second CTA would require a major pipeline
   redesign and may sacrifice the current tensor/TMA overlap.

## Reproduction commands

The collection and analysis scripts contain the complete Nsys/NCU arguments,
kernel-name filters, JIT environment, and source folders. From a fresh isolated
goal worktree where the timestamped output directories do not yet exist:

```bash
cd /home/qinhaiyan/glm52-goal-runs/08-moe_w2_prefill/kernel-harness
/home/qinhaiyan/glm52-goal-runs/with_all_gpus_lock.sh \
  bash serving_native/evidence/08_moe_w2_prefill/run_profile_collection.sh
env CUDA_VISIBLE_DEVICES= \
  bash serving_native/evidence/08_moe_w2_prefill/analyze_profiles.sh
```

Both scripts deliberately refuse to overwrite immutable evidence. For a new
collection, use a new timestamp suffix in an isolated copy. Existing raw
reports can be inspected without launching GPU work:

```bash
ncu --import profile/moe-w2-prefill-stock-ncu-20260722a/reports/stock_full.ncu-rep \
  --page details
ncu --import profile/moe-w2-prefill-psum-ncu-20260722a/reports/psum_full.ncu-rep \
  --page details
nsys stats --report=cuda_gpu_kern_gb_sum:mangled \
  profile/moe-w2-prefill-paired-nsys-20260722a/reports/paired.nsys-rep
```

## Final diagnosis

PSUM improves this exact component by reducing row-layout global loads and
tail-validity/output work while keeping the same counted input-TMA traffic, 128-row
tile count, one-wave grid, and one-CTA-per-SM occupancy. It trades higher
register/control/shared-conflict cost for substantially less long-scoreboard
waiting. The remaining kernel is compute/latency/synchronization limited.

No production replacement follows from this diagnosis: endpoint retention in
the real scatter path, full-output semantics, overlap/graph behavior, the
eight-rank MoE region, and SGLang end-to-end performance remain unvalidated.
