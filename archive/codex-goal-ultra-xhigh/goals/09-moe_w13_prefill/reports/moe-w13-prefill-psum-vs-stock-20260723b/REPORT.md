# MoE W13 prefill: stock contiguous layout versus PSUM

## Scope and verdict

This report compares the stock row-wise contiguous DeepGEMM specialization
(`GemmType=1`) with the library-native PSUM specialization (`GemmType=5`) for
the single-B200 `moe_w13_grouped_prefill_m4096` production-ABI replay. The
workload has 32 local experts, 32,982 valid rows in 35,200 aligned rows,
K6144, fused N4096, FP8 E4M3 tensors, packed `int32` UE8M0 scales, and BF16
output. It is a provisional replay of one EP8 rank's router contract, not a
live checkpoint-derived EP8 capture.

PSUM is a real component-level kernel improvement. Nsight Compute measures
629.824 -> 599.328 us (1.050884x), and the five matching Nsight Systems
launches have 633.117 -> 601.277 us kernel medians (1.052954x). The locked
10-repeat component gate independently measures a 1.052746x paired median.

The much larger 1.173107x CUDA-event ratio inside the one-repeat profiled
runner is rejected as a performance result. Nsight Systems shows that PSUM
host-side descriptor/launch preparation overlaps the preceding untimed output
poison fill before the start marker completes, whereas stock preparation
occurs after its start marker.

This is component-development evidence only. It contains neither live DeepEP
traffic nor SwiGLU+quant, W2, combine, CUDA Graph replay, or a TP8/DP8/EP8
server. The candidate remains disabled in production.

## Collection contract

- GPU: physical B200 0, UUID
  `GPU-30b619de-87f2-1862-0d07-a595da8fe417`, one visible logical GPU.
- Code under test: Kernel-Harness
  `b7808f3b80aa3c8ccef8eb8ae92f904103a72252`, SGLang
  `7f0365a457379b8f6d435f55fcd6d16b817f733a`; both were clean before
  collection.
- Stack: Torch 2.11.0+cu130, CUDA 13.0, Nsight Compute 2026.1.1.0,
  Nsight Systems 2025.6.3.541.
- Runtime controls: PDL true, 148 SMs, tensor-core-util 100, eager stream 0.
- DeepGEMM JIT: fresh cache with `DG_JIT_WITH_LINEINFO=1`; the preserved
  compiler output and cubins resolve separate E1 and E5 specializations.
- NCU full reports: 45 replay passes for stock and 47 for PSUM. The source
  reports use 12 passes each. They are separate single observations, not
  paired statistics.
- Collection `20260723a` stopped after Nsys because `nsys stats` rejected an
  already-exported SQLite sibling. Its partial artifacts are retained and
  excluded. The corrected `20260723b` run uses fresh directories and passes
  the SQLite file directly to `nsys stats`.

## 1. Latency and speed of light

| Metric | Stock | PSUM | Delta |
|---|---:|---:|---:|
| NCU kernel duration | 629.824 us | 599.328 us | -30.496 us (-4.842%, 1.050884x) |
| Nsys kernel median, 5 launches | 633.117 us | 601.277 us | 1.052954x |
| Nsys kernel mean, 5 launches | 636.0418 us | 604.9190 us | 1.051450x |
| Nsys single timed kernel | 633.117 us | 601.053 us | 1.053346x |
| SM throughput | 88.3327% | 89.6407% | +1.3080 pp |
| tensor-pipe active, elapsed | 84.4151% | 83.9023% | -0.5128 pp |
| compute-memory throughput | 70.6829% | 73.9609% | +3.2780 pp |
| DRAM active cycles | 31.5519% | 27.9355% | -3.6164 pp |
| issue active | 11.1455% | 12.0537% | +0.9082 pp |
| eligible warps/cycle | 0.117086 | 0.127376 | +8.788% |

The kernel is compute/latency/synchronization limited, not HBM-bandwidth
limited. Both variants keep the Blackwell tensor pipe near 84% of elapsed
cycles while DRAM activity remains below 32%. The curated HMMA metric is zero
because this path uses UTCQMMA; it is not evidence that tensor cores are idle.

## 2. Memory, TMA, TMEM, and cache

| Metric | Stock | PSUM | Relative delta |
|---|---:|---:|---:|
| DRAM read bytes | 1,247,816,704 | 1,023,947,264 | -17.941% |
| DRAM write bytes | 276,776,448 | 260,486,400 | -5.886% |
| total DRAM bytes | 1,524,593,152 | 1,284,433,664 | -15.752% |
| scalar/global-load SASS instructions | 528,000 | 31,006 | -94.128% |
| global-load L1 lookup misses | 8,800 | 592 | -93.273% |
| L1 sector hit rate | 98.3196% | 97.8571% | -0.4624 pp |
| L2 sector hit rate | 76.8537% | 79.8719% | +3.0182 pp |
| input TMA bytes | 10,489,036,800 | 10,489,036,800 | unchanged |
| TMA load instructions | 1,056,000 | 1,056,000 | unchanged |
| output TMA bytes | 288,358,400 | 272,367,616 | -5.545% |
| TMA store instructions | 140,800 | 132,992 | -5.545% |
| TMEM load instructions | 1,126,400 | 1,063,936 | -5.545% |
| UTCQMMA work | 1,771,674,009,600 | 1,673,426,632,704 | -5.545% |

Input TMA volume and load instructions are identical, so PSUM does not reduce
the counted input-tile feed. The matched 5.545% reductions in UTCQMMA work,
TMEM loads, and output TMA stores support reduced tail-validity/epilogue work
inside the same 148-CTA persistent launch. The 94.1% reduction in scalar
global loads is consistent with replacing row-wise expert lookup by one
cumulative endpoint per expert. These are correlations across two template
specializations, not proof of one exclusive cause.

PSUM trades this benefit for more shared-memory conflict work: shared-load
conflicts rise 500,754 -> 603,573 (+20.533%), and shared-store conflicts rise
3,651,771 -> 5,239,413 (+43.476%). NCU rule estimates overlap and cannot be
summed into a projected speedup.

## 3. Occupancy, waves, and expert tails

Both kernels launch 148 blocks of 256 threads as 74 two-CTA clusters: one CTA
and eight warps per B200 SM, one wave total. Both allocate 214,828 bytes of
shared memory per block (213,804 dynamic plus 1,024 driver/cubin shared bytes),
which limits residency to one block per SM.

| Resource | Stock | PSUM |
|---|---:|---:|
| registers/thread | 38 | 50 |
| achieved active warps | 12.5214% | 12.5187% |
| register occupancy limit | 6 blocks | 4 blocks |
| shared-memory occupancy limit | 1 block | 1 block |
| cubin stack/local bytes | 0 / 0 | 0 / 0 |
| local-load/store instructions | 0 / 0 | 0 / 0 |

The register increase causes no residency loss and no spill. A second resident
CTA would require a major shared-memory/pipeline redesign, not register
tuning.

| Per-SM active cycles | Average | Maximum | Minimum |
|---|---:|---:|---:|
| Stock | 850,775.115 | 861,322 (+1.240%) | 839,950 (-1.272%) |
| PSUM | 808,589.764 | 819,366 (+1.333%) | 790,175 (-2.277%) |

No small set of SMs dominates the isolated kernel. The fixture's 32 expert
counts have min/max 975/1104, mean 1030.6875, median 1034, and population
standard deviation 29.2804. Alignment yields 13 slabs of 1024 and 19 slabs of
1152 rows; 2,218 padded rows are 6.3011% of the aligned allocation. This is a
deterministic router-contract replay, not checkpoint-derived routing evidence.

## 4. Scheduler stalls and synchronization

| Average other stalled warps per issue-active cycle | Stock | PSUM |
|---|---:|---:|
| long scoreboard | 10.075905 | 9.012832 |
| barrier | 4.347149 | 3.930249 |
| wait | 1.153951 | 1.164052 |
| short scoreboard | 0.460182 | 0.505980 |
| MIO throttle | 0.037065 | 0.084630 |
| branch resolving | 0.522442 | 0.580148 |
| no instruction | 0.197974 | 0.150155 |

Average warp latency per issued instruction improves 17.9577 -> 16.6472
cycles (-7.298%). Source PC sampling also moves modestly away from long
scoreboard: 17,152/30,766 samples (55.750%) become 16,032/29,214 (54.878%).
Barrier samples fall from 7,452 to 6,857. Residual wait, short-scoreboard,
branch, and MIO work become proportionally more visible.

The dominant mapped hotspot is `cutlass/arch/barrier.h:424`, an
`mbarrier.try_wait.parity` loop: 16,092 total/15,348 long-scoreboard samples
under stock versus 14,555/13,700 under PSUM. Stock also has 328
long-scoreboard samples at `cute/numeric/math.hpp:51`, the row-wise group
lookup site; that line is absent from the PSUM hotspot table. PSUM moves more
work to endpoint/control and store synchronization sites such as
`barrier.h:286` and `sm100_store_cd_swap_ab.cuh`.

## 5. Instruction and source evidence

Source/SASS mapping resolves the stock and PSUM symbols to distinct DeepGEMM
template specializations (`GemmType=1` and `GemmType=5`) at real positive
source lines. Both contain `UTCQMMA.2CTA`, `UTMALDG.2D`, `UTMASTG.2D`,
`LDTM`, `SYNCS.PHASECHK`, and `UCGABAR_WAIT`, confirming that the comparison
retains the same Blackwell tensor/TMA pipeline.

PSUM executes more total warp instructions (55,618,137 -> 57,023,168,
+2.526%) and branches (3,933,142 -> 4,926,144, +25.247%). At the same time,
active threads per warp improve 17.53 -> 20.76, predicated-on threads improve
16.85 -> 19.22, and divergent branch targets fall 70,844 -> 66,940
(-5.511%). The shorter dependency waits, higher lane utilization, and reduced
tail/output work outweigh the added control and shared-memory overhead.

`cuobjdump --dump-resource-usage` confirms `REG:38` versus `REG:50`, with
`STACK:0`, `LOCAL:0`, and `SHARED:1024` for both selected cubins. Exact source
lines differ across specializations, so opcode and aggregate-counter agreement
is used before interpreting line-level movement.

## NCU rule-engine estimates

NCU's rule estimates are local, overlapping opportunities. They must not be
summed or treated as predicted component or end-to-end speedup.

| Rule (reported estimate) | Stock | PSUM |
|---|---:|---:|
| L2 compression | 25.24% | 21.76% |
| global-load coalescing | 61.85% | 64.72% |
| shared-load conflicts | 26.71% | 31.29% |
| shared-store conflicts | 38.90% | 48.00% |
| scheduler issue / long scoreboard | 11.67% | 10.36% |
| divergence/predication | 41.83% | 35.80% |
| occupancy | 11.67% | 10.36% |

The coalescing estimate is particularly easy to overread. PSUM's remaining
scalar loads use only 4/32 bytes per sector, but absolute global-load
instructions are already 94.1% lower and misses are only 592. Shared conflicts,
branching, and MIO pressure are the clearer residual costs of PSUM.

## 6. PM-sampling tail

WarpStates PM instance series are available. Long-scoreboard active samples
fall 626 -> 594, their mean falls 1525.64 -> 1474.90, and the final-quarter to
first-quarter ratio remains slightly better at 0.6731 -> 0.6696. Wait rises
174.98 -> 190.30 mean, while MIO rises 5.78 -> 14.04. Dominant dependency
stall series decay toward completion rather than showing a worse late tail
under PSUM.

The four requested base-utilization PM series have no instances through this
NCU Python API for either report. `analysis/pm_series_status.txt` records the
available WarpStates series. Therefore PM evidence supports only a
WarpStates-tail statement; time-resolved SM, tensor, L1, and DRAM utilization
are not claimed. Aggregate utilization remains available from the full NCU
reports.

## Nsight Systems timing caveat

The trace contains exactly five stock and five PSUM DeepGEMM launches:
correctness, three warmups, and one timed pair. The profiled runner JSON reports
714.976 versus 609.472 us (1.173107x), but its timed-kernel durations are
633.117 versus 601.053 us. CUPTI's device-event interval decomposes as:

| Interval | Stock | PSUM |
|---|---:|---:|
| start marker to kernel | 71.712 us | 2.752 us |
| kernel | 633.117 us | 601.053 us |
| kernel to end marker | 5.504 us | 5.728 us |
| total marker span | 710.333 us | 609.533 us |

For PSUM, host-side tensor-map/launch preparation begins before its start
marker completes because the preceding untimed poison fill is still in flight;
stock preparation begins after its marker. Each event window contains exactly
one target kernel and no memcpy or memset. The profiled event ratio is
queue-order contamination and is not cited as a gate result. NCU duration,
five-launch Nsys distributions, and the locked unprofiled component gate agree
near 1.05x.

## Raw artifacts and hashes

| Artifact | SHA256 |
|---|---|
| `../moe-w13-prefill-paired-nsys-20260723b/reports/paired.nsys-rep` | `9c000d492439ecc4c064e3debba399d3cd6852ae3cce5b58e5a8ad20569bdb0d` |
| `../moe-w13-prefill-paired-nsys-20260723b/reports/paired.sqlite` | `b374bedc4af01a7828ae89e654f05fcd85eccd95daae0727b55b1bead217d3c2` |
| `../moe-w13-prefill-stock-ncu-20260723b/reports/stock_full.ncu-rep` | `f1dc7b885d6a4332abca6bc3f831b1fa5b537fff070b6cb65df9bd7d76e44e6c` |
| `../moe-w13-prefill-stock-ncu-20260723b/reports/stock_source.ncu-rep` | `8909c40077c15ad40b9ea164e3a86fa3c285ce250e1aae437ccb63223aa22306` |
| `../moe-w13-prefill-psum-ncu-20260723b/reports/psum_full.ncu-rep` | `21d33bec68ae07041747200b27fcd45148e382d6473978c8a9be9ded8edc89cd` |
| `../moe-w13-prefill-psum-ncu-20260723b/reports/psum_source.ncu-rep` | `e9194450672a09f0a54bf95b1c36f01bb7c06d0784a7a27a06be655d4d86961f` |

The machine-readable metric archive, curated comparison, source hotspots,
resource dump, PM plots/status, collection logs, runner JSON, source exports,
and exact helper hashes are preserved beside this report and in the three
collection directories.

## Ranked next directions

1. Retain the real `ep_scatter` cumulative endpoints and test the
   checkpoint-derived EP8 region first. Production adapter, output-gap,
   stream, CUDA Graph, and overlap semantics are now the binding evidence gap.
2. If the full region preserves the gain, reduce PSUM's shared-store conflicts,
   branch/control work, and MIO pressure without restoring row-wise lookup.
3. Investigate the residual `mbarrier.try_wait`, endpoint-control, and output
   store synchronization sites while holding UTCQMMA/TMA work and valid-row
   correctness fixed.
4. Treat higher occupancy as lower priority. Shared memory, not registers,
   fixes one CTA/SM; a second CTA requires a major pipeline redesign that may
   sacrifice the current tensor/TMA overlap.

No further DeepGEMM/CuTe source rewrite is justified before the real endpoint
adapter and containing-region gate exist: the library-native specialization
already clears the component threshold, and the missing production evidence
dominates the decision.

## Reproduction commands

The collection and analysis scripts contain the complete Nsys/NCU arguments,
kernel filters, JIT environment, source folders, and fail-closed directory
checks. From a fresh isolated worktree with a new immutable suffix:

```bash
cd /home/qinhaiyan/glm52-goal-runs/09-moe_w13_prefill/kernel-harness
/home/qinhaiyan/glm52-goal-runs/with_all_gpus_lock.sh \
  serving_native/evidence/09_moe_w13_prefill/run_profile_collection.sh
env -u CUDA_VISIBLE_DEVICES \
  bash serving_native/evidence/09_moe_w13_prefill/analyze_profiles.sh
```

Existing raw reports can be inspected CPU-only with `ncu --import` and
`nsys stats` against the preserved SQLite export. Do not reuse the timestamped
collection directories.

## Final diagnosis

PSUM improves this exact fused-N4096 component by eliminating most row-layout
global loads and reducing tail-validity, tensor, TMEM, and output-store work
while preserving the counted input TMA feed, one-wave grid, and one-CTA-per-SM
occupancy. It trades higher registers, branches, shared conflicts, and MIO
pressure for lower dependency waiting and better lane utilization.

No production replacement follows from this diagnosis. Live EP8 endpoint
capture, full-output semantics, overlap/graph behavior, the eight-rank MoE
region, and SGLang end-to-end performance remain unvalidated.
