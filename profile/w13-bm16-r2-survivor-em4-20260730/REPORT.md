# W13 BM16 two-SM survivor: NCU bottleneck report (round-2 gate question)

One question was asked of Nsight Compute, per the round-2 plan's rule that NCU
is invoked only for a concrete survivor question:

> Is `infini_kernel_glm52_moe_w13_decode_em4_bm16_2sm` already at the achievable
> DRAM speed-of-light, or is a material fraction of its runtime exposed
> mbarrier / cluster-barrier wait?

## Collection

One GPU lease, physical GPU 0 `GPU-30b619de-87f2-1862-0d07-a595da8fe417` (the
same physical B200 round-1 measured on). Warmup 3, one profiled launch,
kernel-name filtered, `--set full` for both arms plus `--set source
--section SourceCounters` for the survivor. Exact frozen ABI: E32, slab 1024,
K6144, N4096, packed int32 UE8M0 scales, `masked_m = 4` on all 32 experts,
PDL on, 148 SMs, `tc_util` 100, `compiled_dims="nk"`.

Reports: `reports/full_candidate.ncu-rep`, `reports/full_stock.ncu-rep`,
`reports/source_candidate.ncu-rep`. Extracted values:
`analysis/metrics_key.json`, `analysis/stalls_per_pc.json`.

## Result

| Metric | BM16 two-SM candidate | stock BM128 two-SM |
|---|---:|---:|
| `gpu__time_duration.sum` (ns) | 126,656 | 135,744 |
| `dram__bytes_read.sum` | 815,190,016 | 841,518,848 |
| `dram__bytes_write.sum` | 6,996,992 | 31,928,576 |
| `dram__bytes_read.sum.pct_of_peak_sustained_elapsed` | **83.92%** | 80.82% |
| `dram__bytes_read.sum.per_second` | **6.436 TB/s** | 6.199 TB/s |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | **84.64%** | 83.89% |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | **4.57%** | 34.30% |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 28.18% | 37.57% |
| `lts__t_sector_hit_rate.pct` | 26.30% | 36.14% |
| registers / dynamic smem | 35 / 230,188 | 36 / 213,804 |
| grid / block / cluster / waves | 148 / 256 / 2 / 1.0 | 148 / 256 / 2 / 1.0 |

Warp stall ratios per issue-active cycle (`--set full`) and per-PC sample
shares (`--set source`):

| Stall reason | candidate ratio | stock ratio | candidate % of pcsamp |
|---|---:|---:|---:|
| `long_scoreboard` | 27.649 | 25.747 | 67.34% |
| `barrier` | 8.217 | 8.272 | 21.21% |
| `wait` | 1.443 | 1.392 | 4.10% |
| `short_scoreboard` | 0.585 | 0.570 | 1.62% |
| `sleeping` | 2.305 | 2.214 | — |
| `membar` / `lg_throttle` / `tex_throttle` | 0.000 | 0.000 | 0.00% |

Per-PC attribution puts 20.97% of all stall samples on a single
`SYNCS.PHASECHK.TRANS64.TRYWAIT`, i.e. an mbarrier **transaction**-barrier
try-wait. Every pipeline barrier in this kernel is a
`cutlass::arch::ClusterTransactionBarrier`, so that instruction is a consumer
waiting for a TMA transaction to complete — a memory-arrival wait expressed
through an mbarrier, not a synchronization-only cost. The epilogue's
`NamedBarrier::sync` lowers to `BAR.SYNC` (5 in the whole kernel) and does not
carry the barrier stall.

## Answer

**The survivor is at the achievable memory speed-of-light.**

1. Achieved 6.436 TB/s of DRAM reads, 83.92% of the tool's peak sustained
   (7.669 TB/s). Realistic sustained HBM3e on B200 is in the 6.5–7.0 TB/s band,
   so the kernel is running essentially at the top of that band.
2. Traffic is already 1.2% above the irreducible minimum. Every expert's full
   FP8 weight matrix must be read once: `32*4096*6144` = 805.31 MB. Measured
   reads are 815.19 MB, i.e. weights plus 9.88 MB of B-scales, activations and
   overhead. There is no traffic left to remove.
3. Compute is irrelevant: tensor pipe at 4.57% of peak. BM16 already converted
   stock's 34.30% tensor-pipe occupancy into memory-bound time, which is why
   `long_scoreboard` *rose* from 25.747 to 27.649 while duration fell.
4. 88.6% of stall samples (67.34% `long_scoreboard` + 21.21% transaction-barrier
   `barrier`) are memory-arrival waits. `membar` and all throttles are zero.

Absolute ceiling if DRAM could be driven to 100% of peak sustained:
`126.656 us * 0.8392` = 106.3 us, i.e. at most 1.19x — and that requires
perfect memory efficiency, which no real kernel reaches.

## Consequences for the round-2 hypotheses

- **H1 epilogue / TMEM store reduction — closed by measurement.** Writes are
  6.997 MB of 822.2 MB total traffic (0.85%). The store path is already at its
  instruction floor: 4 `LDTM.16`, 2 `STSM.16.MT88.4`, 2 `UTMASTG.2D` per
  scheduled block, one store stage of a 16x128 BF16 tile. Removing *all* output
  traffic could not reach 3%.
- **H3 BM32 rescue — closed by measurement.** In a kernel bound by bytes moved,
  BM32 strictly increases bytes: it doubles the activation footprint and the
  store surface versus BM16 (stock BM128 writes 31.93 MB, BM16 writes 6.997 MB).
  Its historical 1.028 BA estimator was a real deficit, and no PTX or tile
  change removes traffic. BM32 stays on stock.
- **H2 barrier / mbarrier overlap — one discriminating experiment remains.**
  The plan's constrained terminal-cluster-sync refactor is *not* authorized:
  its precondition is "if profiling attributes material time to it", and the
  three `UCGABAR` pairs execute once per launch inside ~232,400 SM cycles while
  the barrier stall sits on a steady-state transaction-barrier try-wait. What
  the profile does leave open is a per-k-block serialization in the steady-state
  pipeline, predeclared below.

## Predeclared round-2 identity (the single justified attempt)

**Hypothesis H2-SF-BYPASS.** In `sm100_fp8_fp4_gemm_1d1d.cuh` the UTCCP
transposer warp (warp 2) gates *every* k-block's UMMA issue through
`with_sf_full_barriers[stage_idx]`, but it performs real work only when
`k_block_idx % kNumSFAStagesPerLoad == 0`, which for K-granularity 128 is one
k-block in four. On the other three it waits `full_barriers[stage_idx]` and
immediately arrives at a 64-arrival (`kNumMulticast * 32`) barrier. That
inserts an extra warp-to-warp hop between TMA completion and UMMA issue on 75%
of k-blocks, lengthening the stage-recycle loop
TMA-arrive -> transposer-arrive -> UMMA-issue -> empty-arrive -> TMA-reissue.

Exact delta: for non-SF k-blocks the MMA warp waits `full_barriers[stage_idx]`
directly, and the transposer warp waits/arrives only on SF k-blocks. Safety
argument: `full_barriers` waits are non-consuming phase checks so a second
waiter is legal; `empty_barriers[s]` is still released only by the MMA warp
after it has waited `with_sf_full_barriers[s]` at SF stages, so TMA cannot
overwrite scale-factor shared memory before the transpose is observed; TMEM
scale-factor columns loaded by UTCCP at an SF stage remain valid for the
following three k-blocks by the existing `sfa_id`/`sfb_id` sub-index design,
and UTCCP and UMMA are issued in program order by the same warp. With 12
stages and `12 % 4 == 0`, scale-factor shared memory is only ever written at
stage indices 0, 4 and 8, so the SF-stage set is static.

Expected generated-code and profile change, stated before implementation:

- the `#pragma unroll 4` MMA k-loop specializes so one of four unrolled
  iterations waits `with_sf_full_barriers` and three wait `full_barriers`;
- the transposer loop loses its unconditional `SYNCS.ARRIVE.TRANS64` on three
  of four iterations;
- `barrier` stall share falls from 21.21% of samples;
- if the hypothesis is right, `dram__bytes_read.sum.pct_of_peak_sustained_elapsed`
  rises above 83.92% and duration falls;
- topology, registers, spills, dynamic shared memory, `UTCQMMA.2CTA` count,
  `LDTM`/`STSM`/`UTMASTG` counts, and numerics must be unchanged.

Prior probability of clearing 1.03 is low precisely because items 1-4 above say
the kernel is memory-limited, not serialization-limited. It is nevertheless the
correct single shot: it is the only predeclared round-2 mechanism not already
refuted arithmetically, and it converts "memory-system ceiling rather than
pipeline serialization" from inference into measurement. A negative result
closes the BM16 barrier/overlap direction with evidence.
