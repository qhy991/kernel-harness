# Goal 22 flexible-GPU revalidation

**Disposition: NO REPLACEMENT.** The pinned M16 combine-bound-32 source
experiment is correct but fails the repeated 3% gate in every eager and real
CUDA Graph session. No bucket is enabled; installed stock FlashMLA remains the
runtime for M16, M32, every other ABI, and every topology.

## Campaign identity and method

- Campaign: `flex-20260723T160729Z`, 2026-07-23 16:07:39–16:15:39 UTC.
- Scheduler: one uninterrupted `with_flexible_gpu.sh` lease on physical GPU 1,
  UUID `GPU-5b9be10b-5bfc-b658-9b31-f7ae8516dc54`, exposed as logical GPU 0.
- Source pins: Kernel-Harness `8c18448d9b76d2d648bec6da1e47587c59b26e73`,
  SGLang `d9fb72325cc73bb4b6f3ad96664513eb632e2459`, FlashMLA candidate
  `d18ff63a73dc6519432f59acb9f04365ce14bb10`.
- Stack: B200/SM100 (148 SMs), driver `610.43.02`, PyTorch
  `2.11.0+cu130`, CUDA compiler `13.2.78`, Nsight Systems `2025.6.3`,
  Nsight Compute `2026.1.1`.
- Dynamic snapshots remained P0 with memory clock 3996 MHz; reported SM clocks
  were 645/780/757/682 MHz at start/after-paired/after-Nsys/end. NCU measured
  1.95 GHz while replaying the profiled kernels.
- All 24 paired sessions, six context-only baselines, three Nsys traces, and
  seven NCU reports ran inside the same lease. The wrapper line, device
  snapshots, source status, and tool versions are independently checked by the
  [paired summary](analysis/paired_measurements_summary.md).

The acceptance statistic is the median of 100 alternating per-pair
`reference_ms / candidate_ms` ratios after warmup. The control is a pinned
same-source rebuild; the candidate differs only by the exact M16 bound-32
combine dispatch.

## Correctness and production ABI

- Both fixed local buckets call
  `sgl_kernel.flash_mla.flash_mla_with_kvcache` ->
  `sgl_kernel::fwd_kvcache_mla` ->
  `flash_fwd_splitkv_mla_fp8_sparse_kernel*` ->
  `flash_fwd_mla_combine_kernel*`.
- M16 uses eight splits/request; M32 uses four. Both use scheduler metadata
  `[148,8]`, FP8 paged KV64, sparse top-k 2048, BF16 Q
  `[M,1,64,576]`, and BF16 output `[M,1,64,512]`.
- Invalid `-1` slots, a nondefault stream, native graph capture, three replays,
  mutated-input replay, output non-aliasing, dtype, shape, and numerical checks
  passed.
- All 24 producer comparisons passed correctness. The two exact SGLang
  production-ABI tests passed in 23.608 seconds.

## Paired timing result

| Mode | Bucket | Rebuild-control session speedups | Candidate session speedups | Candidate 3% passes |
|---|---|---|---|---:|
| eager | M16 | `1.007414`, `1.014577`, `1.013117` | `1.024761`, `1.014102`, `1.016634` | 0/3 |
| eager | M32 | `1.007534`, `1.017690`, `0.995031` | `1.004522`, `1.021972`, `0.998688` | 0/3 |
| CUDA Graph | M16 | `0.991351`, `0.991282`, `0.993822` | `0.989357`, `0.990007`, `0.989124` | 0/3 |
| CUDA Graph | M32 | `0.989923`, `0.989446`, `0.993995` | `0.992105`, `0.985662`, `0.991270` | 0/3 |

The context-only M32 stock baseline drifted by -10.27% from session 1 to 3,
which is why it is excluded from gating. Same-session alternating pairs remain
the authority. The candidate has no 3% session and is consistently slower under
graph replay.

## Nsight Systems: complete main-plus-combine region

| Build / bucket | Main | Combine | PDL overlap | Chain span | Host gap |
|---|---:|---:|---:|---:|---:|
| installed stock M16 | 17.504 us | 12.480 us | 4.096 us | 25.888 us | 0 |
| installed stock M32 | 25.088 us | 9.824 us | 4.000 us | 30.912 us | 0 |
| bound-32 candidate M16 | 17.536 us | 12.224 us | 4.224 us | 25.536 us | 0 |

Main and combine execute on stream 7 with programmatic-dependent-launch
overlap. The candidate's one-trace chain change is 1.36%, below the gate and
not an acceptance measurement.

## Nsight Compute: six-dimension diagnosis

The full reports include PM sampling; source reports use the
version-compatible basic set plus `SourceCounters`. The durations below are
profiler-replay values and are not mixed with paired acceptance timing.

| Metric | Main M16 | Main M32 | Stock combine M16 | Candidate combine M16 |
|---|---:|---:|---:|---:|
| NCU duration | 22.720 us | 29.984 us | 10.752 us | 10.912 us |
| SM throughput | 14.21% | 20.07% | 4.35% | 3.62% |
| Tensor-pipe activity | 8.75% | 13.49% | 0% | 0% |
| DRAM read | 23.043 MB | 46.010 MB | 16.818 MB | 16.817 MB |
| DRAM read rate | 1.014 TB/s | 1.534 TB/s | 1.564 TB/s | 1.541 TB/s |
| DRAM peak | 13.24% | 20.04% | 20.49% | 20.19% |
| L2 hit rate | 14.80% | 14.96% | 0.51% | 0.47% |
| Achieved occupancy | 18.45% | 18.56% | 11.29% | 12.28% |
| Eligible warps/scheduler | 0.209 | 0.289 | 0.101 | 0.084 |
| Long-scoreboard ratio | 5.47 | 4.44 | 15.58 | 19.28 |

1. **Launch geometry.** Each main launch is one 148-block wave over 148 SMs.
   M16 combine has 128 blocks (`16 x 1 x 8`) or 0.173 waves/SM, leaving 20 SMs
   idle; NCU estimates a 13.51% grid-underfill opportunity for both builds.
2. **Occupancy and balance.** Main uses 168 registers/thread and 232,656 bytes
   of total per-block shared allocation, limiting theoretical occupancy to
   18.75%. Combine uses 48 registers/thread; the candidate reduces static
   shared allocation from 6,144 to 2,048 bytes, but achieved occupancy remains
   far below its 62.5% theoretical ceiling. Some combine SMs have zero active
   cycles because the grid has only 128 blocks.
3. **Stalls and source hotspots.** M16 main PC sampling records 375/1,054
   long-scoreboard and 373/1,054 barrier samples. Stock combine records 271/390
   long-scoreboard samples (69.5%); candidate records 298/401 (74.3%). The
   primary combine hotspot is the dependent accumulation at `combine.cu:136`,
   fed by the next-split `float4` load at line 141. The smaller split bound does
   not hide that dependency.
4. **Tensor cores.** The sparse main uses tensor instructions but only reaches
   8.75%/13.49% elapsed tensor-pipe activity. Combine is a scalar reduction and
   has no tensor activity. Neither kernel is tensor-compute-bound.
5. **Timeline.** These kernels are too short for a robust throughput timeline:
   PM sampling yields only 19/27 active main samples and 7/8 combine samples,
   while SM/DRAM throughput series are absent. The samples are retained as
   qualitative evidence only.
6. **Memory hierarchy.** Main L1 hit rate is below 1% and L2 is about 15%;
   NCU reports only 17.6/17.7 useful bytes per 32-byte global sector plus
   multi-way shared-memory conflicts. Combine rereads about 16.8 MB with about
   0.5% L2 hit rate in both builds. Its DRAM sectors are already about
   31.5/32 bytes useful, so the attempted bookkeeping reduction removes no
   meaningful gather traffic. No local-memory spills were measured.

The measured pattern matches the source-reported sparse-MLA prior art in
KernelWiki pages `wiki/kernels/flashmla.md` (`kernel-flashmla`) and
`wiki/kernels/sparse-mla.md` (`kernel-sparse-mla`): decode is dominated by
sparse cache movement and latency rather than peak tensor compute. All
acceptance claims here come from the local raw artifacts, not those prior-art
performance figures.

## Decision, fallback, and unavailable gates

The bound-32 body delivers its intended resource reduction but neither removes
the 128-block tail nor the 16.8 MB combine gather. It is rejected. Exact runtime
policy is an empty enable set:

| Case | Runtime |
|---|---|
| M16 production ABI | installed stock `sgl_kernel::fwd_kvcache_mla` |
| M32 production ABI | installed stock `sgl_kernel::fwd_kvcache_mla` |
| every other M, ABI, graph mode, or topology | stock FlashMLA |

The target GLM-5.2 checkpoint directory remains empty, so a complete SGLang
decode/server gate cannot run. This local attention microbenchmark is
world-size 1 on each DP rank and has no useful TP4 collective analogue; a
four-rank model diagnostic also needs the missing checkpoint. The host has four
GPUs, so the official one-node TP8/DP8/EP8 gate remains external and is not
weakened or relabelled. These unavailable promotion gates do not change the
no-replacement result because the candidate already fails its required local
paired and graph gates.

## Artifact map

- raw measurement audit:
  [paired summary](analysis/paired_measurements_summary.md);
- exact runtime semantics:
  [M16](analysis/runtime_stock_m16.json) and
  [M32](analysis/runtime_stock_m32.json);
- Nsys chain records:
  [stock M16](analysis/nsys_chain_stock_m16.json),
  [stock M32](analysis/nsys_chain_stock_m32.json), and
  [candidate M16](analysis/nsys_chain_combine32_m16.json);
- NCU key/full metrics, details, source views, stall hotspots, and PM timeline:
  [`analysis/`](analysis/);
- raw re-openable profiler reports: [`reports/`](reports/);
- wrapper, tests, environment checks, and collection output: [`logs/`](logs/);
- immutable file inventory: [SHA-256 manifest](analysis/artifact_sha256.txt).

## Reproduction

From the isolated Kernel-Harness root, choose a new immutable campaign ID:

```bash
profile/dsa-flashmla-kv-stock-b200-20260722/harness/launch_flexible_campaign.sh \
  flex-YYYYMMDDTHHMMSSZ
```

That launcher performs the only GPU-wrapper invocation; its child script runs
the complete paired campaign, Nsys, and NCU before releasing the lease. After
completion, recompute and check the paired summary on CPU:

```bash
profile/dsa-flashmla-kv-stock-b200-20260722/harness/summarize_flexible_campaign.py \
  --campaign-root profile/dsa-flashmla-kv-stock-b200-20260722/campaigns/<id>
profile/dsa-flashmla-kv-stock-b200-20260722/harness/summarize_flexible_campaign.py \
  --campaign-root profile/dsa-flashmla-kv-stock-b200-20260722/campaigns/<id> \
  --check
```

The exact NCU commands are preserved in `harness/collect_ncu.sh`; all invoked
commands and report paths are also retained in `logs/ncu_collection.txt`.
