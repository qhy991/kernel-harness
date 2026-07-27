# Nsight report: GLM-5.2 indexer score decode

## Scope and conclusion

This profile covers the exact M16/M32 rank-local score/top-k ABI selected by
current SGLang: DeepGEMM split-mode paged FP8 MQA, FP32 logits at context 8192,
and top-k-v2. It compares the shipped DeepGEMM and CuTe-DSL score backends and
profiles top-k separately.

The score kernels are latency/underfill limited, with long-scoreboard pressure,
not close to either the tensor or HBM roof. CuTe-DSL lowers resource use and
slightly shortens individual instrumented score kernels, but the improvement
is too small and unstable to survive the score/top-k and containing-region
gates. Top-k and logits initialization consume at least as much device time as
the M16 score kernel. The optimization decision is therefore no replacement.

## Collection contract

- GPU: NVIDIA B200, SM100, 148 SMs
- physical GPU: 1,
  `GPU-5b9be10b-5bfc-b658-9b31-f7ae8516dc54`
- stack: PyTorch `2.11.0+cu130`, CUDA `13.0`, SGLang
  `f93f8867b4bc124c9809c9110ec7361ed11b6b4a`
- one wrapper lease contained paired runs, Nsys, and NCU
- Nsys: one eager target range per backend/bucket
- NCU: full metric and source-counter reports for DeepGEMM, CuTe-DSL, and
  top-k at M16/M32

Raw reports are under `reports/`; extracted key/all metrics, source details,
and stall tables are under `analysis/`. The source-counter reports contain no
usable file/line correlation for the generated score kernels (`?:0`), so this
report does not invent a source line attribution.

## Nsight Systems: score-to-top-k handoff

One instrumented eager range contains exactly three kernels:

| Backend | M | fill | score | top-k | summed device kernels |
|---|---:|---:|---:|---:|---:|
| DeepGEMM | 16 | 1.280 us | 5.120 us | 5.888 us | 12.288 us |
| CuTe-DSL | 16 | 1.216 us | 4.928 us | 5.888 us | 12.032 us |
| DeepGEMM | 32 | 1.248 us | 6.464 us | 5.984 us | 13.696 us |
| CuTe-DSL | 32 | 1.408 us | 6.240 us | 6.048 us | 13.696 us |

At M16, replacing score saves 0.192 us in an instrumented 12.288 us device
sequence. At M32, the summed device time is unchanged. The projected NVTX
ranges are much longer because CUPTI perturbs Python launch timing; the table
uses kernel durations only, and unprofiled paired CUDA-event results drive the
decision.

## Nsight Compute: score kernels

| Backend | M | duration | grid/block | registers | shared/CTA | waves/SM | SM peak | DRAM read peak | read bytes / rate | tensor elapsed | eligible warps/cycle | long-scoreboard ratio |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| DeepGEMM | 16 | 10.432 us | 148/384 | 168 | 216.5 KiB | 1.0 | 9.79% | 21.80% | 17.398 MB / 1.668 TB/s | 8.78% | 0.125 | 14.107 |
| DeepGEMM | 32 | 13.664 us | 148/384 | 168 | 216.5 KiB | 1.0 | 15.77% | 33.32% | 34.775 MB / 2.545 TB/s | 14.14% | 0.146 | 13.956 |
| CuTe-DSL | 16 | 10.240 us | 148/384 | 80 | 112.625 KiB | 0.5 | 6.88% | 22.26% | 17.403 MB / 1.700 TB/s | 2.18% | 0.155 | 11.142 |
| CuTe-DSL | 32 | 12.192 us | 148/384 | 80 | 112.625 KiB | 0.5 | 11.50% | 37.28% | 34.780 MB / 2.853 TB/s | 3.86% | 0.179 | 11.479 |

Both implementations launch one CTA per SM. DeepGEMM is limited to one
resident block by both registers and shared memory; CuTe-DSL permits two
resident blocks in principle, but the grid still supplies only one CTA per SM.
Measured active warps remain 15.52-17.76% of peak. Neither implementation
spills: local load and store instruction counts are zero.

Long scoreboard dominates PC sampling: DeepGEMM reports 253/393 samples at M16
and 393/543 at M32; CuTe-DSL reports 188/385 and 345/541. Reducing resources
does not add parallel CTAs because the grid is fixed at 148, and it does not
remove the paged gather dependency chain.

## Nsight Compute: top-k

| M | duration | grid/block | registers | shared/CTA | waves/SM | SM peak | DRAM read peak | eligible warps/cycle |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | 9.760 us | 16/1024 | 32 | 27.75 KiB | 0.054 | 2.68% | 0.78% | 1.220 |
| 32 | 10.112 us | 32/1024 | 32 | 27.75 KiB | 0.108 | 5.14% | 1.44% | 1.238 |

Top-k is even more launch-underfilled: only 16 or 32 CTAs run on 148 SMs. Its
NCU replay duration is comparable to the score kernel while moving only
0.581 MB or 1.114 MB from DRAM. This confirms that an isolated score rewrite
cannot remove the score-to-top-k launch floor.

## Six-dimension diagnosis

1. **Compute:** elapsed tensor utilization is 8.78-14.14% for DeepGEMM and
   lower for CuTe-DSL. The path is not compute-roof limited.
2. **Memory:** score reads the expected 17.4/34.8 MB, but reaches only
   21.8-37.3% of DRAM read peak. It is not at an HBM roof.
3. **Launch and kernel count:** the score is one 148-CTA wave; fill and top-k
   are separate launches. At M16, top-k is longer than score in Nsys.
4. **Occupancy:** score active warps are only 15.5-17.8%; top-k launches only
   16/32 CTAs. CuTe-DSL's lower static resources do not create more work.
5. **Stalls:** long scoreboard is the dominant score stall, consistent with
   serialized paged-gather dependencies. There are no spills.
6. **Source/SASS boundary:** source counters map only to `?:0`. Register,
   spill, instruction-pipe, and stall metrics do not support a line-specific
   PTX/SASS rewrite hypothesis, so no unsupported SASS claim is made.

## Containing-region profile

The complete indexer capture contains nine GPU operations. Representative
M16 DeepGEMM kernel durations are 6.784 us for packed-FP8 `wq_b`, 6.208 us for
top-k, 5.056 us for score, 3.616 us for the BF16 projection kernel, and
1.920 us for its reduction. CuTe-DSL lowers score to 4.832 us, but the other
operations remain.

The selected DSA capture adds the TRT-LLM attention main and reduction kernels:
12.000 + 4.128 us at M16 and 17.504 + 4.160 us at M32 for the DeepGEMM
reference capture. This dilution agrees with the unprofiled DSA paired medians
of only `1.011807x` and `1.020334x`.

## Optimization decision

The shipped CuTe-DSL configuration is the relevant low-resource experiment.
It preserves the ABI and proves that a modest score-kernel reduction is
possible, but it reverses in one M32 graph series, regresses eagerly, and never
clears every repeated complete-indexer or selected-DSA series. A new score
kernel alone is unlikely to be worthwhile at these fixed buckets.

Future work would need to remove launches or materialization across the
fill/score/top-k boundary while preserving physical-slot semantics and graph
capture. Such a fusion is a new ABI/integration project and still requires the
same repeated containing-region and TP8 end-to-end gates.
