# Prior Knowledge — MoE Gate Projection Prefill MFU

## Live baseline (2026-07-18)

Measured on idle B200 GPU 3 with Kernel-Harness commit `7d79e5e`, DeepGEMM
0.1.4, and CUDA 13.0. The task candidate was clean; three unrelated Harness
candidates were already dirty.

| M | candidate us | reference us | MFU | target | target us |
|---:|---:|---:|---:|---:|---:|
| 1024 | 98.52 | 98.45 | 46.50% | 60% | 76.355 |
| 2048 | 159.64 | 159.66 | 57.39% | 67% | 136.755 |
| 4096 | 295.58 | 295.50 | 62.00% | 67% | 273.510 |

Every shape passed correctness with `calc_diff=0`. All three are compute-bound
under the frozen 4.50 PFLOP/s FP8 / 8.0 TB/s B200 roofline.

## Workload structure

The kernel is an E=8 masked grouped GEMM with K=6144 and N=2048. Rows are M*8,
distributed by a seeded multinomial into per-expert slabs. `expected_m` is the
largest-bin capacity, and `masked_m` gives each expert's valid row count.

The default uses DeepGEMM's masked grouped f32-block-scale path. It is not
SGLang's packed-scale production dispatch.

## Relevant prior art

KernelWiki page `kernel-grouped-gemm`
(`wiki/kernels/grouped-gemm.md`, confidence `source-reported`) identifies
variable-expert-M tile scheduling, persistent kernels, CLC dynamic scheduling,
TMA loads, and expert tail waste as the central SM100 grouped-GEMM levers.
Its sample performance claim is NVFP4 on a different shape and is not evidence
for this FP8 target.

SGLang PR 16622 records a Blackwell FP8 MoE scale-conversion correctness hazard:
incorrect UE8M0 requantization conditions can produce NaNs. Any scale-format
experiment must therefore retain all Harness anomaly and post-timing checks.
