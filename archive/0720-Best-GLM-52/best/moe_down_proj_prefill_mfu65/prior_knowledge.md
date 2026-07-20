# Prior Knowledge — MoE Down Projection Prefill 65% MFU

## Live baseline (2026-07-18)

Measured on B200 GPU 1 after confirming it was idle. GPU 1 remains shared with
an older campaign, so future runs require explicit interference checks.

| M | candidate us | reference us | MFU | target us |
|---:|---:|---:|---:|---:|
| 1024 | 95.85 | 95.84 | 47.80% | 70.481 |
| 2048 | 163.83 | 164.03 | 55.93% | 140.963 |
| 4096 | 320.06 | 319.74 | 57.26% | 281.926 |

All shapes passed with `calc_diff=0`. Arithmetic intensities 943–1440
FLOP/byte exceed the 562.5 ridge, so the frozen roofline classifies all as
compute-bound.

## Workload and prior art

The kernel is an E=8 masked grouped FP8 GEMM with K=2048 and N=6144. M*8 rows
are multinomially distributed across expert slabs; each shape has a different
`expected_m` and expert-tail profile.

KernelWiki `kernel-grouped-gemm` (source-reported) identifies persistent/CLC
tile scheduling, TMA, variable-M expert tails, and tile geometry as the central
SM100 grouped-GEMM levers. The page's NVFP4 performance number is a different
shape/dtype and is not evidence for this campaign.

The sibling MoE gate prefill campaign uses the same masked grouped DeepGEMM API
with different K/N. Its results may suggest knob families, but every claim must
be remeasured for K=2048,N=6144.
