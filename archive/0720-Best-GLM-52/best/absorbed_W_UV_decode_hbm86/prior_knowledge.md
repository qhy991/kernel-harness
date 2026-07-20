# Prior Knowledge — absorbed_W_UV decode >86% HBM

## Baseline and target

Measured 2026-07-19 on idle NVIDIA B200 GPU 2, Kernel-Harness `7d79e5e`.
Seed SHA-256:
`d4fccc4ae38c1a942c8fbd1fb0c291b88e5ce2b2387879c433f0ebbcb1619a6c`.

| M | candidate us | reference us | HBM utilization | correctness |
|---:|---:|---:|---:|---|
| 16 | 5.408 | 5.408 | 21.81% | PASS, calc_diff 0 |
| 32 | 5.728 | 5.728 | 22.88% | PASS, calc_diff 0 |

At 8.0 TB/s, strict 86% requires:

- M16: 9,437,184 B / (0.86 × 8 TB/s) = `<1.371683721 us`.
- M32: 10,485,760 B / (0.86 × 8 TB/s) = `<1.524093023 us`.

These ceilings are below the ~2 us CUPTI cold-L2 single-kernel floor observed
by prior B200 campaigns. This is a strong prior, not proof for this task:
reproduce the floor using the exact timer and selected idle GPU.

## KernelWiki evidence

- `sources/prs/vllm/PR-27284.md`: swap-AB for SM100 FP8 GEMM at M<=64.
- `sources/prs/vllm/PR-19566.md`: SM100 FP8 CUTLASS tuning.
- `sources/prs/sglang/PR-3056.md`: native FP8 BMM integration.
- `sources/prs/flashinfer/PR-1397.md`: CUTLASS FP8 BMM backend.

Treat source improvements only as mechanism hints. Final claims require this
task's frozen shapes, correctness contract and authoritative CUPTI timer.

## Source-edit opportunity

Baseline `sgl_kernel.bmm_fp8` wraps FlashInfer/cuBLASLt
(`sglang/sgl-kernel/csrc/gemm/bmm_fp8.cu`). If the measured single-kernel floor
is below the 86% ceilings, the plan must explicitly allow:

1. Task-local SM100 BMM source in `candidate/` (CUDA/CuTe/CUTLASS).
2. Isolated fork of `sgl-kernel` / FlashInfer BMM, loaded only by the candidate,
   without replacing Harness stock packages.

Wrapper-only or knob-only attempts are diagnostics, not the exclusive strategy.

## Required BitLesson

- `BL-20260719-single-kernel-span-floor`
