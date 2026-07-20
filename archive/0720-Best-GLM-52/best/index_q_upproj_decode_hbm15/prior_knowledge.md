# Prior Knowledge — index_q_upproj decode >15% HBM

## Contract and target

`out[M,4096] = x_fp8[M,2048] @ w_fp8[4096,2048].T`, M={16,32};
stock oracle `deep_gemm.fp8_gemm_nt`.

- M16: 8,555,520 B, strict ceiling `<7.129600000 us`.
- M32: 8,720,384 B, strict ceiling `<7.266986667 us`.
- Seed SHA:
  `11edbebb6641d8abfc776785ba52dba68aef0f9d67b81f8db7dc5cf774fd2ff6`.

## Evidence

- User reports ~6% HBM and an already validated custom split-K Triton path at
  ~2.5x. Campaign must reproduce this implementation/evidence locally.
- Nearest checked-in mechanism is `index_k_proj_decode` fused split-K, but that
  op has N=128; transfer the mechanism, not its tile assumptions.
- A local baseline probe measured 27.887/25.680 us but was invalid for final
  claims due 3.1–5.7x timing spread. Re-run on a clean idle GPU.

## Required BitLessons

- `BL-20260716-fused-splitk-reused-semaphore`
- `BL-20260718-fused-repack-single-launch`
- `BL-20260718-dont-handroll-fp8-gemm`
- `BL-20260718-deepgemm-ue8m0-prepack`
- `BL-20260719-single-kernel-span-floor`

## Source options

1. Primary: task-local fused split-K Triton, single launch/reduction and reused
   semaphore—no per-call memset.
2. Exact fused UE8M0 pack + packed stock DeepGEMM.
3. NCU-gated task-local CUDA/CuTe or isolated DeepGEMM-GLM52 fork.

Never overwrite Harness stock package or cache transformed operands.
