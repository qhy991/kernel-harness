# Prior Knowledge — o_proj decode extreme

The hbm35 campaign is the required starting point. Its fused CUDA pack plus
packed-UE8M0 DeepGEMM candidate is correct and stateless:

| M | latency | HBM | new 40% ceiling |
|---:|---:|---:|---:|
| 16 | 33.04 us | 38.27% | 31.611 us |
| 32 | 33.84 us | 37.54% | 31.757 us |

The remaining gap is only 1.4–2.1 us. Earlier alternatives establish negative
controls: a single-pass Triton GEMM was ~2x slower and a naive CUDA
weight-streaming GEMM was ~120x slower. Therefore the first optimization target
is the pack/allocation/launch path, not the DeepGEMM tcgen05 body.

Required BitLessons:

- `BL-20260718-fused-repack-single-launch`
- `BL-20260718-dont-handroll-fp8-gemm`

Relevant KernelWiki evidence: `kernel-grouped-gemm` for SM100 scheduling
principles and DeepGEMM/SM100 pages for tcgen05/TMEM/TMA implementation patterns.
Treat performance claims from different shapes/dtypes as hypotheses only.
