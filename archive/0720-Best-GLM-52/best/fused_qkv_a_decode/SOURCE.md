# fused_qkv_a_decode

- Origin: `experiments/fused_qkv_a_decode_deepgemm_fused`
- Mechanism: DeepGEMM-GLM52 fork `fp8_gemm_nt_fused` (UE8M0 pack inside GEMM)
- Why not task-local scale_pack.cu: N=2624, N%128=64 — packed layout assert fails
- CUPTI (2026-07-20): geomean ~1.69× vs stock f32-scale ref (M16/M32)
