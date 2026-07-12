"""Per-token-group activation FP8 quant — sglang baseline (sglang_per_token_group_quant_fp8).

The activation quantization that precedes every block-FP8 GEMM on the MiniMax-M3-MXFP8
checkpoint / fp8 path (it is exactly what sglang's deepgemm_w8a8_block_fp8_linear runs
inline before the matmul). Casts a bf16 activation to fp8_e4m3 with per-token 1x128-group
scales in the deep_gemm ue8m0 / tma-aligned / column-major layout.

  q_fp8[M, K] (fp8_e4m3), scale[M, K//512] (int32, ue8m0) = quant(x[M, K], group=128)

Memory-bound. reference.py IS the correctness oracle AND the latency baseline; run()
returns (q_fp8, scale) and the harness scores both.
"""

import torch
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

BLOCK = 128


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    M = axes_and_scalars["M"]
    K = axes_and_scalars["K"]
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    return {"x": x}


@torch.no_grad()
def run(x):
    q, s = sglang_per_token_group_quant_fp8(
        x, BLOCK, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True
    )
    return q, s
