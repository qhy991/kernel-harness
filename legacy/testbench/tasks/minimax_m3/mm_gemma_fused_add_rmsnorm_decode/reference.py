"""Fused add-Gemma-RMSNorm — sglang baseline (sgl_kernel.gemma_fused_add_rmsnorm).

MiniMax-M3 main-path Gemma layernorm, welded with the residual add. GemmaRMSNorm
dispatches, in-place (layernorm.py, GemmaRMSNorm._forward_impl):

    gemma_fused_add_rmsnorm(x, residual, weight, eps)
    return x, residual          # x = normed, residual = x + residual

INTERFACE-EXACT: run(x, residual, weight, eps) is a verbatim copy of that kernel, same
in-place contract, same (normed, residual) return (eps is a scalar input). Gemma variant
scales by (1 + weight):

  residual_out = x + residual
  x_out        = rmsnorm(residual_out) * (1 + weight)      (fp32 accumulate)

reference.py IS the correctness oracle AND the latency baseline. H=6144 (MiniMax-M3).
"""

import torch
from sgl_kernel import gemma_fused_add_rmsnorm

EPS = 1e-6


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    M = axes_and_scalars["M"]
    H = axes_and_scalars["H"]
    x = torch.randn(M, H, device=device, dtype=torch.bfloat16)
    residual = torch.randn(M, H, device=device, dtype=torch.bfloat16)
    weight = torch.randn(H, device=device, dtype=torch.bfloat16)
    return {"x": x, "residual": residual, "weight": weight, "eps": EPS}


@torch.no_grad()
def run(x, residual, weight, eps):
    gemma_fused_add_rmsnorm(x, residual, weight, eps)
    return x, residual
