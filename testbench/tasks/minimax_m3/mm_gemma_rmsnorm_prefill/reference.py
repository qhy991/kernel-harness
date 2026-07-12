"""Gemma-RMSNorm — sglang production baseline (sgl_kernel.gemma_rmsnorm).

MiniMax-M3 standalone Gemma-RMSNorm (operator 36 in docs/minimax_m3_operators.csv).
The Gemma variant scales the normalized activation by (1 + weight), not weight:

  out[M, H] = x * rsqrt(mean(x**2, -1) + eps) * (1 + weight)    (fp32 accumulate)

INTERFACE-EXACT: run(x, weight, eps) is a verbatim match of sgl_kernel.gemma_rmsnorm
(eps is a scalar input), so a candidate is a symbol-for-symbol drop-in. reference.py IS
the correctness oracle AND the latency baseline: an optimized solution.py must match
this output within tolerance and run faster.

H=6144 is the MiniMax-M3 hidden size (HF text_config; see scripts/bench_minimax_m3.py).
"""

import torch
from sgl_kernel import gemma_rmsnorm

EPS = 1e-6


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    M = axes_and_scalars["M"]
    H = axes_and_scalars["H"]
    x = torch.randn(M, H, device=device, dtype=torch.bfloat16)
    weight = torch.randn(H, device=device, dtype=torch.bfloat16)
    return {"x": x, "weight": weight, "eps": EPS}


@torch.no_grad()
def run(x, weight, eps):
    return gemma_rmsnorm(x, weight, eps)
