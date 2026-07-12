"""Fused add-RMSNorm — sglang production baseline (sgl_kernel.fused_add_rmsnorm).

The Kimi-K2.7 main-path layernorm (input_layernorm / post_attention_layernorm) is
NEVER a standalone RMSNorm in the hot path: sglang welds the residual add into the
norm. RMSNorm.forward_cuda dispatches, in-place, exactly:

    fused_add_rmsnorm(x, residual, weight, eps)      # layernorm.py:327
    return x, residual                               # x=normed, residual=x+residual

This task is INTERFACE-EXACT: run()'s signature and in-place/return contract are a
verbatim copy of that kernel call, so a winning candidate is a symbol-for-symbol
drop-in — integrate.py binds run() to sgl_kernel.fused_add_rmsnorm with an identity
adapter, and a real sglang forward through it is the machine-check that the interface
still matches (the property that lets today's kernel test migrate to e2e later).

  residual_out = x + residual
  x_out        = rmsnorm(residual_out) * weight        (fp32 accumulate)
  # both written in place; the module returns (x_out, residual_out).

reference.py IS the correctness oracle AND the latency baseline: an optimized
solution.py must match BOTH outputs within tolerance, keep the in-place contract, and
run faster.
"""

import torch
from sgl_kernel import fused_add_rmsnorm

EPS = 1e-6  # DeepSeek/Kimi rms_norm_eps


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    M = axes_and_scalars["M"]
    H = axes_and_scalars["H"]
    x = torch.randn(M, H, device=device, dtype=torch.bfloat16)
    residual = torch.randn(M, H, device=device, dtype=torch.bfloat16)
    weight = torch.randn(H, device=device, dtype=torch.bfloat16)
    # eps is a scalar input so run()'s signature stays a verbatim match of the sglang
    # kernel: fused_add_rmsnorm(x, residual, weight, eps).
    return {"x": x, "residual": residual, "weight": weight, "eps": EPS}


@torch.no_grad()
def run(x, residual, weight, eps):
    # Exact sgl_kernel.fused_add_rmsnorm signature + in-place contract; the module
    # ignores the return and uses the mutated (x, residual).
    fused_add_rmsnorm(x, residual, weight, eps)
    return x, residual
