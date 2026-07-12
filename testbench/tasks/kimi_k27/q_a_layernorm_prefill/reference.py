"""RMSNorm — sglang production baseline (sgl_kernel.rmsnorm).

Standalone Kimi-K2.7 RMSNorm on the no-residual path (q_a_layernorm / kv_a_layernorm):
the exact kernel sglang dispatches in RMSNorm.forward_cuda when residual is None
(python/sglang/srt/layers/layernorm.py:329 -> sgl_kernel.rmsnorm). This file reads
the (M, H) axes from definition.json and calls that kernel. `reference.py` IS the
correctness oracle AND the latency baseline: an optimized solution.py must match this
output within the per-workload tolerance and run faster.

  out[M, H] = x[M, H] * rsqrt(mean(x**2, dim=-1) + eps) * weight,  fp32 accumulate.

Memory-bound elementwise op: reads x[M,H] + weight[H], writes out[M,H]. Only the
kernel is timed (inputs are generated untimed by get_inputs), matching the
kernel-harness measurement contract.
"""

import torch
from sgl_kernel import rmsnorm

EPS = 1e-6  # DeepSeek/Kimi rms_norm_eps


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate bf16 hidden states + norm weight in the exact sgl_kernel layout."""
    M = axes_and_scalars["M"]
    H = axes_and_scalars["H"]
    x = torch.randn(M, H, device=device, dtype=torch.bfloat16)
    weight = torch.randn(H, device=device, dtype=torch.bfloat16)
    return {"x": x, "weight": weight}


@torch.no_grad()
def run(x, weight):
    return rmsnorm(x, weight, EPS)
