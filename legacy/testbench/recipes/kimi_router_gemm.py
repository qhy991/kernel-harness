"""MoE Router GEMM (decode) — sglang baseline (sgl_kernel.dsv3_router_gemm).

Kimi-K2.7 MoE router logits on the decode fast path (deepseek_v2.py:599):

    logits = dsv3_router_gemm(hidden_states, router_weights, out_dtype=torch.float32)

The kernel is dispatched (deepseek_v2.py:585) when hidden==7168, num_tokens<=16, and
n_routed_experts in {256, 384}; canonical Kimi-K2.7 uses N = n_routed_experts = 384. It
REQUIRES num_tokens <= 16 (raises otherwise) — matching the DP32xEP32 decode regime
(M_local = 16). So this task's sweep is architecture-true: M in {1,2,4,8,16}.

  out[M, N] = (hidden_states @ router_weights.T)  in float32.

reference.py IS the correctness oracle AND the latency baseline: an optimized
solution.py must match this output within tolerance and run faster.
"""

import torch
from sgl_kernel import dsv3_router_gemm


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    M = axes_and_scalars["M"]
    H = axes_and_scalars["H"]
    N = axes_and_scalars["N"]
    hidden_states = torch.randn(M, H, device=device, dtype=torch.bfloat16)
    router_weights = torch.randn(N, H, device=device, dtype=torch.bfloat16)
    return {"hidden_states": hidden_states, "router_weights": router_weights}


@torch.no_grad()
def run(hidden_states, router_weights):
    return dsv3_router_gemm(hidden_states, router_weights, out_dtype=torch.float32)
