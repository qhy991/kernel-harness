"""MoE Gate + biased TopK routing — sglang baseline (sgl_kernel.kimi_k2_moe_fused_gate).

Kimi K2 fused gate: bias + renormalized top-k expert selection, producing routing
weights AND the selected expert indices in one kernel. This task is the ROUTING oracle:
it outputs the selected expert INDICES [M, topk] (int32) and requires an EXACT match
(atol=0, rtol=0, matched_ratio=1.0). Routing correctness is bit-exact, not tolerant —
and because the harness applies ONE tolerance per workload, int-exact indices can't be
co-judged with the float weights, so indices are the faithful single output here.

Config (authoritative, sgl-kernel test_kimi_k2_moe_fused_gate "Kimi K2"):
num_experts=384, topk=6, routed_scaling_factor=2.872.

Caveat: exact-index match assumes no score ties (measure-zero for random float inputs);
a candidate returning the same expert SET in a different order would (correctly, per the
kernel's ordering contract) fail. reference.py IS the correctness oracle AND the
latency baseline.
"""

import torch
from sgl_kernel import kimi_k2_moe_fused_gate

TOPK = 6
SCALING = 2.872


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    M = axes_and_scalars["M"]
    E = axes_and_scalars["num_experts"]
    input_tensor = torch.rand(M, E, device=device, dtype=torch.float32)
    bias = torch.rand(E, device=device, dtype=torch.float32)
    return {"input_tensor": input_tensor, "bias": bias}


@torch.no_grad()
def run(input_tensor, bias):
    _weights, indices = kimi_k2_moe_fused_gate(
        input_tensor, bias, topk=TOPK, renormalize=True, routed_scaling_factor=SCALING
    )
    return indices
