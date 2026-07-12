"""MoE Gate: sigmoid + biased Top-4 routing — sglang baseline (sgl_kernel.topk_sigmoid).

MiniMax-M3 routing (inventory op37): scoring_func=sigmoid, top_k=4, routed_scaling=2.0,
with a routing correction bias. sglang dispatches sgl_kernel.topk_sigmoid, a DPS kernel
that writes topk_ids (int32) and topk_weights. This is the ROUTING oracle: it outputs the
selected expert INDICES [M, 4] with an EXACT-match tolerance (atol=0, rtol=0, ratio=1.0)
— routing is bit-exact, not tolerant. (M3 uses sigmoid, NOT Kimi's biased grouped topk;
different kernel, different config.)

Caveat: exact-index match assumes no score ties (measure-zero for random inputs); a
same-set-different-order result fails by design. reference.py IS the correctness oracle
AND the latency baseline.
"""

import torch
from sgl_kernel import topk_sigmoid

TOPK = 4


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    M = axes_and_scalars["M"]
    E = axes_and_scalars["num_experts"]
    gating_output = torch.randn(M, E, device=device, dtype=torch.float32)
    correction_bias = torch.randn(E, device=device, dtype=torch.float32)
    return {"gating_output": gating_output, "correction_bias": correction_bias}


@torch.no_grad()
def run(gating_output, correction_bias):
    M = gating_output.shape[0]
    topk_weights = torch.empty(M, TOPK, device=gating_output.device, dtype=torch.float32)
    topk_ids = torch.empty(M, TOPK, device=gating_output.device, dtype=torch.int32)
    topk_sigmoid(topk_weights, topk_ids, gating_output,
                 renormalize=True, correction_bias=correction_bias)
    return topk_ids
