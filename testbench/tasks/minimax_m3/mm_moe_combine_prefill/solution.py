"""MoE Combine — sglang production baseline (sgl_kernel.moe_sum).

MiniMax-M3 MoE combine (operator 41 in docs/minimax_m3_operators.csv): sums the top_k
expert outputs per token into the hidden vector.

  output_tensor[M, H] = sum(input_tensor[M, top_k, H], dim=1)

INTERFACE-EXACT: run(input_tensor, output_tensor) is a verbatim match of the
destination-passing moe_sum(input_tensor, output_tensor) — it writes the sum into
output_tensor in place (and also returns it so the harness can score it). reference.py
IS the correctness oracle AND the latency baseline.

top_k=4, H=6144 are the MiniMax-M3 MoE config (HF text_config; see
scripts/bench_minimax_m3.py).
"""

import torch
from sgl_kernel import moe_sum


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    M = axes_and_scalars["M"]
    TOPK = axes_and_scalars["top_k"]
    H = axes_and_scalars["H"]
    input_tensor = torch.randn(M, TOPK, H, device=device, dtype=torch.bfloat16)
    # DPS destination buffer, exactly as sglang passes to moe_sum (overwritten in full).
    output_tensor = torch.empty(M, H, device=device, dtype=torch.bfloat16)
    return {"input_tensor": input_tensor, "output_tensor": output_tensor}


@torch.no_grad()
def run(input_tensor, output_tensor):
    moe_sum(input_tensor, output_tensor)
    return output_tensor
