"""MoE Router GEMM — sglang baseline (fp32 cuBLAS matmul).

MiniMax-M3 MoE router (inventory op37). The gate is a ReplicatedLinear with fp32
params, called on hidden.to(fp32) (minimax_m2.py: self.gate(hidden_states.to(fp32))),
so it is a plain fp32 matmul -> logits. Unlike Kimi, dsv3_router_gemm does NOT apply
(that kernel is compiled for hidden=7168; M3 hidden=6144, 128 experts -> falls to the
standard Linear path).

  logits[M, 128] = hidden[M, 6144] @ gate_weight[128, 6144].T   (fp32)

reference.py IS the correctness oracle AND the latency baseline.
"""

import torch


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    M = axes_and_scalars["M"]
    H = axes_and_scalars["H"]
    E = axes_and_scalars["E"]
    hidden_states = torch.randn(M, H, device=device, dtype=torch.float32)
    gate_weight = torch.randn(E, H, device=device, dtype=torch.float32) * (H ** -0.5)
    return {"hidden_states": hidden_states, "gate_weight": gate_weight}


@torch.no_grad()
def run(hidden_states, gate_weight):
    return torch.matmul(hidden_states, gate_weight.T)
