"""LM-head logits GEMM — sglang baseline (torch.matmul, bf16).

Kimi-K2.7 final logits projection. On CUDA/B200 the default path
(logits_processor.py:_compute_lm_head) is:

    logits = torch.matmul(hidden_states.bfloat16(), lm_head.weight.T.bfloat16())

  out[M, vocab] = hidden_states[M, hidden] @ lm_head_weight[vocab, hidden].T   (bf16)

Canonical Kimi-K2.7: hidden=7168, vocab=163840 (the 2.35 GB weight makes this a
bandwidth-bound GEMM). Decode computes logits only for the sampled positions, so the
M-sweep is small. reference.py IS the correctness oracle AND the latency baseline.
"""

import torch


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    M = axes_and_scalars["M"]
    H = axes_and_scalars["H"]
    V = axes_and_scalars["V"]
    hidden_states = torch.randn(M, H, device=device, dtype=torch.bfloat16)
    lm_head_weight = torch.randn(V, H, device=device, dtype=torch.bfloat16) * (H ** -0.5)
    return {"hidden_states": hidden_states, "lm_head_weight": lm_head_weight}


@torch.no_grad()
def run(hidden_states, lm_head_weight):
    return torch.matmul(hidden_states.bfloat16(), lm_head_weight.T.bfloat16())
