"""DSA fused Gemma-QK-norm + partial RoPE — sglang baseline (minimax_qknorm_rope, JIT CUDA).

MiniMax-M3 indexer/main QK path (inventory op29/35): a single-launch fused kernel that
Gemma-RMSNorms the per-head q/k slices of a packed QKV and applies partial RoPE, in the
DSA layout. sglang dispatches jit_kernel.minimax_qknorm_rope (C++ fused_gemma_qknorm_rope);
the inventory measures it 1.64-192.7us, ~2.4x faster than two separate launches.

  qkv[T, (nq+nk+nv)*HD] --(per-head Gemma-RMSNorm on q,k + partial RoPE rope_dim=64)--> qkv_out

Config: nq=64, nk=nv=4, head_dim=128, rope_dim=64, theta=5e6; norm weights are per-head_dim
(qk_norm_type=per_head), cos_sin_cache is fp32. Runs against the amd_add_m3 sglang build
(task.json sglang_dir). reference.py IS the correctness oracle AND the latency baseline.
"""

import torch
from sglang.jit_kernel.minimax_qknorm_rope import minimax_qknorm_rope

NQ, NK, NV, HD, RD, EPS, THETA = 64, 4, 4, 128, 64, 1e-6, 5000000
MAX_POS = 1048576


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    T = axes_and_scalars["M"]
    qkv = torch.randn(T, (NQ + NK + NV) * HD, device=device, dtype=torch.bfloat16)
    q_weight = (torch.randn(HD, device=device) * 0.1).to(torch.bfloat16)  # per-head_dim
    k_weight = (torch.randn(HD, device=device) * 0.1).to(torch.bfloat16)
    inv = 1.0 / (THETA ** (torch.arange(0, RD, 2, device=device).float() / RD))
    fr = torch.outer(torch.arange(MAX_POS, device=device).float(), inv)
    cos_sin_cache = torch.cat([fr.cos(), fr.sin()], dim=-1)  # [MAX_POS, RD] fp32
    positions = torch.randint(0, MAX_POS, (T,), device=device, dtype=torch.int64)
    return {"qkv": qkv, "q_weight": q_weight, "k_weight": k_weight,
            "cos_sin_cache": cos_sin_cache, "positions": positions}


@torch.no_grad()
def run(qkv, q_weight, k_weight, cos_sin_cache, positions):
    return minimax_qknorm_rope(qkv, q_weight, k_weight, cos_sin_cache, positions,
                               NQ, NK, NV, EPS)
