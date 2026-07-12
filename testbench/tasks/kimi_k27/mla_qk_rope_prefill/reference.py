"""RoPE (rotary embedding, in-place) — sglang baseline (apply_rope_with_cos_sin_cache_inplace).

Shared by every model that applies partial RoPE to the q/k rope slice (Kimi-K2.7 MLA,
MiniMax-M3 GQA, DeepSeek-V3.2). RotaryEmbedding.forward_cuda dispatches, in-place
(rotary_embedding/base.py:379):

    apply_rope_with_cos_sin_cache_inplace(
        q, k, cos_sin_cache, positions, is_neox=..., fused_args=None)

INTERFACE-EXACT: run()'s signature is a verbatim copy of that kernel — q/k/cos_sin_cache/
positions positional, is_neox/rope_dim/fused_args keyword-only — so a winning candidate
is a symbol-for-symbol drop-in (integrate.py binds run() at that symbol with an identity
adapter and drives a real RotaryEmbedding forward).

  q,k rope portions rotated in place by cos/sin at `positions`; module returns (q, k).

reference.py IS the correctness oracle AND the latency baseline: an optimized solution.py
must match BOTH outputs within tolerance, keep the in-place contract, and run faster. The
cos_sin_cache is built (untimed) by get_inputs from a real get_rope module.
"""

import torch
from sglang.jit_kernel.rope import apply_rope_with_cos_sin_cache_inplace

IS_NEOX = True


def _cos_sin_cache(D: int, max_pos: int, base: float, device) -> torch.Tensor:
    # [max_pos, D] = concat(cos, sin); bit-identical to sglang get_rope's cache.
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, device=device).float() / D))
    freqs = torch.outer(torch.arange(max_pos, device=device).float(), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)  # float32


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    M = axes_and_scalars["M"]
    nH = axes_and_scalars["num_heads"]
    kvH = axes_and_scalars["kv_heads"]
    D = axes_and_scalars["rope_dim"]
    max_pos = axes_and_scalars["max_pos"]      # model max_position_embeddings (real)
    base = axes_and_scalars["rope_theta"]      # model rope_theta (real)
    q = torch.randn(M, nH, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(M, kvH, D, device=device, dtype=torch.bfloat16)
    cos_sin_cache = _cos_sin_cache(D, max_pos, base, device)
    positions = torch.randint(0, max_pos, (M,), device=device, dtype=torch.int64)
    return {"q": q, "k": k, "cos_sin_cache": cos_sin_cache, "positions": positions}


@torch.no_grad()
def run(q, k, cos_sin_cache, positions, *, is_neox=IS_NEOX, rope_dim=0, fused_args=None):
    # Exact apply_rope_with_cos_sin_cache_inplace signature + in-place contract.
    apply_rope_with_cos_sin_cache_inplace(
        q, k, cos_sin_cache, positions,
        is_neox=is_neox, rope_dim=rope_dim, fused_args=fused_args,
    )
    return q, k
