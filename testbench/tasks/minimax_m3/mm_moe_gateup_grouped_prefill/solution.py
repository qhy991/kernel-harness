"""Grouped MoE GEMM (prefill, contiguous) — sglang baseline (deep_gemm.m_grouped_fp8_gemm_nt_contiguous).

The routed-expert GEMM on the prefill (contiguous / NORMAL DeepEP) path, shared by every
DSA/MoE model (MiniMax-M3, Kimi-K2.7, DeepSeek-V3.2). Tokens are laid out contiguously and
`m_indices[i]` maps row i to its expert; each expert's block multiplies its weight. Scale
prep mirrors sglang's moe_runner (per-token act + per-block weight UE8M0 casts) and is done
OFFLINE in get_inputs — run() is only the contiguous grouped GEMM.

  out[E*M, N] = grouped(a_fp8[E*M, K] @ b_fp8[E, N, K].T) routed by m_indices.

M = tokens/expert; E, K, N from the task axes (per-model config). reference.py IS the
correctness oracle AND the latency baseline.
"""

import torch
import deep_gemm
from deep_gemm.utils.math import align, per_block_cast_to_fp8, per_token_cast_to_fp8


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    E = axes_and_scalars["E"]
    M = axes_and_scalars["M"]
    K = axes_and_scalars["K"]
    N = axes_and_scalars["N"]
    Ka, Na = align(K, 128), align(N, 128)
    m_sum = E * M

    a = torch.randn(m_sum, Ka, device=device, dtype=torch.bfloat16)
    b = torch.randn(E, Na, Ka, device=device, dtype=torch.bfloat16) * (Ka ** -0.5)
    a_fp8, a_s = per_token_cast_to_fp8(a, use_ue8m0=True)
    b_fp8, b_s = zip(*[per_block_cast_to_fp8(b[e], use_ue8m0=True) for e in range(E)])
    b_fp8, b_s = torch.stack(list(b_fp8)), torch.stack(list(b_s))
    out = torch.empty(m_sum, Na, device=device, dtype=torch.bfloat16)
    m_indices = torch.arange(m_sum, device=device, dtype=torch.int32) // max(M, 1)
    return {"a_fp8": a_fp8, "a_s": a_s, "b_fp8": b_fp8, "b_s": b_s,
            "out": out, "m_indices": m_indices}


@torch.no_grad()
def run(a_fp8, a_s, b_fp8, b_s, out, m_indices):
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous((a_fp8, a_s), (b_fp8, b_s), out, m_indices)
    return out
