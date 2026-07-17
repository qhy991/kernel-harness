"""Grouped MoE GEMM (decode, masked) — sglang baseline (deep_gemm.fp8_m_grouped_gemm_nt_masked).

The routed-expert GEMM on the DeepEP low-latency (masked) decode path, shared by every
DSA/MoE model (Kimi-K2.7, MiniMax-M3, DeepSeek-V3.2). Each of E local experts multiplies
its own token block; `masked_m[e]` gives the valid token count per expert. Scale prep
mirrors sglang's moe_runner/deep_gemm.py (per-token act + per-block weight UE8M0 casts,
then transform_sf_into_required_layout) and is done OFFLINE in get_inputs — run() is only
the masked grouped GEMM, matching the kernel-harness timing contract.

  out[E, Mp, N] = grouped(a_fp8[E, Mp, K] @ b_fp8[E, N, K].T), masked to masked_m per expert.

E (local experts under EP), K, N come from the task axes (per-model config). reference.py
IS the correctness oracle AND the latency baseline.
"""

import torch
import deep_gemm
from deep_gemm.utils.math import align, per_block_cast_to_fp8, per_token_cast_to_fp8


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    E = axes_and_scalars["E"]
    M = axes_and_scalars["M"]
    K = axes_and_scalars["K"]
    N = axes_and_scalars["N"]
    Ka, Na, Mp = align(K, 128), align(N, 128), align(M, 128)

    a = torch.randn(E, Mp, Ka, device=device, dtype=torch.bfloat16)
    b = torch.randn(E, Na, Ka, device=device, dtype=torch.bfloat16) * (Ka ** -0.5)

    a_fp8, a_s = zip(*[per_token_cast_to_fp8(a[e], use_ue8m0=True) for e in range(E)])
    b_fp8, b_s = zip(*[per_block_cast_to_fp8(b[e], use_ue8m0=True) for e in range(E)])
    a_fp8, a_s = torch.stack(a_fp8), torch.stack(a_s)
    b_fp8, b_s = torch.stack(b_fp8), torch.stack(b_s)
    a_s = deep_gemm.transform_sf_into_required_layout(
        a_s, mn=Mp, k=Ka, recipe=(1, 128, 128), num_groups=E, is_sfa=True
    )
    b_s = deep_gemm.transform_sf_into_required_layout(
        b_s, mn=Na, k=Ka, recipe=(1, 128, 128), num_groups=E, is_sfa=False
    )
    out = torch.empty(E, Mp, Na, device=device, dtype=torch.bfloat16)
    masked_m = torch.full((E,), M, device=device, dtype=torch.int32)
    return {"a_fp8": a_fp8, "a_s": a_s, "b_fp8": b_fp8, "b_s": b_s,
            "out": out, "masked_m": masked_m, "expected_m": M}


@torch.no_grad()
def run(a_fp8, a_s, b_fp8, b_s, out, masked_m, expected_m):
    deep_gemm.fp8_m_grouped_gemm_nt_masked((a_fp8, a_s), (b_fp8, b_s), out, masked_m, expected_m)
    return out
