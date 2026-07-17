"""MLA weight-absorb BMM — sglang production baseline (torch.bmm).

Kimi-K2.7 MLA decode weight-absorb GEMMs (q_nope-absorb / v-absorb): a batched
matmul over attention heads with the absorbed projection weights. The CUDA bf16
default path sglang dispatches is torch.bmm (see kernel_api_mapping op 17/18).
`reference.py` IS the correctness oracle AND the latency baseline: an optimized
solution.py must match this output within tolerance and run faster.

  out[Bh, M, N] = bmm(a[Bh, M, K], b[Bh, K, N]),  Bh = num_heads.

Only the bmm is timed; the (untimed) get_inputs generates the per-head operands.
"""

import torch


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    M = axes_and_scalars["M"]
    Bh = axes_and_scalars["Bh"]
    K = axes_and_scalars["K"]
    N = axes_and_scalars["N"]
    a = torch.randn(Bh, M, K, device=device, dtype=torch.bfloat16)
    b = torch.randn(Bh, K, N, device=device, dtype=torch.bfloat16)
    return {"a": a, "b": b}


@torch.no_grad()
def run(a, b):
    return torch.bmm(a, b)
