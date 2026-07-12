"""BF16 linear GEMM — sglang baseline (torch.matmul / cuBLAS).

The standard non-quantized projection sglang dispatches on B200 for the base
MiniMax-M3 checkpoint: LinearBase -> cuBLAS bf16 GEMM (see the M3 backend inventory,
op28 "Main Q/K/V 投影"). Covers the dense-FFN GateUp/Down, main QKV/O projections,
and shared-expert GEMMs — all of which are plain bf16 GEMMs on this checkpoint (the
-MXFP8 checkpoint would instead use deep_gemm/flashinfer, tracked separately).

  out[M, N] = x[M, K] @ weight[N, K].T   (bf16)

reference.py IS the correctness oracle AND the latency baseline. Weight is generated
untimed in get_inputs; only the GEMM is timed.
"""

import torch


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    M = axes_and_scalars["M"]
    K = axes_and_scalars["K"]
    N = axes_and_scalars["N"]
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    weight = torch.randn(N, K, device=device, dtype=torch.bfloat16) * (K ** -0.5)
    return {"x": x, "weight": weight}


@torch.no_grad()
def run(x, weight):
    return torch.matmul(x, weight.T)
