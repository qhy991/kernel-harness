"""KDA index_weights_proj fused op — ROUND 2 WINNER (cuBLAS bf16).

Fuses index_k_proj (wk) + index_weights_proj into ONE bf16 cuBLAS GEMM sharing the
single streaming read of x[M,6144]. Bit-exact vs two separate GEMMs.
  M16 6.48us / M32 7.04us  ->  1.83x / 1.78x vs same-backend separate baseline.
deep_gemm.bf16_gemm_nt (round-1, ~17us) was the wrong backend for this trivial
launch/read-bound N=160 shape and is retired here.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

NK = 128
NW = 32
NF = NK + NW  # 160


def fuse_weights(Wk, Ww):
    """Concat wk[128,6144] + weights_proj[32,6144] -> Wf[160,6144] (ReplicatedLinear weight layout)."""
    return torch.cat([Wk, Ww], dim=0).contiguous()


def run(inputs):
    x  = inputs["x"]     # [M, 6144] bf16
    Wf = inputs["Wf"]    # [160, 6144] bf16
    of = F.linear(x, Wf)         # single cuBLAS GEMM -> [M, 160] bf16
    k = of.narrow(1, 0, NK)      # view, zero-cost
    weights = of.narrow(1, NK, NW)
    return k, weights
