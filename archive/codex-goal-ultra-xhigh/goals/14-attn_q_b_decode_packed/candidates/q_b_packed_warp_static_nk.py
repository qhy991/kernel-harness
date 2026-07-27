"""Packed-warp q_b experiment with fixed production N/K specialization."""

from __future__ import annotations

import torch

from serving_native.candidates.q_b_packed_warp import _load_fork


def run(inputs: dict, runtime):
    del runtime
    x = inputs["x_fp8"]
    weight = inputs["weight_fp8"]
    x_scale = inputs["x_scale"]
    weight_scale = inputs["weight_scale"]
    if x_scale.dtype != torch.int32 or weight_scale.dtype != torch.int32:
        raise TypeError("q_b packed-warp candidate requires int32 UE8M0 scales")
    out = x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16)
    _load_fork().fp8_gemm_nt_packed_warp(
        (x, x_scale),
        (weight, weight_scale),
        out,
        compiled_dims="nk",
    )
    return out
