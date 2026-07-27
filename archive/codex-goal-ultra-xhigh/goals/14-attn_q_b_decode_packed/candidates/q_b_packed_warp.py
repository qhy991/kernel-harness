"""GLM-5.2 attention q_b candidate for the production packed-UE8M0 ABI."""

from __future__ import annotations

import os

import torch

_FORK = None


def _load_fork():
    # Reuse SGLang's audited side-by-side loader so stock deep_gemm remains
    # untouched and the overlay manifest/provenance contract is identical to
    # production dispatch.
    os.environ.setdefault(
        "SGLANG_GLM52_DEEPGEMM_VARIANT",
        "glm52-qb-sm100-packed-broadcast-v3",
    )
    from sglang.srt.layers.glm52_opt.experimental_deepgemm import (
        get_experimental_deep_gemm,
    )

    global _FORK
    if _FORK is not None:
        return _FORK
    module = get_experimental_deep_gemm()
    if module is None or not hasattr(module, "fp8_gemm_nt_packed_warp"):
        raise RuntimeError("packed-warp DeepGEMM overlay entry is unavailable")
    _FORK = module
    return _FORK


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
        compiled_dims="",
    )
    return out
