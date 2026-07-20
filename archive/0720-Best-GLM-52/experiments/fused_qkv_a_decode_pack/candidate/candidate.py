"""fused_qkv_a decode — packed UE8M0 scales (port of o_proj_decode_hbm35)."""
from __future__ import annotations

import os
import warnings

import torch
import deep_gemm

_HERE = os.path.dirname(os.path.abspath(__file__))
_pack_scales = None
try:
    from torch.utils.cpp_extension import load as _load_ext
    _ext = _load_ext(
        name="glm52_fused_qkv_a_decode_scale_pack",
        sources=[os.path.join(_HERE, "scale_pack.cu")],
        verbose=False,
    )
    _pack_scales = _ext.pack_scales
except Exception as e:
    _pack_scales = None
    warnings.warn(f"fused_qkv_a pack build failed: {type(e).__name__}: {e}", RuntimeWarning)


def _pack_scales_torch(x_scale, w_scale):
    from deep_gemm import get_mn_major_tma_aligned_packed_ue8m0_tensor as _packt
    n = w_scale.shape[0] * 128
    row_of_block = torch.arange(n, device=w_scale.device) // 128
    return _packt(x_scale), _packt(w_scale.index_select(-2, row_of_block))


@torch.no_grad()
def run(inputs: dict):
    out = inputs["out"]
    if _pack_scales is not None:
        xs, ws = _pack_scales(inputs["x_scale"], inputs["w_scale"])
    else:
        xs, ws = _pack_scales_torch(inputs["x_scale"], inputs["w_scale"])
    deep_gemm.fp8_gemm_nt(
        (inputs["x_fp8"], xs),
        (inputs["w_fp8"], ws),
        out,
        compiled_dims="nk",
    )
    return out
