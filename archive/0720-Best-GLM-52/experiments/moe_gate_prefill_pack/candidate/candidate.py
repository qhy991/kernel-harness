"""moe_gate_proj prefill — packed UE8M0 + PDL (port of moe_gate_proj_decode_hbm40)."""
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
        name="glm52_moe_gate_prefill_scale_pack",
        sources=[os.path.join(_HERE, "scale_pack.cu")],
        verbose=False,
    )
    _pack_scales = _ext.pack_scales
except Exception as e:
    _pack_scales = None
    warnings.warn(f"moe_gate prefill pack build failed: {type(e).__name__}: {e}", RuntimeWarning)


def _pack_scales_torch(x_scale, w_scale):
    from deep_gemm import get_mn_major_tma_aligned_packed_ue8m0_tensor as _packt
    n = w_scale.shape[1] * 128
    row = torch.arange(n, device=w_scale.device) // 128
    return _packt(x_scale), _packt(w_scale.index_select(-2, row))


def _get_pdl():
    try:
        return deep_gemm.get_pdl()
    except Exception:
        return None


def _set_pdl(value):
    if value is None:
        return
    try:
        deep_gemm.set_pdl(value)
    except Exception:
        pass


@torch.no_grad()
def run(inputs: dict):
    out = inputs["out"]
    if _pack_scales is not None:
        xs, ws = _pack_scales(inputs["x_scale"], inputs["w_scale"])
    else:
        xs, ws = _pack_scales_torch(inputs["x_scale"], inputs["w_scale"])
    saved = _get_pdl()
    try:
        _set_pdl(True)
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (inputs["x_fp8"], xs),
            (inputs["w_fp8"], ws),
            out, inputs["masked_m"], inputs["expected_m"],
            disable_ue8m0_cast=True,
        )
    finally:
        _set_pdl(saved)
    return out
