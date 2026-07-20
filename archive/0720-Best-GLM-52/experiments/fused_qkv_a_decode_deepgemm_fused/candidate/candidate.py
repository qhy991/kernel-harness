"""fused_qkv_a decode — DeepGEMM-GLM52 fork fused UE8M0 pack (same path as q_b)."""
from __future__ import annotations

import os
import sys
import warnings

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOADER_DIR = "/home/qinhaiyan/KDA-Pilot-Exp/llm/scripts/deepgemm_glm52"
if _LOADER_DIR not in sys.path:
    sys.path.insert(0, _LOADER_DIR)

from loader import load_deep_gemm_experimental  # noqa: E402

deep_gemm_experimental = load_deep_gemm_experimental()
_HAS_FUSED = hasattr(deep_gemm_experimental, "fp8_gemm_nt_fused")

_pack_scales = None
try:
    from torch.utils.cpp_extension import load as _load_ext

    _ext = _load_ext(
        name="glm52_fused_qkv_a_decode_fork_scale_pack",
        sources=[os.path.join(_HERE, "scale_pack.cu")],
        verbose=False,
    )
    _pack_scales = _ext.pack_scales
except Exception as e:  # pragma: no cover
    _pack_scales = None
    warnings.warn(
        f"fused_qkv_a deepgemm-fork: scale-pack build failed "
        f"({type(e).__name__}: {e}); using torch fallback.",
        RuntimeWarning,
        stacklevel=2,
    )


def _pack_scales_torch(x_scale: torch.Tensor, w_scale: torch.Tensor):
    from deep_gemm import get_mn_major_tma_aligned_packed_ue8m0_tensor as _packt

    n = w_scale.shape[0] * 128
    row_of_block = torch.arange(n, device=w_scale.device) // 128
    return _packt(x_scale), _packt(w_scale.index_select(-2, row_of_block))


_KNOB_NAMES = ("pdl", "num_sms", "tc_util")


def _save_knobs(mod):
    saved = {}
    for name in _KNOB_NAMES:
        try:
            saved[name] = getattr(mod, f"get_{name}")()
        except Exception:
            saved[name] = None
    return saved


def _apply_knobs(mod, pdl=None, num_sms=None, tc_util=None):
    for name, value in (("pdl", pdl), ("num_sms", num_sms), ("tc_util", tc_util)):
        if value is None:
            continue
        try:
            getattr(mod, f"set_{name}")(value)
        except Exception:
            pass


@torch.no_grad()
def run(inputs: dict):
    out = inputs["out"]
    saved = _save_knobs(deep_gemm_experimental)
    try:
        if _HAS_FUSED:
            deep_gemm_experimental.fp8_gemm_nt_fused(
                (inputs["x_fp8"], inputs["x_scale"]),
                (inputs["w_fp8"], inputs["w_scale"]),
                out,
            )
        else:
            if _pack_scales is not None:
                xs, ws = _pack_scales(inputs["x_scale"], inputs["w_scale"])
            else:
                xs, ws = _pack_scales_torch(inputs["x_scale"], inputs["w_scale"])
            deep_gemm_experimental.fp8_gemm_nt(
                (inputs["x_fp8"], xs),
                (inputs["w_fp8"], ws),
                out,
            )
    finally:
        _apply_knobs(deep_gemm_experimental, **saved)
    return out
