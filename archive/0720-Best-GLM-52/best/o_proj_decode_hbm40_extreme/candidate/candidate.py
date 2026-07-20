"""GLM-5.2 o_proj decode FP8 GEMM — packed-UE8M0 weight-streaming candidate (best-knob).

The frozen reference (``deep_gemm.fp8_gemm_nt`` with float32 block scales) streams the
~100 MB fp8 weight but consumes float32 scaling factors, which forces DeepGEMM's slower
float32-scale path (~52 us, ~24% HBM on B200). SGLang's production dispatch hands DeepGEMM
*packed int32 UE8M0* scales instead and reaches ~33 us (~38% HBM) — the same kernel, only the
scale representation differs.

This candidate reproduces that packed dispatch **statelessly and per call**: it losslessly repacks
the supplied float32 UE8M0-valued scales (exact powers of two) into DeepGEMM's packed-int32 layout
via a single fused CUDA kernel (built once at import, outside the timed window) and calls the same
``fp8_gemm_nt``. The packed UE8M0 output bytes are byte-identical to the proven hbm35 seed
(preserved pristine at ``../variants/seed``); ``scale_pack.cu`` here is that seed's kernel unchanged.
Nothing is cached across calls: every invocation reads the frozen inputs and rebuilds the packed
scales into a freshly allocated buffer.

The only difference from the pristine seed is a set of **DeepGEMM public launch knobs** chosen
per-shape from the campaign's measured 200-row knob cross-product (best correct configuration; see
``docs/deep_gemm_knob_sweep.md``): programmatic dependent launch (PDL) plus a
``(compiled_dims, num_sms, tc_util)`` triple. These are documented DeepGEMM launch knobs — not a GEMM
rewrite, no caching, no operand transform — and they are **saved and restored around the GEMM call**
so the process-global DeepGEMM state is unchanged after ``run`` returns (the harness reference timing
and later shapes run under the frozen defaults). The knobs shave ~0.6-0.7 us versus the pristine seed
but still leave both frozen decode shapes short of the 40% HBM ceiling — this candidate is the best
correct configuration the campaign found; the target is an evidence-backed no-go
(``docs/no_go_disposition.md``).

Byte accounting: the ~100 MB ``w_fp8`` weight is passed through untouched and streamed once. The only
added per-call materialization is the packed scale buffer — ~32 KB of source float32 scales read, and
a ~0.77 MB packed int32 output (with the weight block scale expanded to per-row, as the packed layout
requires), i.e. <0.8% of the weight stream.
"""
from __future__ import annotations

import os
import warnings

import torch
import deep_gemm

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Fused scale-pack CUDA kernel, compiled once at import (outside the timed call) ──
# A pure-torch path (via DeepGEMM's own helpers) is kept as a correctness fallback so the
# candidate always produces the right answer even if the extension cannot be built; the
# fast path is the compiled kernel. A build failure is surfaced loudly (not silent) because
# it would drop the candidate onto the slow path and quietly miss the HBM target.
_pack_scales = None
try:
    from torch.utils.cpp_extension import load as _load_ext

    _ext = _load_ext(
        name="glm52_oproj_decode_scale_pack",
        sources=[os.path.join(_HERE, "scale_pack.cu")],
        verbose=False,
    )
    _pack_scales = _ext.pack_scales
except Exception as _build_err:  # pragma: no cover - build environment fallback only
    _pack_scales = None
    warnings.warn(
        f"glm52 o_proj decode: scale-pack CUDA extension failed to build "
        f"({type(_build_err).__name__}: {_build_err}); falling back to the slower "
        f"pure-torch packing path — the HBM target will likely be missed.",
        RuntimeWarning,
        stacklevel=2,
    )

# Per-shape best DeepGEMM launch knobs from the measured cross-product (docs/deep_gemm_knob_sweep.md).
# compiled_dims is a per-call argument; pdl/num_sms/tc_util are process-global setters that are
# saved and restored around the GEMM so nothing leaks into the harness reference timing.
_BEST_KNOBS = {
    16: {"compiled_dims": "mnk", "num_sms": 148, "tc_util": 50, "pdl": True},
    32: {"compiled_dims": "mk", "num_sms": 74, "tc_util": 80, "pdl": True},
}
_DEFAULT_KNOBS = {"compiled_dims": "nk", "num_sms": 148, "tc_util": 100, "pdl": True}


_KNOB_NAMES = ("pdl", "num_sms", "tc_util")


def _save_knobs():
    """Snapshot the process-global DeepGEMM launch knobs, each independently (value is None if that
    getter is unavailable so restoration never skips the others)."""
    saved = {}
    for name in _KNOB_NAMES:
        try:
            saved[name] = getattr(deep_gemm, f"get_{name}")()
        except Exception:  # pragma: no cover - older/newer deep_gemm without this getter
            saved[name] = None
    return saved


def _apply_knobs(pdl=None, num_sms=None, tc_util=None):
    """Apply each knob independently; a None value (or a failing setter) is skipped without
    preventing the others from being set/restored."""
    for name, value in (("pdl", pdl), ("num_sms", num_sms), ("tc_util", tc_util)):
        if value is None:
            continue
        try:
            getattr(deep_gemm, f"set_{name}")(value)
        except Exception:  # pragma: no cover
            pass


def _pack_scales_torch(x_scale: torch.Tensor, w_scale: torch.Tensor):
    """Correctness fallback: pack via DeepGEMM's reference helper (slower, not on the
    fast path). Weight block scales are expanded to per-row before packing, exactly as
    SGLang's production weight-requant does."""
    from deep_gemm import get_mn_major_tma_aligned_packed_ue8m0_tensor as _packt

    n = w_scale.shape[0] * 128
    row_of_block = torch.arange(n, device=w_scale.device) // 128
    xp = _packt(x_scale)
    wp = _packt(w_scale.index_select(-2, row_of_block))
    return xp, wp


@torch.no_grad()
def run(inputs: dict):
    x_fp8 = inputs["x_fp8"]
    w_fp8 = inputs["w_fp8"]
    x_scale = inputs["x_scale"]
    w_scale = inputs["w_scale"]
    out = inputs["out"]

    # Repack the float32 UE8M0 scales into DeepGEMM's packed-int32 layout (byte-identical to the
    # seed), then run the production-class packed dispatch under the per-shape best launch knobs.
    if _pack_scales is not None:
        x_scale_packed, w_scale_packed = _pack_scales(x_scale, w_scale)
    else:
        x_scale_packed, w_scale_packed = _pack_scales_torch(x_scale, w_scale)

    knobs = _BEST_KNOBS.get(x_fp8.shape[0], _DEFAULT_KNOBS)
    saved = _save_knobs()
    try:
        _apply_knobs(pdl=knobs["pdl"], num_sms=knobs["num_sms"], tc_util=knobs["tc_util"])
        deep_gemm.fp8_gemm_nt(
            (x_fp8, x_scale_packed),
            (w_fp8, w_scale_packed),
            out,
            compiled_dims=knobs["compiled_dims"],
        )
    finally:
        # Restore the frozen global defaults (each knob independently) so the harness reference and
        # later shapes are unaffected — even if fp8_gemm_nt raised or a single setter failed.
        _apply_knobs(**saved)
    return out
