"""Bakeoff-compatible Codex MoE W13 DeepGEMM overlay for mainline serving_native.

Adapts goal-19 ``moe_w13_deepgemm_overlay.py`` to the mainline input contract
(no ``recipe_a``/``recipe_b``/``masked_m_host`` keys). Requires
``SGLANG_ROOT`` pointing at the goal-19 sglang worktree so the overlay
manifest resolves.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

_OVERLAY_ID = "731e7c7a97d269e4b9f482ea18d0e709a948f293-w13-a674bcf69"
_VARIANT = "w13-bm32-a674bcf69"
_DG = None
_EXPECTED_M_BY_BUCKET = {16: frozenset((4, 5)), 32: frozenset((8, 9))}


def _load_overlay():
    global _DG
    if _DG is not None:
        return _DG

    sglang_root = Path(os.environ.get("SGLANG_ROOT", "/home/qinhaiyan/sglang")).resolve()
    default_manifest = (
        sglang_root / "third_party" / "deepgemm_glm52" / "manifest.json"
    )
    manifest = Path(
        os.environ.get("SGLANG_GLM52_DEEPGEMM_MANIFEST", str(default_manifest))
    )
    os.environ["SGLANG_GLM52_DEEPGEMM_VARIANT"] = _VARIANT
    os.environ["SGLANG_GLM52_DEEPGEMM_MANIFEST"] = str(manifest)

    from sglang.srt.layers.glm52_opt.experimental_deepgemm import (
        get_experimental_deep_gemm,
    )

    _DG = get_experimental_deep_gemm()
    if _DG is None:
        raise RuntimeError("DeepGEMM overlay loader returned None")
    resolved = Path(_DG.__file__).resolve()
    if _OVERLAY_ID not in str(resolved):
        raise RuntimeError(
            f"unexpected DeepGEMM overlay import: {resolved}; expected {_OVERLAY_ID}"
        )
    return _DG


def run(inputs, runtime):
    expected = int(inputs["expected_m"])
    if expected in (4, 5):
        decode_m = 16
    elif expected in (8, 9):
        decode_m = 32
    else:
        raise RuntimeError(f"unsupported expected_m={expected}")
    if expected not in _EXPECTED_M_BY_BUCKET[decode_m]:
        raise RuntimeError(f"unsupported W13 overlay expected_m={expected}")
    if runtime.workload.family != "moe_grouped_masked":
        raise RuntimeError(f"unsupported family {runtime.workload.family}")
    if inputs["activation_scale"].dtype != torch.int32:
        raise RuntimeError("W13 activation scale must use packed int32 UE8M0")
    if inputs["weight_scale"].dtype != torch.int32:
        raise RuntimeError("W13 weight scale must use packed int32 UE8M0")

    deep_gemm = _load_overlay()
    deep_gemm.fp8_m_grouped_gemm_nt_masked(
        (inputs["activation_fp8"], inputs["activation_scale"]),
        (inputs["weight_fp8"], inputs["weight_scale"]),
        inputs["out"],
        inputs["masked_m"],
        inputs["expected_m"],
        compiled_dims="nk",
        disable_ue8m0_cast=True,
    )
    return [
        inputs["out"][expert, : int(count)]
        for expert, count in enumerate(inputs["masked_m"].tolist())
    ]
