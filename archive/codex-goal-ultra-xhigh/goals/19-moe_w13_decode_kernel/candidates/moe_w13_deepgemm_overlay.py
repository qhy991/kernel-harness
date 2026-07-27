"""Pinned DeepGEMM overlay candidate for fused GLM-5.2 W13.

The experimental module is loaded lazily on the first correctness call.  The
serving-native runner has already executed stock DeepGEMM at that point, so its
compiler/cache remains independent from the overlay compiler/cache.
"""

from __future__ import annotations

import os
from pathlib import Path


_OVERLAY_ID = (
    "731e7c7a97d269e4b9f482ea18d0e709a948f293-w13-a674bcf69"
)
_VARIANT = "w13-bm32-a674bcf69"
_DG = None
_EXPECTED_M_BY_BUCKET = {16: frozenset((4, 5)), 32: frozenset((8, 9))}


def _load_overlay():
    global _DG
    if _DG is not None:
        return _DG

    sglang_root = Path(os.environ["SGLANG_ROOT"]).resolve()
    manifest = sglang_root / "third_party" / "deepgemm_glm52" / "manifest.json"
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


def _validate_contract(inputs, runtime) -> None:
    p = runtime.workload.params
    common_actual = (
        p["experts_per_rank"],
        p["expert_slab"],
        p["k"],
        p["n"],
        inputs["expected_m"],
    )
    common_expected = (32, 1024, 6144, 4096)
    if (
        common_actual[:4] != common_expected
        or p["decode_m"] not in _EXPECTED_M_BY_BUCKET
        or inputs["expected_m"] != p["expected_m"]
        or inputs["expected_m"] not in _EXPECTED_M_BY_BUCKET[p["decode_m"]]
    ):
        raise RuntimeError(f"unsupported W13 overlay shape: {common_actual!r}")
    if str(inputs["activation_scale"].dtype) != "torch.int32":
        raise RuntimeError("W13 activation scale must use packed int32 UE8M0")
    if str(inputs["weight_scale"].dtype) != "torch.int32":
        raise RuntimeError("W13 weight scale must use packed int32 UE8M0")
    abi = runtime.workload.family
    if abi in ("moe_grouped_masked", "moe_compute_region"):
        if (
            str(inputs["weight_fp8"].dtype) != "torch.float8_e4m3fn"
            or inputs["recipe_a"] is not None
            or inputs["recipe_b"] is not None
        ):
            raise RuntimeError("invalid packed-FP8 W13 contract")
    elif abi == "moe_grouped_masked_nvfp4":
        if (
            str(inputs["weight_fp8"].dtype) != "torch.int8"
            or inputs["recipe_a"] != (1, 128)
            or inputs["recipe_b"] != (1, 32)
        ):
            raise RuntimeError("invalid NVFP4 W13 contract")
    else:
        raise RuntimeError(f"unsupported W13 overlay family: {abi}")


def launch(inputs, runtime):
    _validate_contract(inputs, runtime)
    deep_gemm = _load_overlay()
    kwargs = {
        "compiled_dims": "nk",
        "disable_ue8m0_cast": True,
    }
    if runtime.workload.family == "moe_grouped_masked_nvfp4":
        kwargs["recipe_a"] = inputs["recipe_a"]
        kwargs["recipe_b"] = inputs["recipe_b"]
    return deep_gemm.fp8_m_grouped_gemm_nt_masked(
        (inputs["activation_fp8"], inputs["activation_scale"]),
        (inputs["weight_fp8"], inputs["weight_scale"]),
        inputs["out"],
        inputs["masked_m"],
        inputs["expected_m"],
        **kwargs,
    )


def run(inputs, runtime):
    launch(inputs, runtime)
    return [
        inputs["out"][expert, : int(count)]
        for expert, count in enumerate(inputs["masked_m_host"])
    ]
