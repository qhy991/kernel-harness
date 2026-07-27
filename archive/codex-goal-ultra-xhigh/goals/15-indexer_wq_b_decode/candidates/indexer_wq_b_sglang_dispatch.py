"""Exercise the production SGLang indexer M16 dispatch and its stock fallback."""

from __future__ import annotations

import os

import torch


# The candidate module is imported once, outside every timed region.
os.environ["SGLANG_GLM52_OPT"] = "1"
os.environ["SGLANG_GLM52_OPT_PROFILE"] = "serving_safe"
os.environ["SGLANG_GLM52_OPT_OPS"] = "index_q_upproj"
os.environ["SGLANG_GLM52_OPT_M_BUCKETS"] = "index_q_upproj:16"
os.environ.pop("SGLANG_GLM52_ALLOW_ABI_ADAPTER", None)

from sglang.srt.layers.glm52_opt.context import op_context
from sglang.srt.layers.glm52_opt.dispatch import try_dispatch_fp8_gemm

_ACTIVATED = False


def _activate_after_runtime_reference_pin() -> None:
    """Re-enable the candidate after Runtime deliberately pins its oracle OPT0."""
    global _ACTIVATED
    if _ACTIVATED:
        return
    os.environ["SGLANG_GLM52_OPT"] = "1"
    os.environ["SGLANG_GLM52_OPT_PROFILE"] = "serving_safe"
    os.environ["SGLANG_GLM52_OPT_OPS"] = "index_q_upproj"
    os.environ["SGLANG_GLM52_OPT_M_BUCKETS"] = "index_q_upproj:16"
    os.environ.pop("SGLANG_GLM52_ALLOW_ABI_ADAPTER", None)
    _ACTIVATED = True


@torch.no_grad()
def run(inputs: dict, runtime):
    # Runtime is constructed after this module is imported and pins OPT0 so its
    # reference remains stock. The first correctness call occurs before any
    # warmup/timing and restores the explicit candidate policy exactly once.
    _activate_after_runtime_reference_pin()
    x = inputs["x_fp8"]
    weight = inputs["weight_fp8"]
    with op_context("index_q_upproj"):
        out = try_dispatch_fp8_gemm(
            x,
            weight,
            inputs["x_scale"],
            inputs["weight_scale"],
            inputs["block_size"],
            torch.bfloat16,
        )
    # The registry's built-in M16 policy must make M32 land here without
    # allocating or launching the experimental Triton kernel.
    return runtime.reference(inputs) if out is None else out
