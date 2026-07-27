"""Diagnostic W2 launch-floor candidate for packed UE8M0, no-overlap calls.

This deliberately bypasses SGLang's grouped-GEMM wrapper after the stock arm
has compiled/warmed the exact kernel.  It is an experiment, not a promotable
implementation: active overlap, recipes, compilation hooks, and registry
dispatch remain owned by the stock wrapper.
"""

from __future__ import annotations

import deep_gemm
import torch


def _checked_direct_w2(down_input, down_scale, inputs, runtime):
    if down_scale.dtype != torch.int32 or inputs["weight_scale"].dtype != torch.int32:
        raise RuntimeError("direct-launch experiment requires packed int32 UE8M0 scales")
    out = torch.empty(
        (down_input.shape[0], down_input.shape[1], inputs["output_n"]),
        device=down_input.device,
        dtype=torch.bfloat16,
    )
    deep_gemm.fp8_m_grouped_gemm_nt_masked(
        (down_input, down_scale),
        (inputs["weight_fp8"], inputs["weight_scale"]),
        out,
        inputs["masked_m"],
        inputs["expected_m"],
        disable_ue8m0_cast=True,
    )
    return runtime.moe_result(out, inputs)


def _handoff(inputs, runtime):
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.moe.moe_runner.deep_gemm import (
        _varlen_deep_gemm_silu_mul_quant,
    )

    down_input, down_scale = _varlen_deep_gemm_silu_mul_quant(
        inputs["gateup_output"],
        inputs["masked_m"],
        group_size=inputs["group_size"],
        topk=inputs["topk"],
    )
    if not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
        raise RuntimeError("direct-launch experiment requires the packed UE8M0 ABI")
    return _checked_direct_w2(down_input, down_scale, inputs, runtime)


def run(inputs, runtime):
    family = runtime.workload.family
    if family == "moe_grouped_masked":
        if (
            inputs["activation_scale"].dtype != torch.int32
            or inputs["weight_scale"].dtype != torch.int32
        ):
            raise RuntimeError(
                "direct-launch experiment requires packed int32 UE8M0 scales"
            )
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (inputs["activation_fp8"], inputs["activation_scale"]),
            (inputs["weight_fp8"], inputs["weight_scale"]),
            inputs["out"],
            inputs["masked_m"],
            inputs["expected_m"],
            disable_ue8m0_cast=True,
        )
        return runtime.moe_result(inputs["out"], inputs)
    if family == "moe_w2_handoff":
        return _handoff(inputs, runtime)
    raise RuntimeError(
        "moe_w2_direct_launch supports only isolated grouped W2 and W2 handoff"
    )
