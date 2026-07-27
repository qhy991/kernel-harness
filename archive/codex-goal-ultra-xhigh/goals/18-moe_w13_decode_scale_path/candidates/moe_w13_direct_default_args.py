"""Diagnostic direct launch with DeepGEMM's default scale-cast argument.

Comparing this arm with ``moe_w13_direct_launch.py`` separates SGLang wrapper
overhead from the effect of explicitly declaring that both scale tensors are
already packed int32 UE8M0. It bypasses the production wrapper and is therefore
not deployable.
"""

from __future__ import annotations

import deep_gemm
import torch


CANDIDATE_METADATA = {
    "role": "non_promotable_direct_default_args_decomposition",
    "operator": "fused_w13",
    "packed_int32_ue8m0_only": True,
    "disable_ue8m0_cast": False,
    "purpose": "separate wrapper overhead from packed-scale declaration",
}


def run(inputs, runtime):
    family = runtime.workload.family
    if family not in {"moe_grouped_masked", "moe_w13_handoff"}:
        raise RuntimeError(
            "direct-default decomposition supports W13 component/handoff only"
        )
    if (
        tuple(inputs["activation_fp8"].shape) != (32, 1024, 6144)
        or tuple(inputs["weight_fp8"].shape) != (32, 4096, 6144)
    ):
        raise RuntimeError("direct-default launch received a non-W13 shape")
    if (
        inputs["activation_scale"].dtype != torch.int32
        or inputs["weight_scale"].dtype != torch.int32
    ):
        raise RuntimeError("direct-default launch requires packed int32 scales")

    out = inputs.get("out")
    if out is None:
        out = torch.empty(
            (
                inputs["activation_fp8"].shape[0],
                inputs["activation_fp8"].shape[1],
                inputs["weight_fp8"].shape[1],
            ),
            device=inputs["activation_fp8"].device,
            dtype=torch.bfloat16,
        )
    deep_gemm.fp8_m_grouped_gemm_nt_masked(
        (inputs["activation_fp8"], inputs["activation_scale"]),
        (inputs["weight_fp8"], inputs["weight_scale"]),
        out,
        inputs["masked_m"],
        inputs["expected_m"],
    )
    return runtime.moe_result(out, inputs)
