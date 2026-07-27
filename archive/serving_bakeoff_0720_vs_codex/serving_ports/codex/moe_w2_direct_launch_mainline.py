"""Bakeoff-compatible Codex MoE W2 direct launch for mainline serving_native.

The archived goal-06 candidate calls ``runtime.moe_result`` which only exists in
goal worktree runners. This wrapper keeps the same DeepGEMM direct launch but
returns expert-valid slices matching mainline ``Runtime.reference``.
"""

from __future__ import annotations

import deep_gemm
import torch


def run(inputs, runtime):
    family = runtime.workload.family
    if family != "moe_grouped_masked":
        raise RuntimeError(
            f"bakeoff moe_w2 wrapper supports moe_grouped_masked only, got {family}"
        )
    if (
        inputs["activation_scale"].dtype != torch.int32
        or inputs["weight_scale"].dtype != torch.int32
    ):
        raise RuntimeError("direct-launch experiment requires packed int32 UE8M0 scales")

    deep_gemm.fp8_m_grouped_gemm_nt_masked(
        (inputs["activation_fp8"], inputs["activation_scale"]),
        (inputs["weight_fp8"], inputs["weight_scale"]),
        inputs["out"],
        inputs["masked_m"],
        inputs["expected_m"],
        disable_ue8m0_cast=True,
    )
    return [
        inputs["out"][expert, : int(count)]
        for expert, count in enumerate(inputs["masked_m"].tolist())
    ]
