"""Diagnostic W13 launch floor for the native packed-UE8M0 ABI.

This bypasses SGLang's grouped-GEMM wrapper after the stock arm has compiled
and warmed the exact DeepGEMM kernel.  It intentionally drops the compile hook,
registry policy, recipes, and any future overlap return contract, so it is a
measurement floor rather than a deployable replacement.
"""

from __future__ import annotations

import deep_gemm
import torch


CANDIDATE_METADATA = {
    "role": "non_promotable_direct_launch_floor",
    "operator": "fused_w13",
    "shape": {"experts": 32, "slab": 1024, "k": 6144, "n": 4096},
    "packed_int32_ue8m0_only": True,
    "dropped_contracts": [
        "deep_gemm_execution_hook",
        "glm52_registry_policy",
        "recipe_forwarding",
        "overlap_return_contract",
    ],
}


def run(inputs, runtime):
    family = runtime.workload.family
    if family in {"moe_grouped_masked", "moe_w13_handoff", "moe_compute_region"}:
        if (
            tuple(inputs["activation_fp8"].shape) != (32, 1024, 6144)
            or tuple(inputs["weight_fp8"].shape) != (32, 4096, 6144)
        ):
            raise RuntimeError("direct W13 launch received a non-W13 shape")
        if (
            inputs["activation_scale"].dtype != torch.int32
            or inputs["weight_scale"].dtype != torch.int32
        ):
            raise RuntimeError("direct W13 launch requires packed int32 UE8M0 scales")
        if family == "moe_grouped_masked":
            return runtime.run_w13_handoff(
                inputs, direct=True, out=inputs["out"]
            )
        if family == "moe_w13_handoff":
            return runtime.run_w13_handoff(inputs, direct=True)
        return runtime.run_moe_compute_region(inputs, direct_w13=True)
    if family == "deepep_ll_moe_region":
        return runtime.run_deepep_ll_moe_region(inputs, direct_w13=True)
    raise RuntimeError(f"direct W13 launch does not support {family}")
