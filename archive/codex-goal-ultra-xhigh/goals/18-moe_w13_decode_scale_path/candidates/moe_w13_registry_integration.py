"""Exercise SGLang's fail-closed fused-W13 decode registry path.

The benchmark process remains globally pinned to ``SGLANG_GLM52_OPT=0`` for
reference calls.  This module installs the same serving-safe profile and M
allowlist once, then flips a process-local predicate only around candidate
calls.  Recipe- or overlap-bearing calls still fall back through the wrapper.
"""

from __future__ import annotations

from typing import Any

from sglang.srt.layers.glm52_opt import dispatch, registry
from sglang.srt.layers.glm52_opt.context import (
    get_forward_m,
    get_forward_mode,
    set_forward_mode,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


CANDIDATE_METADATA = {
    "role": "production_registry_integration_trial",
    "operator": "moe_gate_proj",
    "profile": "serving_safe",
    "allowed_decode_m": [16, 32],
    "packed_int32_ue8m0_only": True,
    "fallback": "stock wrapper for recipes, overlap, unsupported ABI or M",
}

_ACTIVE = False


def _candidate_enabled() -> bool:
    return _ACTIVE


dispatch.config.is_enabled = _candidate_enabled
registry.profile_name = lambda: "serving_safe"
registry.opt_ops_allowlist = lambda: frozenset({"moe_gate_proj"})
registry.opt_m_buckets = lambda: {
    "moe_gate_proj": frozenset(CANDIDATE_METADATA["allowed_decode_m"])
}


def _freeze_decode_context(runtime: Any) -> None:
    decode_m = int(runtime.workload.params["decode_m"])
    if decode_m not in CANDIDATE_METADATA["allowed_decode_m"]:
        raise RuntimeError(f"unsupported W13 registry M bucket: {decode_m}")
    mode = get_forward_mode()
    forward_m = get_forward_m()
    if mode is None and forward_m is None:
        set_forward_mode(ForwardMode.DECODE, decode_m)
        return
    if mode != ForwardMode.DECODE or forward_m != decode_m:
        raise RuntimeError(
            "W13 registry observed conflicting forward context: "
            f"mode={mode}, M={forward_m}, expected decode/M{decode_m}"
        )


def run(inputs, runtime):
    if runtime.workload.family not in {
        "moe_grouped_masked",
        "moe_w13_handoff",
        "moe_compute_region",
        "deepep_ll_moe_region",
    }:
        raise RuntimeError("W13 registry trial received an unsupported family")
    experts = int(runtime.workload.params["experts_per_rank"])
    if tuple(inputs["weight_fp8"].shape) != (experts, 4096, 6144):
        raise RuntimeError("W13 registry trial received a non-W13 weight")
    if inputs["weight_scale"].dtype != runtime.torch.int32:
        raise RuntimeError("W13 registry requires packed int32 UE8M0 weights")

    _freeze_decode_context(runtime)
    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError("nested W13 registry candidate invocation")
    _ACTIVE = True
    try:
        return runtime.reference(inputs)
    finally:
        _ACTIVE = False
