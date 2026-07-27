"""Exercise SGLang's fail-closed GLM-5.2 W2 registry integration.

The serving-native runner keeps the process reference policy at
``SGLANG_GLM52_OPT=0``.  This diagnostic arm switches only the candidate call
onto the same registry lookup, wrapper, op tag, decode mode, and M allowlist
used by an explicitly enabled serving bucket.  The toggle itself is a tiny
Python-side conservative tax inside candidate timing.

This is not an EP8 promotion artifact.  The reached wrapper still owns stock
fallback for recipes or overlap, and production enablement additionally needs
the external graph/overlap/full-region/end-to-end gates.
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
    "operator": "moe_down_proj",
    "profile": "serving_safe",
    "allowed_decode_m": [16, 32],
    "packed_int32_ue8m0_only": True,
    "overlap_recipe_policy": "wrapper_fails_closed_to_stock",
    "promotion_boundary": "requires_external_ep8_acceptance",
}

_ACTIVE = False


def _candidate_enabled() -> bool:
    return _ACTIVE


# Install the same resolved serving policy once, outside the timed window.
# The active bit remains false for every stock/reference arm.
dispatch.config.is_enabled = _candidate_enabled
registry.profile_name = lambda: "serving_safe"
registry.opt_ops_allowlist = lambda: frozenset({"moe_down_proj"})
registry.opt_m_buckets = lambda: {
    "moe_down_proj": frozenset(CANDIDATE_METADATA["allowed_decode_m"])
}


def _freeze_decode_context(runtime: Any) -> None:
    decode_m = int(runtime.workload.params["decode_m"])
    if decode_m not in CANDIDATE_METADATA["allowed_decode_m"]:
        raise RuntimeError(f"unsupported registry-integration M bucket: {decode_m}")
    current_mode = get_forward_mode()
    current_m = get_forward_m()
    if current_mode is None and current_m is None:
        # The first candidate correctness call is outside timing; subsequent
        # paired arms reuse this exact production-forward context.
        set_forward_mode(ForwardMode.DECODE, decode_m)
        return
    if current_mode != ForwardMode.DECODE or current_m != decode_m:
        raise RuntimeError(
            "registry-integration candidate observed conflicting forward context: "
            f"mode={current_mode}, M={current_m}, expected decode/M{decode_m}"
        )


def run(inputs: dict[str, Any], runtime: Any):
    if runtime.workload.family not in {"moe_grouped_masked", "moe_w2_handoff"}:
        raise RuntimeError(
            "moe_w2_registry_integration supports isolated W2 and W2 handoff only"
        )
    if inputs["weight_scale"].dtype != runtime.torch.int32:
        raise RuntimeError("registry integration requires packed int32 weight scales")
    if runtime.workload.family == "moe_grouped_masked" and (
        inputs["activation_scale"].dtype != runtime.torch.int32
    ):
        raise RuntimeError("registry integration requires packed int32 activation scales")

    _freeze_decode_context(runtime)
    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError("nested registry-integration candidate invocation")
    _ACTIVE = True
    try:
        # This deliberately enters the normal SGLang wrapper, including its
        # compatibility check and stock fallback, rather than calling the
        # lower-level kernel directly.
        return runtime.reference(inputs)
    finally:
        _ACTIVE = False
