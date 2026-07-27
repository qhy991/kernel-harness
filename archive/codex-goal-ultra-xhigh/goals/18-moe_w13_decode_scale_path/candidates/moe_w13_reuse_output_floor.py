"""Diagnostic lower bound for removing only the W13 output allocation.

The buffer contains no cached input data, but one process-global tensor is not
safe across concurrent requests, streams, or independent CUDA-graph instances.
This is therefore decomposition evidence only and must never be enabled.
"""

from __future__ import annotations

import torch


CANDIDATE_METADATA = {
    "role": "non_promotable_output_reuse_floor",
    "operator": "fused_w13",
    "risk": "process-global output aliases concurrent streams and graph instances",
}

_OUTPUTS = {}


def _output(inputs):
    activation = inputs["activation_fp8"]
    key = (
        activation.device.type,
        activation.device.index,
        activation.shape[0],
        activation.shape[1],
        inputs["weight_fp8"].shape[1],
    )
    out = _OUTPUTS.get(key)
    if out is None:
        out = torch.empty(
            (key[2], key[3], key[4]),
            device=activation.device,
            dtype=torch.bfloat16,
        )
        _OUTPUTS[key] = out
    return out


def run(inputs, runtime):
    if runtime.workload.family != "moe_w13_handoff":
        raise RuntimeError("W13 output-reuse floor supports only the handoff task")
    return runtime.run_w13_handoff(inputs, out=_output(inputs))
