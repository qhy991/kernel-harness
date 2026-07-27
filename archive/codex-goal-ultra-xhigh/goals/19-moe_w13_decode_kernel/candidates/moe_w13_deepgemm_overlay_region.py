"""Swap only W13 to the pinned overlay inside the local MoE compute region."""

from __future__ import annotations

from serving_native.candidates import moe_w13_deepgemm_overlay


def run(inputs, runtime):
    result = runtime.run_moe_compute_region(
        inputs,
        w13_launcher=lambda current: moe_w13_deepgemm_overlay.launch(
            current, runtime
        ),
    )
    return result.observed
