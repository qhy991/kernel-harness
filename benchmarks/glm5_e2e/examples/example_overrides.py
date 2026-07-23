"""Example overrides file — showing two patch idioms side by side.

Delete or copy this file when writing your own; it exists purely as a reference.

Two ways to declare patches — pick whichever reads better in your case:

  A. `patch(dotted, new_value)` for one-off, fully-qualified attribute swaps
  B. `register_from_dict({short_name: fn})` for well-known targets

Both routes end up calling `setattr(module, attr, new)` on the sglang / aiter
module that OWNS the attribute — that's the only reliable way to make every
call site pick up the new implementation, because sglang holds the module
reference, not a captured local of the old function.
"""
from __future__ import annotations


def register():
    # Import the harness helpers (already on PYTHONPATH by the bench script).
    from operator_overrides import patch, register_from_dict

    changes: list[dict] = []

    # ── Idiom A: patch a fully-qualified attribute ──────────────────────────
    # Example: replace sglang's FP8 dense GEMM dispatcher with your own kernel.
    # Uncomment and point at your implementation.
    #
    # from my_kernel_pkg import faster_ar_gemm
    # changes.append(patch(
    #     "sglang.srt.layers.quantization.fp8_utils.aiter_w8a8_block_fp8_linear",
    #     faster_ar_gemm,
    # ))

    # ── Idiom B: named registry (KNOWN_OVERRIDE_TARGETS) ────────────────────
    # Same effect, but names shielded from module renames upstream.
    #
    # from my_kernel_pkg import fused_moe_v2, mla_decode_v2
    # changes.extend(register_from_dict({
    #     "aiter_fused_moe":  fused_moe_v2,
    #     "aiter_mla_decode": mla_decode_v2,
    # }).values())

    # Return list of {target, old_id, new_id} dicts so the bench log names
    # exactly what changed. Optional — returning None is fine too.
    return changes
