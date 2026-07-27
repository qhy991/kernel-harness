"""DeepGEMM grouped-W13 configuration probe with stock-safe restoration.

This candidate changes only process-local DeepGEMM launch heuristics for the
duration of its own call.  The reference call immediately before/after it sees
the original values.  It consumes the packed-int32 UE8M0 ABI and preserves the
FP8/FP4 recipes supplied by the serving-native workload.
"""

from __future__ import annotations

import os

import deep_gemm


_BLOCK_M = int(os.environ.get("GLM52_W13_BLOCK_M", "128"))
_NUM_SMS = int(os.environ.get("GLM52_W13_NUM_SMS", "148"))
_PDL = os.environ.get("GLM52_W13_PDL", "0").lower() in ("1", "true", "yes", "on")
# The stock wrapper compiles with "nk". DeepGEMM's process-local host cache
# does not include the mutable block-M heuristic in its key, so a configuration
# probe must use a distinct compiled-dims key after the reference has run.
_COMPILED_DIMS = os.environ.get("GLM52_W13_COMPILED_DIMS", "mnk")

if _BLOCK_M not in (32, 64, 96, 128, 160, 192, 224):
    raise ValueError(f"unsupported GLM52_W13_BLOCK_M={_BLOCK_M}")
if _NUM_SMS <= 0 or _NUM_SMS % 2:
    raise ValueError(f"GLM52_W13_NUM_SMS must be a positive even number: {_NUM_SMS}")


def run(inputs, runtime):
    old_block_m = deep_gemm.get_mk_alignment_for_contiguous_layout()
    old_num_sms = deep_gemm.get_num_sms()
    old_pdl = deep_gemm.get_pdl()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(_BLOCK_M)
        deep_gemm.set_num_sms(_NUM_SMS)
        deep_gemm.set_pdl(_PDL)
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (inputs["activation_fp8"], inputs["activation_scale"]),
            (inputs["weight_fp8"], inputs["weight_scale"]),
            inputs["out"],
            inputs["masked_m"],
            inputs["expected_m"],
            recipe_a=inputs["recipe_a"],
            recipe_b=inputs["recipe_b"],
            compiled_dims=_COMPILED_DIMS,
            disable_ue8m0_cast=True,
        )
    finally:
        deep_gemm.set_pdl(old_pdl)
        deep_gemm.set_num_sms(old_num_sms)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_block_m)

    valid = [
        inputs["out"][expert, : int(count)]
        for expert, count in enumerate(inputs["masked_m"].tolist())
    ]
    return valid
