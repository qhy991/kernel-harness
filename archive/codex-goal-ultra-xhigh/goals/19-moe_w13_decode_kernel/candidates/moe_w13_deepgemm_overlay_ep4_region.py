"""Diagnostic EP4 full-region candidate using the pinned W13 source overlay.

This deliberately bypasses the production E32/slab1024 integration guard for
the different E64/slab512 EP4 geometry.  It is useful only as four-rank
diagnostic evidence and cannot enable or validate the EP8 production path.
"""

from __future__ import annotations

from serving_native.candidates.moe_w13_deepgemm_overlay import _load_overlay


def run(inputs, runtime):
    p = runtime.workload.params
    if (
        runtime.workload.family != "deepep_ll_moe_region"
        or runtime.workload.world_size != 4
        or p["experts_per_rank"] != 64
        or p["expert_slab"] != 512
        or p["decode_m"] not in (16, 32)
        or p["expected_m"] not in (3, 5)
    ):
        raise RuntimeError("EP4 W13 overlay received an unsupported diagnostic ABI")

    deep_gemm = _load_overlay()

    def launch(lhs, rhs, out, masked_m, expected_m):
        return deep_gemm.fp8_m_grouped_gemm_nt_masked(
            lhs,
            rhs,
            out,
            masked_m,
            expected_m,
            compiled_dims="nk",
            disable_ue8m0_cast=True,
        )

    return runtime.run_deepep_ll_moe_region(inputs, w13_launcher=launch)
