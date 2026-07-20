"""GLM-5.2 MoE Down Projection (prefill) — masked FP8 grouped GEMM candidate.

The compute is DeepGEMM's masked grouped FP8 GEMM
(`deep_gemm.fp8_m_grouped_gemm_nt_masked`, E=8, K=2048, N=6144). The reference is
the same call, so the only headroom the caller can touch is DeepGEMM's own
per-shape configuration surface: programmatic dependent launch (`set_pdl`), the
SM count (`set_num_sms`), tensor-core utilization (`set_tc_util`), and which GEMM
dimensions are baked in at compile time (`compiled_dims`). Tile / stage / cluster
/ swizzle are chosen by a compiled `get_best_config` and are not caller-tunable in
this build.

Everything here is lossless and stateless: the frozen inputs are consumed as-is,
the provided `out` is written in place, `masked_m` semantics are preserved, and no
tensor is re-quantized, re-seeded, cached, or repacked across calls. The only
per-shape decision is which lossless knob values to pass; the programmatic-launch
flag is restored after the call so the process-global DeepGEMM state does not
drift across shapes.

Per-shape knob choice comes from a cold-L2 device-kernel sweep on B200 GPU 1
(bench/sweep.py): `compiled_dims='mnk'` is fastest at the two larger shapes
(M=2048, M=4096), while at the smallest shape programmatic dependent launch is the
only lever measured outside noise. Reducing the SM count regressed every shape
(the kernel already runs one CTA per SM at ~209 KB shared memory), so the SM count
is left at the full 148 and tensor-core utilization at 100.
"""
from __future__ import annotations

import contextlib

import deep_gemm

# Map the per-expert slab capacity (a frozen property of each workload shape) to
# the lossless DeepGEMM knob values measured fastest for that shape. expected_m is
# 1152 / 2176 / 4224 for M = 1024 / 2048 / 4096.
#   pdl           -> programmatic dependent launch (set_pdl)
#   compiled_dims -> which of (m, n, k) are compile-time constants
_KNOBS_BY_EXPECTED_M = {
    1152: {"pdl": True, "compiled_dims": "nk"},
    2176: {"pdl": False, "compiled_dims": "mnk"},
    4224: {"pdl": False, "compiled_dims": "mnk"},
}
_DEFAULT_KNOBS = {"pdl": False, "compiled_dims": "nk"}


@contextlib.contextmanager
def _pdl(enabled: bool):
    """Enable programmatic dependent launch for one call, then restore the prior
    value so the process-global DeepGEMM launch state does not drift across shapes.
    """
    previous = deep_gemm.get_pdl()
    deep_gemm.set_pdl(enabled)
    try:
        yield
    finally:
        deep_gemm.set_pdl(previous)


def run(inputs: dict):
    out = inputs["out"]
    knobs = _KNOBS_BY_EXPECTED_M.get(int(inputs["expected_m"]), _DEFAULT_KNOBS)
    with _pdl(knobs["pdl"]):
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (inputs["x_fp8"], inputs["x_scale"]),
            (inputs["w_fp8"], inputs["w_scale"]),
            out, inputs["masked_m"], inputs["expected_m"],
            compiled_dims=knobs["compiled_dims"],
        )
    return out
