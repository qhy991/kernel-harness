"""GLM-5.2 MoE Up Projection (prefill) — the one file to edit for this task.

Run `./run.sh --describe` for the full contract. The short version:

`inputs` is the frozen dict from glm52_ops.build_inputs. The very same dict feeds
the reference, so do NOT re-quantize, re-seed, or rebuild any tensor inside
run() — that would measure a different problem than the one the gate checked.

Tensors at M=1024:

    x_fp8            (8, 1152, 6144)          torch.float8_e4m3fn
    x_scale          (8, 1152, 48)            torch.float32
    w_fp8            (8, 2048, 6144)          torch.float8_e4m3fn
    w_scale          (8, 16, 48)              torch.float32
    masked_m         (8,)                     torch.int32
    out              (8, 1152, 2048)          torch.bfloat16

Return the output. Correctness is cosine >= 0.999 AND
rel_l2 <= 0.044721 against glm52_ops.reference on these inputs;
cosine alone is scale-blind, so both gate.

`inputs["out"]` is pre-allocated and may be written in place, but the harness
NaN-poisons it before calling run(): returning it unwritten FAILS.

Baseline to beat: the call below, timed CUPTI cold-L2 on these same inputs.

    ./run.sh --repeat 3
"""
from __future__ import annotations

import deep_gemm


def run(inputs: dict):
    # Starting point: the reference call itself — correct, speedup ~1.0. Replace it.
    out = inputs["out"]
    deep_gemm.fp8_m_grouped_gemm_nt_masked(
        (inputs["x_fp8"], inputs["x_scale"]),
        (inputs["w_fp8"], inputs["w_scale"]),
        out, inputs["masked_m"], inputs["expected_m"],
    )
    return out
