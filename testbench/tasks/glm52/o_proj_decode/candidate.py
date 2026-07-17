"""GLM-5.2 O Projection (decode) — the one file to edit for this task.

    out[M, 6144] = x_fp8[M, 16384] @ w_fp8[6144, 16384].T        M in {16, 32}

Run `./run.sh --describe` for the full contract: tensor table, baseline, gates.
The short version:

`inputs` is the frozen dict from glm52_ops.build_inputs. The very same dict feeds
the reference, so do NOT re-quantize, re-seed, or rebuild any tensor inside
run() — that would measure a different problem than the one the gate checked.

    x_fp8    [M, 16384]      float8_e4m3fn   per-token ue8m0 scales
    x_scale  [M, 128]        float32         mn-major, TMA-aligned
    w_fp8    [6144, 16384]   float8_e4m3fn   128x128 per-block ue8m0 scales
    w_scale  [48, 128]       float32
    out      [M, 6144]       bfloat16        pre-allocated, NaN-poisoned

Return the output. Writing into `inputs["out"]` and returning it is fine; so is
returning a fresh tensor. Returning `inputs["out"]` *without writing it* is not:
the harness poisons that buffer with NaN before calling run(), precisely so a
no-op cannot inherit the reference's answer.

Baseline to beat: the deep_gemm call below, timed cold-L2 on these same inputs.

    ./run.sh --repeat 3
"""
from __future__ import annotations

import deep_gemm


def run(inputs: dict):
    # Starting point: the production DeepGEMM path — i.e. the reference itself.
    # As written this is correct and scores speedup ~1.0. Replace it.
    out = inputs["out"]
    deep_gemm.fp8_gemm_nt(
        (inputs["x_fp8"], inputs["x_scale"]),
        (inputs["w_fp8"], inputs["w_scale"]),
        out,
    )
    return out
