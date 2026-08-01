"""GLM-5.2 Masked MoE SwiGLU+Packed-UE8M0 Quant (fused region) (decode) — the one file to edit for this task.

This file is the DEFAULT candidate, not the only one: `./run.sh --candidate PATH`
tests any .py defining run(inputs), from anywhere on disk, without touching the task.
Editing this file is just the convenient path.

Run `./run.sh --describe` for the full contract. The short version:

`inputs` is the frozen dict from glm52_ops.build_inputs. The very same dict feeds
the reference, so do NOT re-quantize, re-seed, or rebuild any tensor inside
run() — that would measure a different problem than the one the gate checked.

Tensors at M=16:

    gate_up          (8, 128, 4096)           torch.bfloat16
    down_input       (8, 128, 2048)           torch.float8_e4m3fn
    down_scale_storage (8, 4, 128)              torch.int32
    masked_m         (8,)                     torch.int32

Return exactly `(down_input_fp8, packed_ue8m0_scale)`; every logical output is gated. Correctness against glm52_ops.reference on these inputs is
FlashMLA's three-layer check: matching inf/nan positions, then every element
abs_err < abs_tol OR rel_err < 0.0157, then DeepGEMM's calc_diff
<= 5e-06. `./run.sh --describe` prints all of it.

The production output buffer(s) are pre-allocated and may be written in place,
but the harness poisons every observable row before calling run(). Returning a
shared buffer unwritten therefore FAILS.

Baseline to beat: the call below, timed CUPTI cold-L2 on these same inputs.

    ./run.sh
"""
from __future__ import annotations

from sglang.jit_kernel.dsv4 import (
    silu_and_mul_masked_post_quant,
)


def run(inputs: dict):
    # Production masked DeepEP activation: fused SwiGLU and packed-UE8M0 quant.
    silu_and_mul_masked_post_quant(
        inputs["gate_up"],
        inputs["down_input"],
        inputs["down_scale_storage"],
        inputs["group_size"],
        inputs["masked_m"],
        scale_ue8m0=True,
        topk=inputs["topk"],
        transposed=True,
    )
    return inputs["down_input"], inputs["down_scale_storage"].transpose(-1, -2)
