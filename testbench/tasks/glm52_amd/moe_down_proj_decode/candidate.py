"""GLM-5.2 MoE Down Projection (decode) — the one file to edit for this task.

Platform: AMD (this task is on the amd-only tree
`testbench/tasks/glm52_amd/`; the sibling platform lives under
`glm52_cuda/`).

This file is the DEFAULT candidate, not the only one: `./run.sh --candidate PATH`
tests any .py defining run(inputs), from anywhere on disk, without touching the task.
Editing this file is just the convenient path.

Run `./run.sh --describe` for the full contract. The short version:

`inputs` is the frozen dict from glm52_ops.build_inputs. The very same dict feeds
the reference, so do NOT re-quantize, re-seed, or rebuild any tensor inside
run() — that would measure a different problem than the one the gate checked.

Tensors at M=16:

    x_fp8            (8, 128, 2048)           torch.float8_e4m3fnuz
    x_scale          (8, 1)                   torch.float32
    w_fp8            (8, 6144, 2048)          torch.float8_e4m3fnuz
    w_scale          (8, 1)                   torch.float32
    masked_m         (8,)                     torch.int32
    out              (8, 128, 6144)           torch.bfloat16

Return the output. Correctness against glm52_ops.reference on these inputs is
FlashMLA's three-layer check: matching inf/nan positions, then every element
abs_err < abs_tol OR rel_err < 0.0157, then DeepGEMM's calc_diff
<= 5e-06. `./run.sh --describe` prints all of it.

`inputs["out"]` is pre-allocated and may be written in place, but the harness
NaN-poisons it before calling run(): returning it unwritten FAILS.

Baseline to beat: the call below, timed CUPTI cold-L2 on these same inputs.

    ./run.sh
"""
from __future__ import annotations

from aiter.fused_moe import fused_moe
from aiter import ActivationType, QuantType
import torch


def run(inputs: dict):
    # aiter's fused_moe consumes per-tensor scales on gfx942. If your aiter build
    # uses a different signature, fall back to glm52_ops.reference in the outer
    # dispatch and shape-branch here for the wins.
    out = fused_moe(
        inputs["x_fp8"], inputs["w_fp8"], None,
        inputs["masked_m"], inputs["masked_m"],
        activation=ActivationType.Silu, quant_type=QuantType.per_1x128,
        w1_scale=inputs["x_scale"], w2_scale=inputs["w_scale"],
    )
    inputs["out"].copy_(out)
    return inputs["out"]
