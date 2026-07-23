"""GLM-5.2 TP AllGather (token axis) (decode) — the one file to edit for this task.

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

    x                (16, 6144)               torch.bfloat16

Return the output. Correctness against glm52_ops.reference on these inputs is
FlashMLA's three-layer check: matching inf/nan positions, then every element
abs_err < abs_tol OR rel_err < 0.0157, then DeepGEMM's calc_diff
<= 1e-05. `./run.sh --describe` prints all of it.

Baseline to beat: the call below, timed CUPTI cold-L2 on these same inputs.

    ./run.sh
"""
from __future__ import annotations

import torch
import torch.distributed as dist


def run(inputs: dict):
    """MI300X TP AllGather reference — RCCL all_gather across ranks."""
    x = inputs["x"]
    ws = inputs["world_size"]
    gather = [torch.empty_like(x) for _ in range(ws)]
    dist.all_gather(gather, x)
    return torch.cat(gather, dim=0)
