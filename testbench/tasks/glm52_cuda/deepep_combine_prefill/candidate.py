"""GLM-5.2 DeepEP Combine (EP MoE) (prefill) — the one file to edit for this task.

Platform: CUDA (this task is on the cuda-only tree
`testbench/tasks/glm52_cuda/`; the sibling platform lives under
`glm52_amd/`).

This file is the DEFAULT candidate, not the only one: `./run.sh --candidate PATH`
tests any .py defining run(inputs), from anywhere on disk, without touching the task.
Editing this file is just the convenient path.

Run `./run.sh --describe` for the full contract. The short version:

`inputs` is the frozen dict from glm52_ops.build_inputs. The very same dict feeds
the reference, so do NOT re-quantize, re-seed, or rebuild any tensor inside
run() — that would measure a different problem than the one the gate checked.

Tensors at M=1024:

    x                (1024, 6144)             torch.bfloat16
    topk_ids         (1024, 8)                torch.int32
    topk_weights     (1024, 8)                torch.float32

Return the output. Correctness against glm52_ops.reference on these inputs is
FlashMLA's three-layer check: matching inf/nan positions, then every element
abs_err < abs_tol OR rel_err < 0.0157, then DeepGEMM's calc_diff
<= 1e-05. `./run.sh --describe` prints all of it.

Baseline to beat: the call below, timed CUPTI cold-L2 on these same inputs.

    ./run.sh
"""
from __future__ import annotations

from testbench.harness import glm52_ops


def run(inputs: dict):
    return glm52_ops.reference("deepep_combine", "prefill", inputs)
