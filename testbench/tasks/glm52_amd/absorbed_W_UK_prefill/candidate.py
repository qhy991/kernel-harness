"""GLM-5.2 Absorbed W_UK BMM (prefill) — the one file to edit for this task.

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

Tensors at M=1024:

    A_fp8            (64, 1024, 192)          torch.float8_e4m3fnuz
    B_fp8            (64, 192, 512)           torch.float8_e4m3fnuz
    A_scale          (1,)                     torch.float32
    B_scale          (1,)                     torch.float32

Return the output. Correctness against glm52_ops.reference on these inputs is
FlashMLA's three-layer check: matching inf/nan positions, then every element
abs_err < abs_tol OR rel_err < 0.0157, then DeepGEMM's calc_diff
<= 5e-06. `./run.sh --describe` prints all of it.

Baseline to beat: the call below, timed CUPTI cold-L2 on these same inputs.

    ./run.sh
"""
from __future__ import annotations

import torch

# absorbed_W_UK / _UV BMM on MI300X. There is no fused FP8 BMM on gfx942 — sglang's
# production path loops per-head torch._scaled_mm (hipBLASLt). This candidate is
# the reference call itself; replace with a batched kernel (MFMA head-folding) to win.
from testbench.harness import glm52_ops


def run(inputs: dict):
    return glm52_ops.reference("absorbed_W_UK", "prefill", inputs)  # phase inferred by shape
