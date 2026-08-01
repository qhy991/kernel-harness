"""GLM-5.2 DSA Indexer Q RoPE+Quant+Head-Gate Scale (fused region) (decode) — the one file to edit for this task.

This file is the DEFAULT candidate, not the only one: `./run.sh --candidate PATH`
tests any .py defining run(inputs), from anywhere on disk, without touching the task.
Editing this file is just the convenient path.

Run `./run.sh --describe` for the full contract. The short version:

`inputs` is the frozen dict from glm52_ops.build_inputs. The very same dict feeds
the reference, so do NOT re-quantize, re-seed, or rebuild any tensor inside
run() — that would measure a different problem than the one the gate checked.

Tensors at M=16:

    q_input          (16, 32, 128)            torch.bfloat16
    head_gate        (16, 32)                 torch.bfloat16
    cos_sin_cache    (65553, 64)              torch.float32
    positions        (16,)                    torch.int64

Return exactly `(q_fp8, head_gate_with_q_scale)`; every logical output is gated. Correctness against glm52_ops.reference on these inputs is
FlashMLA's three-layer check: matching inf/nan positions, then every element
abs_err < abs_tol OR rel_err < 0.0157, then DeepGEMM's calc_diff
<= 5e-06. `./run.sh --describe` prints all of it.

Baseline to beat: the call below, timed CUPTI cold-L2 on these same inputs.

    ./run.sh
"""
from __future__ import annotations

from sglang.jit_kernel.dsv4 import (
    fused_q_indexer_rope_first_quant,
)


def run(inputs: dict):
    # Starting point is the already-fused production Q preparation kernel.
    return fused_q_indexer_rope_first_quant(
        inputs["q_input"],
        inputs["head_gate"],
        inputs["weight_scale"],
        inputs["cos_sin_cache"],
        inputs["positions"],
    )
