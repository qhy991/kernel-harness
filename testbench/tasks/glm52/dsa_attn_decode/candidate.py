"""GLM-5.2 DSA Sparse Attention (decode) — the one file to edit for this task.

This file is the DEFAULT candidate, not the only one: `./run.sh --candidate PATH`
tests any .py defining run(inputs), from anywhere on disk, without touching the task.
Editing this file is just the convenient path.

Run `./run.sh --describe` for the full contract. The short version:

`inputs` is the frozen dict from glm52_ops.build_inputs. The very same dict feeds
the reference, so do NOT re-quantize, re-seed, or rebuild any tensor inside
run() — that would measure a different problem than the one the gate checked.

Tensors at M=16:

    q                (16, 64, 576)            torch.bfloat16
    kv               (65536, 1, 576)          torch.bfloat16
    indices          (16, 1, 2048)            torch.int32

Return the output. Correctness against glm52_ops.reference on these inputs is
FlashMLA's three-layer check: matching inf/nan positions, then every element
abs_err < abs_tol OR rel_err < 0.0157, then DeepGEMM's calc_diff
<= 5e-06. `./run.sh --describe` prints all of it.

Baseline to beat: the call below, timed CUPTI cold-L2 on these same inputs.

    ./run.sh
"""
from __future__ import annotations

from testbench.harness import glm52_ops


OP = 'dsa_attn'
PHASE = 'decode'


def run(inputs: dict):
    # Starting point: the reference call itself — correct, speedup ~1.0. Replace it.
    # glm52_ops.reference dispatches to the per-platform production kernel
    # (flash_mla_sparse_fwd on CUDA; aiter/torch sparse MLA on ROCm).
    return glm52_ops.reference(OP, PHASE, inputs)
