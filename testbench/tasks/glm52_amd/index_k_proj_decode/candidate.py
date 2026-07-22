"""GLM-5.2 Indexer K Projection (decode) — the one file to edit for this task.

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

    x_fp8            (16, 6144)               torch.float8_e4m3fnuz
    x_scale          (16, 48)                 torch.float32
    w_fp8            (128, 6144)              torch.float8_e4m3fnuz
    w_scale          (1, 48)                  torch.float32
    out              (16, 128)                torch.bfloat16

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

from aiter.ops.triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale
import torch


def run(inputs: dict):
    # Starting point: aiter's Triton blockscale FP8 GEMM (a fallback path — sglang's
    # default gfx942 dispatch is bpreshuffle_asm/ck, but the ASM build is not always
    # available on every node). Replace with a direct MFMA kernel to beat it.
    out = gemm_a8w8_blockscale(inputs["x_fp8"], inputs["w_fp8"],
                               inputs["x_scale"], inputs["w_scale"],
                               dtype=torch.bfloat16)
    inputs["out"].copy_(out)
    return inputs["out"]
