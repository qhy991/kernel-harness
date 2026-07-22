"""GLM-5.2 Routed Expert Gate+Up/Down Total (decode) — the one file to edit for this task.

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

Tensors at M=16:

    hidden_states    (16, 6144)               torch.bfloat16
    w1               (8, 4096, 6144)          torch.float8_e4m3fn
    w2               (8, 6144, 2048)          torch.float8_e4m3fn
    topk_weights     (16, 8)                  torch.float32
    topk_ids         (16, 8)                  torch.int32
    router_logits    (16, 8)                  torch.float32
    w1_scale         (8,)                     torch.float32
    w2_scale         (8,)                     torch.float32
    a1_scale         (1,)                     torch.float32
    a2_scale         (1,)                     torch.float32

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
    # Starting point: the SGLang fused MoE reference — correct, speedup ~1.0.
    # Replace with a fused MoE kernel to beat it (e.g. a Triton fused_moe with
    # your own tuning table).
    return glm52_ops.reference("moe_total", "prefill", inputs)
