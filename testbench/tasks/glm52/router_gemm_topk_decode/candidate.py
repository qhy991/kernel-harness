"""GLM-5.2 MoE Router GEMM+Sigmoid/Correction/Top-K (fusion region) (decode) — the one file to edit for this task.

This file is the DEFAULT candidate, not the only one: `./run.sh --candidate PATH`
tests any .py defining run(inputs), from anywhere on disk, without touching the task.
Editing this file is just the convenient path.

Run `./run.sh --describe` for the full contract. The short version:

`inputs` is the frozen dict from glm52_ops.build_inputs. The very same dict feeds
the reference, so do NOT re-quantize, re-seed, or rebuild any tensor inside
run() — that would measure a different problem than the one the gate checked.

Tensors at M=16:

    hidden           (16, 6144)               torch.bfloat16
    router_weight_bf16 (256, 6144)              torch.bfloat16
    router_weight_fp32 (256, 6144)              torch.float32
    correction_bias  (256,)                   torch.float32

Return exactly `(topk_weights, topk_ids)`; every logical output is gated. Correctness against glm52_ops.reference on these inputs is
FlashMLA's three-layer check: matching inf/nan positions, then every element
abs_err < abs_tol OR rel_err < 0.0157, then DeepGEMM's calc_diff
<= 5e-06. `./run.sh --describe` prints all of it.

Baseline to beat: the call below, timed CUPTI cold-L2 on these same inputs.

    ./run.sh
"""
from __future__ import annotations

import torch.nn.functional as F
from sglang.jit_kernel.moe_fused_gate import moe_fused_gate


def run(inputs: dict):
    # Open production boundary: FP32 router projection, then fused sigmoid/top-k.
    logits = F.linear(inputs["hidden"].float(), inputs["router_weight_fp32"])
    return moe_fused_gate(
        logits,
        inputs["correction_bias"],
        inputs["topk"],
        scoring_func="sigmoid",
        renormalize=inputs["renormalize"],
        routed_scaling_factor=inputs["routed_scaling_factor"],
        apply_routed_scaling_factor_on_output=False,
    )
