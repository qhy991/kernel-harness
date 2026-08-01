"""Decode-only GLM-5.2 router region candidate.

For graph bucket M<=16, replace the FP32 ``F.linear`` launch with SGLang's
production-shaped BF16-input/FP32-output router kernel.  The sigmoid/correction/
top-k consumer remains the production Triton kernel.  M>16 deliberately falls
back to the reference projection because the JIT router kernel's contract ends
at 16 tokens.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.jit_kernel.dsv3_router_gemm import dsv3_router_gemm
from sglang.jit_kernel.moe_fused_gate import moe_fused_gate


def run(inputs: dict):
    hidden = inputs["hidden"]
    if hidden.shape[0] <= 16:
        logits = dsv3_router_gemm(
            hidden,
            inputs["router_weight_bf16"],
            out_dtype=torch.float32,
        )
    else:
        logits = F.linear(hidden.float(), inputs["router_weight_fp32"])
    return moe_fused_gate(
        logits,
        inputs["correction_bias"],
        inputs["topk"],
        scoring_func="sigmoid",
        renormalize=inputs["renormalize"],
        routed_scaling_factor=inputs["routed_scaling_factor"],
        apply_routed_scaling_factor_on_output=False,
    )
