"""Diagnostic lower bound for removing only W2 output allocation.

The cached output contains no input data, but reusing one buffer is not safe for
concurrent batches, independent streams, or CUDA-graph instances.  This file is
therefore a decomposition experiment only and must never be enabled in serving.
"""

from __future__ import annotations

import torch


_OUTPUTS = {}


def _output(down_input, output_n: int):
    key = (
        down_input.device.type,
        down_input.device.index,
        down_input.shape[0],
        down_input.shape[1],
        output_n,
    )
    out = _OUTPUTS.get(key)
    if out is None:
        out = torch.empty(
            (down_input.shape[0], down_input.shape[1], output_n),
            device=down_input.device,
            dtype=torch.bfloat16,
        )
        _OUTPUTS[key] = out
    return out


def run(inputs, runtime):
    if runtime.workload.family != "moe_w2_handoff":
        raise RuntimeError("output-reuse floor supports only the W2 handoff workload")

    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.moe.moe_runner.deep_gemm import (
        _varlen_deep_gemm_silu_mul_quant,
    )

    down_input, down_scale = _varlen_deep_gemm_silu_mul_quant(
        inputs["gateup_output"],
        inputs["masked_m"],
        group_size=inputs["group_size"],
        topk=inputs["topk"],
    )
    if not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
        raise RuntimeError("output-reuse floor requires the packed UE8M0 ABI")
    if down_scale.dtype != torch.int32 or inputs["weight_scale"].dtype != torch.int32:
        raise RuntimeError("output-reuse floor requires packed int32 UE8M0 scales")
    out = _output(down_input, inputs["output_n"])
    from sglang.srt.layers.glm52_opt.context import op_context

    with op_context("moe_down_proj"):
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            (down_input, down_scale),
            (inputs["weight_fp8"], inputs["weight_scale"]),
            out,
            inputs["masked_m"],
            inputs["expected_m"],
            overlap_args=None,
            max_block_n=256,
            recipe_a=None,
            recipe_b=None,
        )
    return runtime.moe_result(out, inputs)
