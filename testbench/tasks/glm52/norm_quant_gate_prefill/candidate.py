"""GLM-5.2 Residual+RMSNorm+FP8Quant -> MoE Gate Projection (fusion region) (prefill) — the one file to edit for this task.

This file is the DEFAULT candidate, not the only one: `./run.sh --candidate PATH`
tests any .py defining run(inputs), from anywhere on disk, without touching the task.
Editing this file is just the convenient path.

Run `./run.sh --describe` for the full contract. The short version:

`inputs` is the frozen dict from glm52_ops.build_inputs. The very same dict feeds
the reference, so do NOT re-quantize, re-seed, or rebuild any tensor inside
run() — that would measure a different problem than the one the gate checked.

Tensors at M=1024:

    hidden           (1024, 6144)             torch.bfloat16
    residual         (1024, 6144)             torch.bfloat16
    norm_weight      (6144,)                  torch.bfloat16
    w_fp8            (2048, 6144)             torch.float8_e4m3fn
    w_scale          (2048, 12)               torch.int32
    out              (1024, 2048)             torch.bfloat16
    _hidden0         (1024, 6144)             torch.bfloat16
    _residual0       (1024, 6144)             torch.bfloat16

Return exactly `(out, residual)`; both tensors are gated. Correctness against glm52_ops.reference on these inputs is
FlashMLA's three-layer check: matching inf/nan positions, then every element
abs_err < abs_tol OR rel_err < 0.0157, then DeepGEMM's calc_diff
<= 5e-06. `./run.sh --describe` prints all of it.

`inputs["out"]` is pre-allocated and may be written in place, but the harness
NaN-poisons it before calling run(): returning it unwritten FAILS.

Baseline to beat: the call below, timed CUPTI cold-L2 on these same inputs.

    ./run.sh
"""
from __future__ import annotations

import deep_gemm
from sgl_kernel import fused_add_rmsnorm
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8


def run(inputs: dict):
    """Starting point: the production three-kernel sequence; replace to optimize.

    QKV must return normalized BF16 because GLM-5.2's DSA indexer consumes it.
    The post-attention gate site has no such side consumer and returns two tensors.
    """
    hidden, residual = inputs["hidden"], inputs["residual"]
    fused_add_rmsnorm(hidden, residual, inputs["norm_weight"], inputs["eps"])
    x_fp8, x_scale = sglang_per_token_group_quant_fp8(
        hidden,
        inputs["group_size"],
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    out = inputs["out"]
    deep_gemm.fp8_gemm_nt(
        (x_fp8, x_scale), (inputs["w_fp8"], inputs["w_scale"]), out)
    if inputs["requires_bf16_output"]:
        return out, residual, hidden
    return out, residual
