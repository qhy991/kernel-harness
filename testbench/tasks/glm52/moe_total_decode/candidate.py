"""GLM-5.2 Routed Expert Gate+Up/Down Total (decode) — the one file to edit for this task.

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
<= 5e-06. `./run.sh --describe` prints all of it.

Baseline to beat: the call below, timed by the selected backend protocol:
CUPTI cold-L2 device-kernel median: inputs cloned per iteration and L2 flushed before each, both outside the measured window

    ./run.sh


Optimization (this candidate)
-----------------------------
The reference is sglang's Triton fused MoE (`fused_moe(..., use_fp8_w8a8=True)`).
Its correctness gate (`calc_diff <= 5e-6`) is so tight that the fp8 intermediate
activation saturates hard (act amax ~76k vs FP8_MAX 224, ~45% of values clamp),
so any *reimplementation* of the fp8 kernels diverges by ~3e-2 in calc_diff — even
when fed the reference's exact intermediate tensors. The only correctness-safe
lever is to drive the reference's OWN Triton kernels with a numerically-identical
but faster launch config.

Decode here is dense-degenerate: top_k == num_experts == 8 and topk_ids == arange(8),
so every one of the M tokens routes to every expert. The reference picks
BLOCK_SIZE_M = 128 (or 64), but at decode M <= 64 each expert only owns M rows, so
`moe_align_block_size` pads every expert block up to BLOCK_SIZE_M -> up to ~8x wasted
padded-row work in both fp8 GEMMs. Shrinking BLOCK_SIZE_M to ~match M removes that
waste. BLOCK_SIZE_M changes only the tile grid, NOT the per-output-element fp32
K-accumulation order (that is BLOCK_SIZE_K, left untouched), so the result is
bit-identical to the reference (measured calc_diff == 0.0 for M in {1,4,8,16,32,64}).

We reuse the reference's own resolved config (so BLOCK_SIZE_N/K, GROUP_SIZE_M,
num_warps, num_stages stay exactly as tuned) and override only BLOCK_SIZE_M, then
call sglang's `_fused_moe_kernel_sequence` directly. Any deviation from the expected
dense/fp8 setup, or any API/shape surprise, falls back to the untouched reference.
"""
from __future__ import annotations

import torch

from testbench.harness import glm52_ops


OP = 'moe_total'
PHASE = 'decode'


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _pick_block_size_m(m: int) -> int:
    # Match BLOCK_SIZE_M to the per-expert row count (dense: M rows/expert),
    # clamped to the Triton-supported [16, 128] range.
    return max(16, min(128, _next_pow2(m)))


def _fast_moe_total_decode(inputs: dict):
    """Bit-exact fast path: reference Triton kernels with a shrunk BLOCK_SIZE_M.

    Raises on any unexpected condition so run() can fall back to the reference.
    """
    hidden = inputs["hidden_states"]
    w1 = inputs["w1"]
    w2 = inputs["w2"]
    topk_weights = inputs["topk_weights"]
    topk_ids = inputs["topk_ids"]
    w1_scale = inputs["w1_scale"]
    w2_scale = inputs["w2_scale"]
    a1_scale = inputs["a1_scale"]
    a2_scale = inputs["a2_scale"]

    E, _, _ = w1.shape
    M = hidden.shape[0]
    topk = topk_ids.shape[1]

    # Only take the fast path for the dense-degenerate routing this task uses;
    # otherwise defer to the reference (correctness first).
    if topk != E:
        raise RuntimeError("non-dense routing; use reference")

    # Shrinking BLOCK_SIZE_M only pays off while the per-expert row count (M) is
    # far below the reference block: at M <= 32 the default block pads 4-8x, so a
    # smaller block is a clean device-kernel win. By M >= 64 the padding is <=2x
    # and the smaller block loses on occupancy (measured 0.993x), so defer to the
    # reference there rather than take a marginal regression.
    if M > 32:
        raise RuntimeError("no beneficial block-size shrink at this M; use reference")

    import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fm
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)

    cfg, (down_cfg, _) = fm.try_get_optimal_moe_config(
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2]),
        topk,
        "fp8_w8a8",
        M,
        block_shape=None,
        per_channel_quant=False,
        return_down_config=True,
    )
    cfg = dict(cfg)
    down_cfg = dict(down_cfg) if down_cfg is not None else None

    bm = _pick_block_size_m(M)
    # Never grow the block (that would only add padding / slow down); only shrink.
    cfg["BLOCK_SIZE_M"] = min(cfg["BLOCK_SIZE_M"], bm)
    if down_cfg is not None:
        down_cfg["BLOCK_SIZE_M"] = min(down_cfg["BLOCK_SIZE_M"], bm)

    # Alignment must use the (shrunk) gemm1 BLOCK_SIZE_M.
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, cfg["BLOCK_SIZE_M"], E
    )

    return fm._fused_moe_kernel_sequence(
        hidden,
        w1,
        w2,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        cfg,
        down_cfg,
        False,  # down_moe_use_tma: resolver returns TMA disabled on this fp8 path
        b1=None,
        b2=None,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=None,
        w2_zp=None,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=None,
        activation="silu",
        is_gated=True,
        no_combine=False,
        inplace=False,
        apply_router_weight_on_input=False,
        routed_scaling_factor=None,
        gemm1_alpha=None,
        gemm1_limit=None,
        filter_expert=True,
        hooks=None,
        swiglu_limit=None,
    )


def run(inputs: dict):
    try:
        return _fast_moe_total_decode(inputs)
    except Exception:
        # Correctness first: on anything unexpected, fall back to the reference.
        return glm52_ops.reference(OP, PHASE, inputs)
