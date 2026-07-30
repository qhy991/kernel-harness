"""GLM-5.2 Routed Expert Gate+Up/Down Total (prefill) — the one file to edit for this task.

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

    hidden_states    (1024, 6144)             torch.bfloat16
    w1               (8, 4096, 6144)          torch.float8_e4m3fnuz
    w2               (8, 6144, 2048)          torch.float8_e4m3fnuz
    topk_weights     (1024, 8)                torch.float32
    topk_ids         (1024, 8)                torch.int32
    router_logits    (1024, 8)                torch.float32
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


Optimization (this candidate)
-----------------------------
The reference is sglang's Triton fused MoE (`fused_moe(..., use_fp8_w8a8=True)`).
Its correctness gate (`calc_diff <= 1e-5`) is so tight that the fp8 intermediate
activation saturates hard (act amax ~76k vs FP8_MAX 224, ~45% of values clamp),
so any *reimplementation* of the fp8 kernels diverges by ~3e-2 in calc_diff — even
when fed the reference's exact intermediate tensors. The only correctness-safe
lever is to drive the reference's OWN Triton kernels with a numerically-identical
but faster launch config.

Prefill here is dense-degenerate (top_k == num_experts == 8, topk_ids == arange(8),
so every one of the M tokens routes to every expert), and compute-bound (fp8
GEMM; primary_util == MFU). Two bit-exact launch-config levers on the reference's
own resolved config, both on the token (M) output dimension, stack:

  * BLOCK_SIZE_M = 256 (resolver default 128) — a larger token-tile amortises the
    shared fp8 weight loads over 2x the rows and fills the MFMA pipeline better;
    measured ~8-9% uniformly across M in {1024, 2048, 4096} (512 overflows LDS).
  * GROUP_SIZE_M — the L2-swizzle grouping is too coarse for these fused-MoE grids:
    device-kernel medians favor GROUP_SIZE_M = 1 at M <= 1024, = 4 at M = 2048, and
    = 16 at M >= 4096 (see _pick_group_size_m; the M=4096 winner is 16 because GM=4
    softened to reference parity under the pinned-CK reference).

Neither lever changes which per-output-element fp32 K-accumulation runs (that is
BLOCK_SIZE_K, left untouched) — they only resize / reorder which (m, n) output tile
each Triton program owns — so the result is bit-identical to the reference (measured
calc_diff == 0.0 for M in {1024, 2048, 4096}, for the GROUP_SIZE_M sweep at each M,
and for BLOCK_SIZE_M in {128, 256}).

We reuse the reference's own resolved config (so BLOCK_SIZE_N/K, num_warps,
num_stages, waves_per_eu stay exactly as tuned) and override BLOCK_SIZE_M and
GROUP_SIZE_M on both the gemm1 and down configs, then call sglang's
`_fused_moe_kernel_sequence` directly. Any deviation from the expected dense/fp8
setup, or any API/shape surprise, falls back to the untouched reference.
"""
from __future__ import annotations

import torch

from testbench.harness import glm52_ops


OP = 'moe_total'
PHASE = 'prefill'


def _pick_group_size_m(m: int) -> int:
    # Measured bit-exact device-kernel winners for the dense fp8 fused-MoE grid
    # under the pinned-CK reference:
    #   M <= 1024 -> GROUP_SIZE_M = 1   (~1.08x over the resolver default)
    #   M == 2048 -> GROUP_SIZE_M = 4   (~1.04x)
    #   M >= 4096 -> GROUP_SIZE_M = 16  (~1.02x; GM=4 softened to reference parity
    #                                    here, so the M=4096 winner moved to 16)
    if m <= 1024:
        return 1
    if m < 4096:
        return 4
    return 16


def _fast_moe_total_prefill(inputs: dict):
    """Bit-exact fast path: reference Triton kernels with a tuned BLOCK_SIZE_M
    and GROUP_SIZE_M.

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

    gm = _pick_group_size_m(M)
    bm = 256
    # Two bit-exact launch-config levers, both on the M (token) output dimension:
    #   * BLOCK_SIZE_M: the token-tile size. The resolver defaults to 128 here; a
    #     larger 256-row A tile amortises the shared weight loads over 2x the tokens
    #     and packs the MFMA pipeline better (measured ~8-9% uniformly across all
    #     three prefill shapes; 512 overflows LDS).
    #   * GROUP_SIZE_M: the L2-swizzle grouping (per-M, see _pick_group_size_m).
    # Both only reorder / resize which (m, n) output tile each Triton program owns
    # (program->tile mapping, grid, and moe_align padding); NEITHER touches the
    # per-output-element fp32 K-accumulation (that is BLOCK_SIZE_K, left untouched),
    # so the result is bit-identical to the reference (measured calc_diff == 0.0 for
    # M in {1024, 2048, 4096}). If the resolver already picked exactly this
    # (BLOCK_SIZE_M, GROUP_SIZE_M), there is nothing to gain -> defer.
    already = cfg.get("BLOCK_SIZE_M") == bm and cfg.get("GROUP_SIZE_M") == gm and (
        down_cfg is None
        or (down_cfg.get("BLOCK_SIZE_M") == bm and down_cfg.get("GROUP_SIZE_M") == gm)
    )
    if already:
        raise RuntimeError("resolver config already optimal; use reference")
    cfg["BLOCK_SIZE_M"] = bm
    cfg["GROUP_SIZE_M"] = gm
    if down_cfg is not None:
        down_cfg["BLOCK_SIZE_M"] = bm
        down_cfg["GROUP_SIZE_M"] = gm

    # Alignment must use the SAME (overridden) gemm1 BLOCK_SIZE_M the grid tiles on.
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
        return _fast_moe_total_prefill(inputs)
    except Exception:
        # Correctness first: on anything unexpected, fall back to the reference.
        return glm52_ops.reference(OP, PHASE, inputs)
