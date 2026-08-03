"""GLM-5.2 Routed Expert Gate+Up/Down Total (decode) candidate.

This keeps the official SGLang fused MoE math path. The optimization is to
replace SGLang's ROCm graph-safe PyTorch alignment chain with AITER's Triton
alignment kernel, then run SGLang's own fused MoE sequence with a small-M
BLOCK_SIZE_M cap. Routing is computed from the current topk_ids every call; no
route metadata or output is cached.
"""
from __future__ import annotations

import torch
import triton

from testbench.harness import glm52_ops


OP = "moe_total"
PHASE = "decode"


def _pick_block_size_m(m: int) -> int:
    if m <= 16:
        return 16
    if m <= 32:
        return 32
    raise RuntimeError("no safe generic win at this M; use reference")


def _cap_block_size_m(cfg: dict | None, m: int) -> dict:
    if cfg is None:
        cfg = {}
    else:
        cfg = dict(cfg)
    cap = _pick_block_size_m(m)
    if cfg.get("BLOCK_SIZE_M", cap) > cap:
        cfg["BLOCK_SIZE_M"] = cap
    return cfg


def _aiter_triton_align(topk_ids: torch.Tensor, block_size: int, num_experts: int):
    from aiter.ops.triton.moe.moe_align_block_size import moe_align_block_size_triton

    if topk_ids.numel() < num_experts + 1:
        max_num_tokens_padded = topk_ids.numel() * block_size
    else:
        max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)

    sorted_token_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_token_ids.fill_(topk_ids.numel())
    expert_ids = torch.empty(
        (triton.cdiv(max_num_tokens_padded, block_size),),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    num_tokens_post_padded = torch.empty(
        (1,), dtype=torch.int32, device=topk_ids.device
    )

    moe_align_block_size_triton(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def _fast_moe_total_decode(inputs: dict):
    hidden = inputs["hidden_states"]
    w1 = inputs["w1"]
    w2 = inputs["w2"]
    topk_weights = inputs["topk_weights"]
    topk_ids = inputs["topk_ids"]
    w1_scale = inputs["w1_scale"]
    w2_scale = inputs["w2_scale"]
    a1_scale = inputs["a1_scale"]
    a2_scale = inputs["a2_scale"]

    M = hidden.shape[0]
    E = w1.shape[0]
    topk = topk_ids.shape[1]

    import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fm
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig

    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)

    runner_cfg = inputs.get("moe_runner_config")
    if runner_cfg is None:
        runner_cfg = MoeRunnerConfig(
            **inputs["moe_config_kwargs"],
            params_dtype=hidden.dtype,
        )
    filter_expert = (
        runner_cfg.num_experts is None
        or runner_cfg.num_experts != runner_cfg.num_local_experts
    )
    if filter_expert:
        raise RuntimeError("filtered experts require reference alignment")

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
    cfg = _cap_block_size_m(cfg, M)
    down_cfg = _cap_block_size_m(down_cfg if down_cfg is not None else cfg, M)
    down_cfg["BLOCK_SIZE_M"] = cfg["BLOCK_SIZE_M"]
    down_moe_use_tma = (
        fm._down_moe_use_tma()
        and down_cfg is not None
        and down_cfg.pop("USE_TMA", False)
    )

    sorted_token_ids, expert_ids, num_tokens_post_padded = _aiter_triton_align(
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
        down_moe_use_tma,
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
        activation=runner_cfg.activation,
        is_gated=runner_cfg.is_gated,
        no_combine=runner_cfg.no_combine,
        inplace=runner_cfg.inplace,
        apply_router_weight_on_input=runner_cfg.apply_router_weight_on_input,
        routed_scaling_factor=runner_cfg.routed_scaling_factor,
        gemm1_alpha=runner_cfg.gemm1_alpha,
        gemm1_limit=runner_cfg.gemm1_clamp_limit,
        filter_expert=filter_expert,
        hooks=None,
        swiglu_limit=runner_cfg.swiglu_limit,
    )


def run(inputs: dict):
    try:
        return _fast_moe_total_decode(inputs)
    except Exception:
        return glm52_ops.reference(OP, PHASE, inputs)
