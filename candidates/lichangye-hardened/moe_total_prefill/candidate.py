"""GLM-5.2 Routed Expert Gate+Up/Down Total (prefill) — hardened candidate.

Optimization: bit-exact Triton GROUP_SIZE_M override on the reference's own kernels.
See archive/0720-Best-GLM-52/lichangye/moe_total_prefill/ for original rationale.

Hardening (fixes from Phase C1):
  1. Multi-path import resilient to sglang internal reorganisation (source vs conda).
  2. Signal-based timeout watchdog (120s) so a fallback never hangs the harness.
  3. Conservative M4096 handling — skip override where it regresses (~4%).
  4. ImportError → RuntimeError (fail fast, never silent hang).
"""
from __future__ import annotations

import importlib
import signal
import torch

from testbench.harness import glm52_ops


OP = "moe_total"
PHASE = "prefill"

# ── Robust multi-path import ──────────────────────────────────────────────────
# sglang reorganised its MoE internals across versions. We try each known path
# and fail cleanly rather than letting ImportError propagate to a hanging fallback.

_fm = None  # fused_moe module
_moe_align = None  # moe_align_block_size function

_IMPORT_PATHS = [
    # Source sglang (20fc529ab+) / recent dev builds:
    (
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe",
        "sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size",
    ),
    # Conda sglang 0.5.9 (reorganised):
    (
        "sglang.srt.layers.moe.fused_moe_triton.fused_moe",
        "sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size",
    ),
    # Older sglang (pre moe_runner split):
    (
        "sglang.srt.layers.moe.fused_moe_triton.fused_moe",
        "sglang.srt.layers.moe.fused_moe_triton.fused_moe",
    ),
]


def _resolve_imports():
    """Attempt each known import path. Returns (fm_module, align_fn) or raises."""
    global _fm, _moe_align
    if _fm is not None:
        return _fm, _moe_align

    last_err = None
    for fm_path, align_path in _IMPORT_PATHS:
        try:
            fm_mod = importlib.import_module(fm_path)
            align_mod = importlib.import_module(align_path)
            # Verify the critical private API exists
            if not hasattr(fm_mod, "_fused_moe_kernel_sequence"):
                continue
            if not hasattr(fm_mod, "try_get_optimal_moe_config"):
                continue
            align_fn = getattr(align_mod, "moe_align_block_size", None)
            if align_fn is None:
                continue
            _fm = fm_mod
            _moe_align = align_fn
            return _fm, _moe_align
        except Exception as e:
            last_err = e
            continue
    raise ImportError(
        f"Cannot find sglang fused_moe with _fused_moe_kernel_sequence in any "
        f"known path. Last error: {last_err!r}"
    )


# ── Timeout watchdog ──────────────────────────────────────────────────────────

class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("reference fallback exceeded timeout")


# ── GROUP_SIZE_M tuning ───────────────────────────────────────────────────────

def _pick_group_size_m(m: int) -> int | None:
    """Return tuned GROUP_SIZE_M, or None if this shape should skip the override.

    Measured bit-exact device-kernel winners for the dense fp8 fused-MoE grid:
      M <= 1024 -> GROUP_SIZE_M = 1  (~11% win over resolver default GM=32)
      M == 2048 -> GROUP_SIZE_M = 4  (~3% win)
      M >= 4096 -> None (skip — override regresses ~4% at this shape)
    """
    if m <= 1024:
        return 1
    elif m <= 2048:
        return 4
    else:
        # M4096 regresses with GM=4; defer to the reference's own config.
        return None


def _fast_moe_total_prefill(inputs: dict):
    """Bit-exact fast path: reference Triton kernels with a tuned GROUP_SIZE_M."""
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

    # Only take the fast path for the dense-degenerate routing this task uses.
    if topk != E:
        raise RuntimeError("non-dense routing; use reference")

    gm = _pick_group_size_m(M)
    if gm is None:
        raise RuntimeError(f"M={M} skipped (override regresses); use reference")

    fm, moe_align_block_size = _resolve_imports()

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

    # GROUP_SIZE_M only changes L2-tiling, not fp32 K-accumulation → bit-exact.
    if cfg.get("GROUP_SIZE_M") == gm and (
        down_cfg is None or down_cfg.get("GROUP_SIZE_M") == gm
    ):
        raise RuntimeError("resolver GROUP_SIZE_M already optimal; use reference")
    cfg["GROUP_SIZE_M"] = gm
    if down_cfg is not None:
        down_cfg["GROUP_SIZE_M"] = gm

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
        False,  # down_moe_use_tma
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


_FALLBACK_TIMEOUT_S = 120  # 2 min max; reference should complete in < 5s


def run(inputs: dict):
    try:
        return _fast_moe_total_prefill(inputs)
    except _TimeoutError:
        raise  # propagate — harness should see a real failure, not silent hang
    except ImportError as e:
        # sglang private API not available — fail fast with a clear message.
        raise RuntimeError(
            f"moe_total_prefill candidate: sglang private API unavailable ({e}). "
            f"Cannot fall back safely."
        ) from e
    except Exception:
        # For other exceptions (non-dense routing, GM already optimal, M4096 skip),
        # fall back to reference with a timeout watchdog.
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(_FALLBACK_TIMEOUT_S)
        try:
            result = glm52_ops.reference(OP, PHASE, inputs)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        return result
