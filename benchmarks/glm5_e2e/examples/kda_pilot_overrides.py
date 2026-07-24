"""kda-pilot e2e overrides — wire the two landing-capable optimized kernels
(moe GROUP_SIZE_M + index_score fp8_mqa_logits BLOCK_KV) into the live model.

Both patches are BIT-EXACT (they change only launch/tiling config, never the
math) and each falls back to the untouched path on any surprise. A call counter
is printed at process exit so we can prove the fast path actually fired (and
report no-ops honestly, e.g. fp8_mqa_logits is only reached when DSA uses sparse
indexing — kv_len > index_topk=2048).

dsa (MLA) is deliberately NOT wired: its candidate regresses ~4x vs the ASM
baseline and the sparse-MLA wrapper is off the e2e prefill hot path.

Tuning constants come from:
  archive/0720-Best-GLM-52/lichangye/index_score_prefill/candidate/candidate.py
  candidates/lichangye-hardened/moe_total_prefill/candidate.py
"""
from __future__ import annotations

import atexit
import torch


# ── counters (honesty gate: prove the fast path fired) ──────────────────────
_STATS = {
    "mqa_fast": 0, "mqa_fallback": 0,
    "moe_override": 0, "moe_defer": 0,
}


def _dump_stats():
    print(f"[overrides] call stats: {_STATS}", flush=True)


atexit.register(_dump_stats)


# ── index_score: force BLOCK_KV=256 on aiter fp8_mqa_logits (gfx942) ─────────
def _make_fast_fp8_mqa_logits(mod):
    """Return a drop-in fp8_mqa_logits that mirrors mod's gfx942 Triton branch
    but launches with BLOCK_KV=256, num_stages=1 (bit-exact: tiling only)."""
    _orig = mod.fp8_mqa_logits
    _kernel = mod._fp8_mqa_logits_kernel
    _TARGET_BLOCK_KV = 256
    _TARGET_NUM_STAGES = 1

    def fp8_mqa_logits(Q, KV, kv_scales, weights, cu_starts, cu_ends,
                       clean_logits=True):
        try:
            use_gluon = mod.TRITON_GE_36 and mod._gluon_fp8_mqa_logits_kernel is not None
            if mod.arch != "gfx942" or use_gluon:
                raise RuntimeError("not the validated gfx942 Triton path")
            if Q.ndim != 3:
                raise RuntimeError("unexpected Q rank")
            if weights.ndim == 3 and weights.shape[-1] == 1:
                weights = weights.squeeze(-1)
            seq_len, num_heads, head_size = Q.shape
            seq_len_kv = KV.shape[0]
            if num_heads & (num_heads - 1) or head_size & (head_size - 1):
                raise RuntimeError("num_heads/head_size not power of 2")
            # only take the win where the heuristic would have dropped to the
            # small tile (otherwise defer to the untouched reference).
            if mod._gfx942_tile_fits_lds(block_kv=128, head_size=head_size,
                                         num_stages=2, occupancy=2):
                raise RuntimeError("heuristic already uses the large tile")

            aligned = 256
            skv_aligned = (seq_len_kv + aligned - 1) // aligned * aligned
            if clean_logits:
                logits = torch.full((seq_len, skv_aligned), -float("inf"),
                                    dtype=torch.float32, device=Q.device)[:, :seq_len_kv]
            else:
                logits = torch.empty((seq_len, skv_aligned),
                                     dtype=torch.float32, device=Q.device)[:, :seq_len_kv]

            _fnuz = torch.float8_e4m3fnuz
            convert_q = Q.dtype != _fnuz
            convert_kv = KV.dtype != _fnuz
            scale_mul = 1.0
            if convert_q:
                scale_mul *= 2.0
                Q = (Q.to(torch.float32) * 0.5).to(_fnuz)
            if convert_kv:
                scale_mul *= 2.0
                KV = (KV.to(torch.float32) * 0.5).to(_fnuz)
            if scale_mul != 1.0:
                kv_scales = kv_scales.to(torch.float32) * scale_mul

            matrix_instr_nonkdim = 16 if seq_len <= 1024 else 32
            sq_s, sq_h, sq_d = Q.stride()
            skv_s, skv_d = KV.stride()
            sw_s, sw_h = weights.stride()
            sl_s, sl_k = logits.stride()

            _kernel[(seq_len,)](
                Q_ptr=Q, KV_ptr=KV, kv_scales_ptr=kv_scales, weights_ptr=weights,
                cu_start_ptr=cu_starts, cu_end_ptr=cu_ends, logits_ptr=logits,
                seq_len=seq_len, seq_len_kv=seq_len_kv,
                NUM_HEADS=num_heads, HEAD_SIZE=head_size,
                stride_q_s=sq_s, stride_q_h=sq_h, stride_q_d=sq_d,
                stride_kv_s=skv_s, stride_kv_d=skv_d,
                stride_w_s=sw_s, stride_w_h=sw_h,
                stride_logits_s=sl_s, stride_logits_k=sl_k,
                BLOCK_KV=_TARGET_BLOCK_KV, num_warps=4,
                num_stages=_TARGET_NUM_STAGES, waves_per_eu=2,
                matrix_instr_nonkdim=matrix_instr_nonkdim,
            )
            _STATS["mqa_fast"] += 1
            return logits
        except Exception:
            _STATS["mqa_fallback"] += 1
            return _orig(Q, KV, kv_scales, weights, cu_starts, cu_ends,
                         clean_logits=clean_logits)

    return fp8_mqa_logits


# ── moe: force tuned GROUP_SIZE_M in the fused-MoE config resolver ───────────
def _pick_group_size_m(m: int):
    if m <= 1024:
        return 1
    elif m <= 2048:
        return 4
    return None  # M>=4096 regresses; defer to the resolver default


def _make_group_size_override(orig_fn):
    def try_get_optimal_moe_config(*args, **kwargs):
        result = orig_fn(*args, **kwargs)
        # M is the 5th positional arg (w1_shape, w2_shape, top_k, dtype, M, ...)
        M = kwargs.get("M")
        if M is None and len(args) >= 5:
            M = args[4]
        gm = _pick_group_size_m(int(M)) if M is not None else None
        if gm is None:
            _STATS["moe_defer"] += 1
            return result
        # result is either config-dict or (config, (down_config, max_block_m))
        if isinstance(result, tuple):
            cfg = result[0]
            down = result[1][0] if isinstance(result[1], (tuple, list)) else None
        else:
            cfg, down = result, None
        applied = False
        if isinstance(cfg, dict) and cfg.get("GROUP_SIZE_M") != gm:
            cfg["GROUP_SIZE_M"] = gm
            applied = True
        if isinstance(down, dict) and down.get("GROUP_SIZE_M") != gm:
            down["GROUP_SIZE_M"] = gm
            applied = True
        _STATS["moe_override" if applied else "moe_defer"] += 1
        return result

    return try_get_optimal_moe_config


def register():
    from operator_overrides import patch

    changes = []

    # index_score — patch the module sglang imports from (top-level, distinct
    # object from aiter.ops.triton.attention.fp8_mqa_logits).
    import aiter.ops.triton.fp8_mqa_logits as mqa_mod
    fast = _make_fast_fp8_mqa_logits(mqa_mod)
    changes.append(patch("aiter.ops.triton.fp8_mqa_logits.fp8_mqa_logits", fast))

    # moe — patch try_get_optimal_moe_config where fused_moe.py looks it up.
    import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fm
    wrapped = _make_group_size_override(fm.try_get_optimal_moe_config)
    changes.append(patch(
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.try_get_optimal_moe_config",
        wrapped,
    ))

    return changes
