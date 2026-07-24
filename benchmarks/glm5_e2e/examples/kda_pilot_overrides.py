"""kda-pilot e2e overrides — wire the landing-capable optimized kernels into the
live GLM-5.2 model, each mirroring an accepted op-level candidate EXACTLY so the
GATE-2 A/B tests the validated win (and no more):

  1. index_score  — aiter fp8_mqa_logits with BLOCK_KV=256 (mirror index_score
     candidate 68c9b6). Only reached when DSA uses sparse indexing
     (kv_len > index_topk=2048); otherwise the counter records a fallback.
  2. moe prefill  — fused-MoE GROUP_SIZE_M override (mirror moe_total_prefill
     candidate daddf4 / sha 221718a3): M<=1024->1, M<4096->4, M>=4096->16.
     Applied only for the validated prefill regime (M>=1024); the e2e resolver
     M == num_tokens == input_len, so in={1024,2048,4096} hit exactly {1,4,16}.
  3. moe decode   — fused-MoE BLOCK_SIZE_M shrink (mirror moe_total_decode
     candidate cc6538): only when topk==E (dense) and M<=32. Real GLM-5.2 decode
     is sparse (topk=8 << E) with num_tokens=bs, so this DEFERS at e2e — the
     counter's defer_reason proves the op-level decode win does not translate to
     the sparse e2e shape (an honest Amdahl no-op, not a failure).

All three are BIT-EXACT (launch/tiling config only, never BLOCK_SIZE_K or the
math) and every path falls back to the untouched kernel on any surprise. A
per-M / per-reason call counter is printed at each rank's process exit to prove
which fast path fired and to report no-ops honestly.

register() wires each override best-effort and records the outcome (_REGISTER,
printed at each rank's exit). The index_score patch imports aiter's triton
fp8_mqa_logits, which touches aiter's gluon path; that import raises
"aiter gluon kernels require triton>=3.6.0" on a triton<3.6 node (this node has
historically carried triton 3.5.1; the current e2e venv is 3.6.0, where it
imports cleanly and the gfx942 BLOCK_KV=256 fast path is live). Guarding the
import/patch means a failure there can NEVER abort registration of the (working,
bit-exact) MoE overrides, nor desync a TP rank at the active_ranks barrier the
way an unguarded per-rank sitecustomize failure would.

dsa (MLA) is deliberately NOT wired: no bit-exact launch-config trick beats the
production paged sparse-MLA baseline (torch-gather reimpl regresses ~4x), and the
sparse-MLA wrapper is off the e2e prefill hot path.

Tuning provenance:
  index_score  runs/glm52/index_score_prefill/20260724T102649Z-68c9b6/candidate.py
  moe prefill  runs/glm52/moe_total_prefill/20260723T043507Z-daddf4/candidate.py
  moe decode   runs/glm52/moe_total_decode/20260724T102632Z-cc6538/candidate.py
"""
from __future__ import annotations

import atexit
import torch


# ── counters (honesty gate: prove the intended fast path fired, per shape) ───
_STATS = {
    "mqa_fast": 0, "mqa_fallback": 0,
    "moe_gm_applied": 0,   # prefill GROUP_SIZE_M override applied
    "moe_bm_applied": 0,   # decode BLOCK_SIZE_M shrink applied (dense topk==E)
    "moe_defer": 0,        # neither validated override applied
}
# per-M breakdown so a coarse aggregate cannot hide which shape actually fired.
_MOE_BY_M: dict = {}

# register() outcome per override (honesty: prove what actually wired, and why
# not, without ever letting a wiring failure abort the model boot).
_REGISTER = {"index_score": "not-attempted", "moe": "not-attempted"}


def _note_moe(m: int, gm=None, bm=False, defer_reason=None):
    rec = _MOE_BY_M.setdefault(int(m), {"count": 0, "gm": None, "bm": False,
                                        "defer_reason": None})
    rec["count"] += 1
    if gm is not None:
        rec["gm"] = gm
    if bm:
        rec["bm"] = True
    if defer_reason is not None and rec["defer_reason"] is None:
        rec["defer_reason"] = defer_reason


def _dump_stats():
    print(f"[overrides] register: {_REGISTER}", flush=True)
    print(f"[overrides] call stats: {_STATS}", flush=True)
    for m in sorted(_MOE_BY_M):
        print(f"[overrides] moe M={m}: {_MOE_BY_M[m]}", flush=True)


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


# ── moe: force tuned GROUP_SIZE_M / BLOCK_SIZE_M in the fused-MoE resolver ────
def _pick_group_size_m(m: int):
    # mirror moe_total_prefill candidate daddf4 (sha 221718a3): GM=16 at M>=4096.
    if m <= 1024:
        return 1
    if m < 4096:
        return 4
    return 16


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _pick_block_size_m(m: int) -> int:
    # mirror moe_total_decode candidate cc6538: match block to per-expert rows.
    return max(16, min(128, _next_pow2(m)))


def _split_config(result):
    # resolver returns either a config-dict or (config, (down_config, max_bm)).
    if isinstance(result, tuple):
        cfg = result[0]
        down = result[1][0] if isinstance(result[1], (tuple, list)) else None
    else:
        cfg, down = result, None
    return cfg, down


def _make_group_size_override(orig_fn):
    def try_get_optimal_moe_config(*args, **kwargs):
        result = orig_fn(*args, **kwargs)
        try:
            # signature: (w1_shape, w2_shape, top_k, dtype, M, ...)
            w1_shape = kwargs.get("w1_shape", args[0] if len(args) >= 1 else None)
            topk = kwargs.get("top_k", args[2] if len(args) >= 3 else None)
            M = kwargs.get("M", args[4] if len(args) >= 5 else None)
            if M is None:
                _STATS["moe_defer"] += 1
                return result
            M = int(M)
            E = int(w1_shape[0]) if w1_shape is not None else None
            topk = int(topk) if topk is not None else None
            cfg, down = _split_config(result)
            if not isinstance(cfg, dict):
                _STATS["moe_defer"] += 1
                _note_moe(M, defer_reason="no-config-dict")
                return result

            # (1) decode dense path — mirror cc6538 exactly (topk==E and M<=32);
            #     shrink only (never grow), leaving BLOCK_SIZE_K/N untouched.
            if topk is not None and E is not None and topk == E and M <= 32:
                bm = _pick_block_size_m(M)
                applied = False
                if cfg.get("BLOCK_SIZE_M", 0) > bm:
                    cfg["BLOCK_SIZE_M"] = bm
                    applied = True
                if isinstance(down, dict) and down.get("BLOCK_SIZE_M", 0) > bm:
                    down["BLOCK_SIZE_M"] = bm
                    applied = True
                if applied:
                    _STATS["moe_bm_applied"] += 1
                    _note_moe(M, bm=True)
                else:
                    _STATS["moe_defer"] += 1
                    _note_moe(M, defer_reason="bm-already-small")
                return result

            # (2) prefill path — mirror daddf4; only the validated M>=1024 regime.
            if M >= 1024:
                gm = _pick_group_size_m(M)
                applied = False
                if cfg.get("GROUP_SIZE_M") != gm:
                    cfg["GROUP_SIZE_M"] = gm
                    applied = True
                if isinstance(down, dict) and down.get("GROUP_SIZE_M") != gm:
                    down["GROUP_SIZE_M"] = gm
                    applied = True
                if applied:
                    _STATS["moe_gm_applied"] += 1
                    _note_moe(M, gm=gm)
                else:
                    _STATS["moe_defer"] += 1
                    _note_moe(M, gm=gm, defer_reason="gm-already-set")
                return result

            # (3) neither validated regime (small-M sparse decode, etc.) — defer.
            _STATS["moe_defer"] += 1
            reason = ("sparse-decode:topk!=E"
                      if (topk is not None and E is not None and topk != E)
                      else "unvalidated-small-M")
            _note_moe(M, defer_reason=reason)
            return result
        except Exception as exc:  # never let the override crash the model
            _STATS["moe_defer"] += 1
            _note_moe(-1, defer_reason=f"exc:{type(exc).__name__}")
            return result

    return try_get_optimal_moe_config


def register():
    from operator_overrides import patch

    changes = []

    # index_score — patch the module sglang imports from (top-level, distinct
    # object from aiter.ops.triton.attention.fp8_mqa_logits). GUARDED: importing
    # aiter's triton fp8_mqa_logits touches the gluon path, which raises
    # "aiter gluon kernels require triton>=3.6.0" on a triton<3.6 node. A raise
    # here must NOT abort the (working, bit-exact) MoE wiring below nor desync a
    # TP rank, so we isolate it and record the outcome instead of propagating.
    try:
        import aiter.ops.triton.fp8_mqa_logits as mqa_mod
        fast = _make_fast_fp8_mqa_logits(mqa_mod)
        changes.append(patch("aiter.ops.triton.fp8_mqa_logits.fp8_mqa_logits", fast))
        _REGISTER["index_score"] = "wired"
    except Exception as exc:
        # off the e2e hot path anyway (index_score fired 0x in prior A/B runs);
        # skipping it only forfeits a no-op, never a validated e2e win.
        _REGISTER["index_score"] = f"skipped:{type(exc).__name__}:{exc}"

    # moe — patch try_get_optimal_moe_config where fused_moe.py looks it up.
    # Independently guarded so it wires even if index_score above is skipped.
    try:
        import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fm
        wrapped = _make_group_size_override(fm.try_get_optimal_moe_config)
        changes.append(patch(
            "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.try_get_optimal_moe_config",
            wrapped,
        ))
        _REGISTER["moe"] = "wired"
    except Exception as exc:
        _REGISTER["moe"] = f"skipped:{type(exc).__name__}:{exc}"

    return changes
