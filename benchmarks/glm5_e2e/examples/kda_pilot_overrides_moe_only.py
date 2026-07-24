"""kda-pilot e2e overrides — MOE-ONLY variant for the decode arm.

Identical moe GROUP_SIZE_M override as kda_pilot_overrides.py, but WITHOUT the
index_score (fp8_mqa_logits) patch. Rationale:

  * index_score is a confirmed NO-OP end-to-end (fired 0x in both prefill A/B
    runs — DSA uses dense/fused MLA attention at these lengths, and decode uses
    aiter mla_decode_fwd, so the fp8-MQA indexer is off the hot path).
  * importing `aiter.ops.triton.fp8_mqa_logits` at register() time touches
    aiter's gluon path, which raises `aiter gluon kernels require triton>=3.6.0`
    on this node (triton 3.5.1). On the decode boot that per-rank sitecustomize
    failure desynced one TP rank and deadlocked the `active_ranks` collective
    barrier (7/8 ranks hung). Dropping the no-op index_score patch removes the
    fragile import so all 8 ranks load overrides identically → clean boot.

The moe patch is bit-exact (GROUP_SIZE_M changes only L2 tiling, not the fp32
K-accumulation) and falls back to the resolver default on any surprise.
"""
from __future__ import annotations

import atexit


_STATS = {"moe_override": 0, "moe_defer": 0}


def _dump_stats():
    print(f"[overrides] call stats: {_STATS}", flush=True)


atexit.register(_dump_stats)


def _pick_group_size_m(m: int):
    if m <= 1024:
        return 1
    elif m <= 2048:
        return 4
    return None  # M>=4096 regresses; defer to the resolver default


def _make_group_size_override(orig_fn):
    def try_get_optimal_moe_config(*args, **kwargs):
        result = orig_fn(*args, **kwargs)
        M = kwargs.get("M")
        if M is None and len(args) >= 5:
            M = args[4]
        gm = _pick_group_size_m(int(M)) if M is not None else None
        if gm is None:
            _STATS["moe_defer"] += 1
            return result
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
    import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fm
    wrapped = _make_group_size_override(fm.try_get_optimal_moe_config)
    changes.append(patch(
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.try_get_optimal_moe_config",
        wrapped,
    ))
    return changes
