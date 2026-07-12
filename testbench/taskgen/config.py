"""Canonical model configs — the single source of truth for every task's dims.

Each config is grounded in the model's authoritative source (see per-field notes).
Family builders read these; nothing hardcodes a dim twice.
"""
from pathlib import Path

RECIPES_DIR = Path(__file__).resolve().parent.parent / "recipes"


def recipe(name: str) -> str:
    """Load a family's reference.py source by filename."""
    return (RECIPES_DIR / name).read_text()


# ===================== Kimi-K2.7 (canonical) =====================
# Grounded in the repo's fp8-linear shapes + the sgl-kernel fused-gate test, cross-checked
# against sglang's router dispatch (deepseek_v2.py:585 accepts weight.shape[0] in {256,384}).
class KIMI:
    hf_id = "moonshotai/Kimi-K2.7"
    model = "kimi_k27"
    hidden = 7168
    q_lora = 1536
    kv_lora = 512
    num_heads = 64
    qk_nope = 192
    v_head = 128
    qk_rope = 64
    dense_inter_tp = 2304
    moe_inter_tp = 256
    vocab = 163840
    n_routed_experts = 384
    moe_topk = 6
    routed_scaling = 2.872
    ep_local_experts = 384 // 32          # 12
    moe_inter_full = 2048                 # per-expert intermediate under EP


# ===================== MiniMax-M3 (canonical) =====================
# AUTHORITATIVE from the live HF text_config of MiniMaxAI/MiniMax-M3 (minimax_m3_vl,
# a DSA sparse-attention + MoE VL model).
class MM:
    hf_id = "MiniMaxAI/MiniMax-M3"
    model = "minimax_m3"
    hidden = 6144
    nq = 64
    nkv = 4
    hd = 128
    rope_dim = 64
    experts = 128
    topk = 4
    inter = 3072
    shared_inter = 3072
    routed_scaling = 2.0
    vocab = 200064
    idx_dim = 128
    idx_heads = 4
    topk_blocks = 16
    block = 128
    dense_inter = 12288
    qkv_out = (nq + 2 * nkv) * hd         # 9216
    o_in = nq * hd                        # 8192
    ep_local = 16                         # 128 routed / EP8
    sglang_dir = "MM_M3_SGLANG_DIR"       # symbolic; resolved by bin/config.py at runtime
