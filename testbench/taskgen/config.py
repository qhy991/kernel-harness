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


# ===================== GLM-5.2-FP8 (canonical) =====================
# AUTHORITATIVE from zai-org/GLM-5.2-FP8 config.json / GlmMoeDsaConfig.
# Deployment assumption for shapes: B200, DP=1 / TP=1 / EP=32
# (full attention heads = 64, local experts = 8). Aligns single-rank GEMM /
# MoE shapes with llm_flops' unpartitioned-attention + E=8 microbench.
# Sparse MLA decode oracle is the B200 production path (TRT-LLM sparse MLA),
# not Hopper flashmla_sparse.
class GLM52:
    hf_id = "zai-org/GLM-5.2-FP8"
    model = "glm52"
    hidden = 6144
    q_lora = 2048
    kv_lora = 512
    num_heads = 64
    qk_nope = 192
    qk_rope = 64
    v_head = 256                       # config.v_head_dim; absorbed MLA output uses kv_lora
    qk_head = qk_nope + qk_rope        # 256
    head_dim = kv_lora + qk_rope       # 576 latent KV dim for sparse MLA
    n_routed_experts = 256
    moe_topk = 8
    n_shared_experts = 1
    moe_inter = 2048                   # per-expert intermediate
    dense_inter = 12288
    routed_scaling = 2.5
    vocab = 154880
    index_topk = 2048
    index_n_heads = 32
    index_head_dim = 128
    page_size = 64
    dp = 1
    tp = 1
    ep = 32
    # Derived per-rank dims under DP1/TP1/EP32
    local_heads = num_heads // tp      # 64
    o_in = local_heads * v_head        # 16384 — full (unshard) o_proj input
    o_in_absorb = local_heads * kv_lora  # 32768 when O consumes absorbed latent V
    ep_local = n_routed_experts // ep  # 8
    gateup_n = 2 * moe_inter           # 4096 Gate|Up per expert
    deployment = "B200-DP1-TP1-EP32"
    # Blackwell FlashMLA sparse requires q heads multiple of 128; pad 64→128.
    dsa_padded_heads = 128
    first_k_dense_replace = 3
    num_hidden_layers = 78
