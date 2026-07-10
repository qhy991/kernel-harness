"""Shape definitions for MiniMax-M3 / Kimi-K2.7 / DeepSeek-V3.2-style DSA models.

Each shape captures ONE operator called during sglang inference on a DSA MoE model
(hidden=6144, num_heads=64, q_lora_rank=2048, kv_lora_rank=512, qk_nope_head_dim=192,
qk_rope_head_dim=64, v_head_dim=256, num_index_heads=32, index_head_dim=128,
n_routed_experts=256, moe_intermediate_size=2048; TP=8, PP=2 for prefill; DP32xEP32
for decode with M_local=16).

Change these to match your target model config; keep the tuple layout intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---- MLA projections ---------------------------------------------------------
# Every "attr" mirrors the actual python/sglang/srt/models/deepseek_v2.py naming.


@dataclass
class LinearShape:
    op_id: int
    name: str
    phase: str          # 'prefill' | 'decode'
    M: int
    K: int
    N: int
    attr: str = ""      # sglang attribute name
    note: str = ""


LINEAR_OPS: list[LinearShape] = [
    # Q_a and KV_a share one Linear (fused_qkv_a_proj_with_mqa), output = q_lora + kv_lora + qk_rope
    LinearShape(1,  "Q_a (fused w/ KV_a)", "prefill", 16384, 6144, 2048 + 512 + 64, "fused_qkv_a_proj_with_mqa"),
    LinearShape(14, "Q_a (fused w/ KV_a)", "decode",     16, 6144, 2048 + 512 + 64, "fused_qkv_a_proj_with_mqa",
                note="M=16 triggers dsv3_fused_a_gemm on SM>=90 bf16"),
    LinearShape(2,  "Q_b",                 "prefill", 16384, 2048, 8 * 256,         "q_b_proj",
                note="TP-shard: 8 heads * qk_head_dim(=256)"),
    LinearShape(15, "Q_b",                 "decode",     16, 2048, 16384,           "q_b_proj",
                note="full weight, 64 heads * 256"),
    LinearShape(4,  "KV_b",                "prefill", 16384, 512,  8 * (192 + 256), "kv_b_proj",
                note="only in prefill; absorbed into w_kc/w_vc for decode"),
    LinearShape(5,  "O_proj",              "prefill", 16384, 2048, 6144,            "o_proj"),
    LinearShape(19, "O_proj",              "decode",     16, 16384, 6144,           "o_proj"),
    LinearShape(6,  "Index_Q",             "prefill", 16384, 6144, 32 * 128,        "Indexer.wq_b",
                note="Linear only; fused RoPE+quant is a separate JIT kernel"),
    LinearShape(20, "Index_Q",             "decode",     16, 6144, 32 * 128,        "Indexer.wq_b"),
    LinearShape(7,  "Index_K",             "prefill", 16384, 6144, 128,             "Indexer.wk",
                note="single index-K head (MQA-style)"),
    LinearShape(21, "Index_K",             "decode",     16, 6144, 128,             "Indexer.wk"),
    LinearShape(9,  "Dense GateUp",        "prefill", 16384, 6144, 2 * 3072,        "shared_experts.gate_up_proj"),
    LinearShape(10, "Dense Down",          "prefill", 16384, 1536, 6144,            "shared_experts.down_proj"),
    LinearShape(11, "MoE Router",          "prefill", 16384, 6144, 256,             "mlp.gate (MoEGate)"),
    LinearShape(23, "MoE Router",          "decode",     16, 6144, 256,             "mlp.gate (MoEGate)"),
]


# ---- Grouped GEMM (MoE) -----------------------------------------------------

@dataclass
class GroupedGemmShape:
    op_id: int
    name: str
    phase: str
    E: int          # num experts
    M: int          # M per expert
    K: int
    N: int
    note: str = ""


GROUPED_GEMM_OPS: list[GroupedGemmShape] = [
    GroupedGemmShape(12, "MoE GateUp GroupGEMM", "prefill", 256, 512, 6144, 512,
                     note="NORMAL: deep_gemm.m_grouped_fp8_gemm_nt_contiguous"),
    GroupedGemmShape(13, "MoE Down GroupGEMM",   "prefill", 256, 512, 256,  6144,
                     note="K=256 is per-TP-shard intermediate (2048/8), not expert count"),
    GroupedGemmShape(24, "MoE GateUp GroupGEMM", "decode",    8,  16, 6144, 4096,
                     note="LOW_LATENCY: deep_gemm.fp8_m_grouped_gemm_nt_masked"),
    GroupedGemmShape(25, "MoE Down GroupGEMM",   "decode",    8,  16, 2048, 6144),
]


# ---- Absorb BMM (decode MLA) ------------------------------------------------

@dataclass
class BmmShape:
    op_id: int
    name: str
    phase: str
    M: int          # batch/seq
    H: int          # per-head
    IN: int
    OUT: int
    note: str = ""


BMM_OPS: list[BmmShape] = [
    BmmShape(17, "q_nope absorb BMM", "decode", 16, 64, 192, 512,
             note="w_kc precomputed from kv_b_proj; torch.bmm is the actual path (bf16)"),
    BmmShape(18, "v absorb BMM",       "decode", 16, 64, 512, 256,
             note="w_vc precomputed from kv_b_proj"),
]


# ---- DSA sparse indexer score ------------------------------------------------

@dataclass
class IndexScoreShape:
    op_id: int
    name: str
    phase: str
    M_q: int
    M_k: int
    H_idx: int      # num_index_heads (32)
    D: int          # index_head_dim (128)
    topk: int
    note: str = ""


INDEX_SCORE_OPS: list[IndexScoreShape] = [
    IndexScoreShape(8,  "Index_Score (ragged prefill)", "prefill", 16384, 16384, 32, 128, 2048,
                    note="deep_gemm.fp8_mqa_logits + topk_transform"),
    IndexScoreShape(22, "Index_Score (paged decode)",   "decode",     16,  2048, 32, 128, 2048,
                    note="deep_gemm.fp8_paged_mqa_logits + topk"),
]


# ---- Attention ---------------------------------------------------------------

@dataclass
class AttentionShape:
    op_id: int
    name: str
    phase: str
    B: int          # batch
    H_q: int        # num query heads (per TP shard for MHA prefill; total for MLA absorbed decode)
    H_kv: int       # num KV heads (1 for MQA sparse decode)
    T_q: int        # query seq len
    T_kv: int       # KV seq len (topk for sparse decode)
    T_kv_full: int  # full KV pool len (for sparse: total tokens to sample topk from)
    D_qk: int
    D_v: int
    causal: bool = False
    note: str = ""


ATTENTION_OPS: list[AttentionShape] = [
    AttentionShape(27, "FlashAttn causal MHA (prefill)", "prefill",
                   B=1, H_q=8, H_kv=8, T_q=16384, T_kv=16384, T_kv_full=16384,
                   D_qk=256, D_v=256, causal=True,
                   note="sgl_kernel.flash_attn (FA3); head_dim=256"),
    AttentionShape(26, "Flash Decoding MLA sparse MQA", "decode",
                   B=1, H_q=64, H_kv=1, T_q=16, T_kv=2048, T_kv_full=8192,
                   D_qk=512, D_v=512, causal=False,
                   note="q_absorbed [B,H_q,T_q,D]; MQA (1 KV head); topk=2048 out of T_kv_full"),
]
