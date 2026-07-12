"""Kimi K2 / DeepSeek-V3.2 layer-type shape helpers.

Two profiles:
- kimi_k2: moonshotai/Kimi-K2-Instruct (MLA+MoE, no DSA indexer)
- dsa_v32: DeepSeek-V3.2-Exp style (MLA+DSA+MoE) — matches harness shapes.py family

TP assumptions match kernel-harness README: TP=8 prefill, decode DP32×EP32 → M_local=16.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KimiProfile:
    name: str
    hidden: int
    intermediate: int  # dense FFN (DeepseekV2MLP)
    moe_inter: int
    n_shared: int
    n_routed: int
    first_k_dense: int
    q_lora: int
    kv_lora: int
    qk_nope: int
    qk_rope: int
    v_head: int
    num_heads: int
    index_heads: int
    index_dim: int
    index_topk: int
    index_topk_freq: int
    vocab: int
    tp: int
    m_prefill: int
    m_decode: int
    has_dsa: bool


KIMI_K2 = KimiProfile(
    name="kimi_k2",
    hidden=7168,
    intermediate=18432,
    moe_inter=2048,
    n_shared=1,
    n_routed=384,
    first_k_dense=1,
    q_lora=1536,
    kv_lora=512,
    qk_nope=128,
    qk_rope=64,
    v_head=128,
    num_heads=64,
    index_heads=0,
    index_dim=0,
    index_topk=0,
    index_topk_freq=1,
    vocab=163840,
    tp=8,
    m_prefill=16384,
    m_decode=16,
    has_dsa=False,
)

DSA_V32 = KimiProfile(
    name="dsa_v32",
    hidden=6144,  # harness deployment shard (see shapes.py)
    intermediate=18432,
    moe_inter=2048,
    n_shared=1,
    n_routed=256,
    first_k_dense=3,
    q_lora=2048,
    kv_lora=512,
    qk_nope=192,
    qk_rope=64,
    v_head=256,
    num_heads=64,
    index_heads=32,
    index_dim=128,
    index_topk=2048,
    index_topk_freq=1,
    vocab=129280,
    tp=8,
    m_prefill=16384,
    m_decode=16,
    has_dsa=True,
)


def _tp_div(x: int, tp: int) -> int:
    assert x % tp == 0, f"{x} not divisible by TP={tp}"
    return x // tp


def dense_ffn_shapes(p: KimiProfile, phase: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """DeepseekV2MLP gate_up / down GEMM shapes (per TP shard)."""
    m = p.m_prefill if phase == "prefill" else p.m_decode
    inter = _tp_div(p.intermediate, p.tp)
    gate_up = (m, p.hidden, 2 * inter)
    down = (m, inter, p.hidden)
    return gate_up, down


def shared_expert_shapes(p: KimiProfile, phase: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    m = p.m_prefill if phase == "prefill" else p.m_decode
    inter = _tp_div(p.moe_inter * p.n_shared, p.tp)
    gate_up = (m, p.hidden, 2 * inter)
    down = (m, inter, p.hidden)
    return gate_up, down


def moe_grouped_shapes(p: KimiProfile, phase: str) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """(E, M_per_expert, K, N) for gate_up and down grouped GEMMs."""
    inter = _tp_div(p.moe_inter, p.tp)
    if phase == "prefill":
        e, m_e = p.n_routed, 512
        gate = (e, m_e, p.hidden, inter)
        down = (e, m_e, inter, p.hidden)
    else:
        e, m_e = 8, p.m_decode
        gate = (e, m_e, p.hidden, inter * p.tp)  # decode masked path uses wider N on some deploys
        down = (e, m_e, inter * p.tp // 2, p.hidden)  # approximate decode down K
    return gate, down


def layer_kinds(p: KimiProfile) -> list[dict]:
    """Summarize which operator sets fire on each layer type."""
    kinds = []
    for lid in range(3):  # illustrate first few layers
        if lid < p.first_k_dense:
            kinds.append({"layer_id": lid, "kind": "dense_ffn", "mlp": "DeepseekV2MLP", "dsa_indexer": False})
        else:
            kinds.append(
                {
                    "layer_id": lid,
                    "kind": "moe",
                    "mlp": "DeepseekV2MoE",
                    "dsa_indexer": p.has_dsa
                    and (p.index_topk_freq <= 1 or lid % p.index_topk_freq == 1),
                }
            )
    return kinds
