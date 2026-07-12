"""DSA decode main sparse attention — sglang baseline (flash_decode_with_gqa_share_sparse).

MiniMax-M3 Deep-Sparse-Attention decode main attention (inventory op34): GQA-share sparse
attention that, per (kv_head, seq), attends only to the top-k KV blocks selected by the
indexer. sglang dispatches the Triton flash_decode_with_gqa_share_sparse kernel (the B200
fmha_sm100 fast path needs an external package not installed; this benchmarks the Triton
production fallback, which its own unit tests cover).

  o[batch, num_q_heads, head_dim] = sparse_attn(q, paged k/v_cache, req_to_token, topk_idx)

Config: num_q_heads=64, num_kv_heads=4 (GQA 16:1), head_dim=128, block_size=128,
topk_blocks=16. Batch = sweep; ctx fixed per task. paged KV via randperm req_to_token.
Runs against the amd_add_m3 sglang build. reference.py IS the oracle AND baseline.
"""

import torch
from sglang.srt.layers.attention.minimax_sparse_ops.decode.topk_sparse import (
    flash_decode_with_gqa_share_sparse,
)

NQ, NKV, HD, BLOCK, TOPK = 64, 4, 128, 128, 16


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    B = axes_and_scalars["M"]       # decode batch (sweep)
    ctx = axes_and_scalars["ctx"]
    max_kv_len = ctx
    max_slots = B * max_kv_len
    q = torch.randn(B, NQ, HD, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(max_slots, NKV, HD, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(max_slots, NKV, HD, device=device, dtype=torch.bfloat16)
    req_to_token = torch.zeros(B, max_kv_len, dtype=torch.int32, device=device)
    slot_ids = torch.zeros(B, dtype=torch.int64, device=device)
    seq_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    for i in range(B):
        base = i * max_kv_len
        slot_ids[i] = i
        req_to_token[i, :max_kv_len] = torch.randperm(max_kv_len, device=device) + base
    num_blocks = (ctx + BLOCK - 1) // BLOCK
    topk_idx = torch.zeros(NKV, B, TOPK, dtype=torch.int32, device=device)
    for kh in range(NKV):
        for b in range(B):
            ak = min(TOPK, num_blocks)
            topk_idx[kh, b, :ak] = torch.randperm(num_blocks, device=device)[:ak].to(torch.int32)
            if ak < TOPK:
                topk_idx[kh, b, ak:] = -1
    return {"q": q, "k_cache": k_cache, "v_cache": v_cache, "req_to_token": req_to_token,
            "seq_lens": seq_lens, "slot_ids": slot_ids, "topk_idx": topk_idx}


@torch.no_grad()
def run(q, k_cache, v_cache, req_to_token, seq_lens, slot_ids, topk_idx):
    return flash_decode_with_gqa_share_sparse(
        q, None, k_cache, v_cache, req_to_token, seq_lens, slot_ids, BLOCK, topk_idx)
