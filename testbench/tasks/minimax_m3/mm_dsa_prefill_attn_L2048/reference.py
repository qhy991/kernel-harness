"""DSA prefill main sparse attention — sglang baseline (flash_prefill_with_gqa_share_sparse).

MiniMax-M3 DSA prefill main attention (inventory op33, Triton CUDA). GQA-share sparse
attention over indexer-selected top-k KV blocks. Input contract mirrors the sglang test
test_minimax_m3_cuda_config.py::test_m3_prefill_main_sparse exactly (the authoritative
paged-KV layout): per-batch contiguous paged cache (max_slots = B*L), q one token/seq,
block_size_q=1, prefix_lens=L-1. topk_idx is random-but-valid block ids per (kv_head,seq).

  o[B, num_q_heads, head_dim] = sparse_prefill_attn(q, paged k/v_cache, topk_idx)

Config: num_q_heads=64, num_kv_heads=4 (GQA 16:1), head_dim=128, block_size_k=128,
topk_blocks=16. B=sweep, L fixed per task. Uses the shared SGLANG_DIR / installed sglang. reference.py IS the correctness oracle AND the latency baseline.
"""

import torch
from sglang.srt.layers.attention.minimax_sparse_ops.prefill.topk_sparse import (
    flash_prefill_with_gqa_share_sparse,
)

NQ, NKV, HD, BLK_K, TOPK = 64, 4, 128, 128, 16
BLK_Q = 1  # matches the M3 prefill test (max_seqlen_q=1 decode-shaped prefill)


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    B = axes_and_scalars["M"]      # batch (sweep)
    L = axes_and_scalars["L"]      # per-seq context length
    max_slots = B * L
    q = torch.randn(B, NQ, HD, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(max_slots, NKV, HD, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(max_slots, NKV, HD, device=device, dtype=torch.bfloat16)
    req_to_token = torch.zeros(B, L, dtype=torch.int32, device=device)
    slot_ids = torch.zeros(B, dtype=torch.int64, device=device)
    for i in range(B):
        base = i * L
        slot_ids[i] = i
        req_to_token[i, :L] = torch.arange(base, base + L, device=device)
    seq_lens = torch.full((B,), L, dtype=torch.int32, device=device)
    prefix_lens = torch.full((B,), L - 1, dtype=torch.int32, device=device)
    cu_seqlens = torch.arange(0, B + 1, device=device, dtype=torch.int32)  # 1 query token/seq
    num_blocks = (L + BLK_K - 1) // BLK_K
    topk_idx = torch.full((NKV, B, TOPK), -1, dtype=torch.int32, device=device)
    ak = min(TOPK, num_blocks)
    for kh in range(NKV):
        for b in range(B):
            topk_idx[kh, b, :ak] = torch.randperm(num_blocks, device=device)[:ak].to(torch.int32)
    return {"q": q, "k_cache": k_cache, "v_cache": v_cache, "req_to_token": req_to_token,
            "slot_ids": slot_ids, "topk_idx": topk_idx, "cu_seqlens": cu_seqlens,
            "seq_lens": seq_lens, "prefix_lens": prefix_lens}


@torch.no_grad()
def run(q, k_cache, v_cache, req_to_token, slot_ids, topk_idx,
        cu_seqlens, seq_lens, prefix_lens):
    return flash_prefill_with_gqa_share_sparse(
        q, k_cache, v_cache, None, req_to_token, slot_ids, topk_idx,
        BLK_Q, BLK_K, cu_seqlens, seq_lens, prefix_lens, 1)
