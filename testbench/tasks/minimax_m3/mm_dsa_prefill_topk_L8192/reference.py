"""DSA prefill Indexer score + top-k — sglang baseline (flash_prefill_with_topk_index).

MiniMax-M3 DSA prefill indexer (inventory op31, single-launch Triton CUDA): computes
per-block index scores from idx_q x idx_k_cache and selects the top-k blocks in one
kernel. Input contract mirrors the sglang test
test_minimax_m3_cuda_config.py::test_m3_prefill_score_topk (max_seqlen_q=1, block_size_q=1,
prefix_lens=L-1, per-batch paged index cache).

  (score, topk_idx[num_index_heads, B, topk]) = index_score_topk(idx_q, idx_k_cache)

Config: num_index_heads=4, idx_head_dim=128, block_size=128, topk_blocks=16,
disable_index_value=True (M3 sparse_disable_index_value). Returns the integer topk_idx
(EXACT-index oracle). Runs against the amd_add_m3 sglang build. reference.py IS the
correctness oracle AND the latency baseline.
"""

import torch
from sglang.srt.layers.attention.minimax_sparse_ops.prefill.flash_with_topk_idx import (
    flash_prefill_with_topk_index,
)

IDX_HEADS, IDX_DIM, BLK_K, TOPK = 4, 128, 128, 16


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    B = axes_and_scalars["M"]      # batch (sweep)
    L = axes_and_scalars["L"]      # per-seq context length
    max_slots = B * L
    idx_q = torch.randn(B, IDX_HEADS, IDX_DIM, device=device, dtype=torch.bfloat16)
    idx_k = torch.randn(max_slots, 1, IDX_DIM, device=device, dtype=torch.bfloat16)
    req_to_token = torch.zeros(B, L, dtype=torch.int32, device=device)
    slot_ids = torch.zeros(B, dtype=torch.int64, device=device)
    for i in range(B):
        base = i * L
        slot_ids[i] = i
        req_to_token[i, :L] = torch.arange(base, base + L, device=device)
    seq_lens = torch.full((B,), L, dtype=torch.int32, device=device)
    prefix_lens = torch.full((B,), L - 1, dtype=torch.int32, device=device)
    cu_seqlens = torch.arange(0, B + 1, device=device, dtype=torch.int32)
    return {"idx_q": idx_q, "idx_k": idx_k, "req_to_token": req_to_token,
            "slot_ids": slot_ids, "cu_seqlens": cu_seqlens, "seq_lens": seq_lens,
            "prefix_lens": prefix_lens, "seqlen": L}


@torch.no_grad()
def run(idx_q, idx_k, req_to_token, slot_ids, cu_seqlens, seq_lens, prefix_lens, seqlen):
    _score, topk_idx = flash_prefill_with_topk_index(
        idx_q, idx_k, None, None, req_to_token, slot_ids, cu_seqlens, seq_lens,
        prefix_lens, 1, seqlen, 1, BLK_K, TOPK, disable_index_value=True)
    return topk_idx
