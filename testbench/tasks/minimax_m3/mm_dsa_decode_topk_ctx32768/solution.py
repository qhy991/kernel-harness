"""DSA decode Indexer top-k blocks — sglang baseline (minimax_decode_topk, JIT radix).

MiniMax-M3 Deep-Sparse-Attention decode indexer (inventory op32). Given per-index-head
block scores, selects the top-k KV blocks each query attends to. sglang dispatches the
JIT-CUDA radix kernel minimax_decode_topk (SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX,
shape-gated topk<=32); this task benchmarks that production kernel.

  topk_idx[num_index_heads, batch, topk] = top-k block ids by score, per (head, seq).

ROUTING oracle: integer block indices, EXACT-match (atol=0, rtol=0, ratio=1.0). Config:
sparse_num_index_heads=4, sparse_block_size=128, sparse_topk_blocks=16. Runs against the
amd_add_m3 sglang build (task.json sglang_dir). reference.py IS the oracle AND baseline.
"""

import torch
from sglang.jit_kernel.minimax_decode_topk import minimax_decode_topk


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    B = axes_and_scalars["M"]                 # decode batch (sweep)
    H = axes_and_scalars["index_heads"]
    BLOCK = axes_and_scalars["block_size"]
    ctx = axes_and_scalars["ctx"]
    num_blocks = (ctx + BLOCK - 1) // BLOCK
    score = torch.randn(H, B, num_blocks, device=device, dtype=torch.float32)
    seq_lens = torch.full((B,), ctx, device=device, dtype=torch.int32)
    return {"score": score, "seq_lens": seq_lens}


@torch.no_grad()
def run(score, seq_lens):
    # block_size and topk are fixed by the M3 DSA config.
    return minimax_decode_topk(score, seq_lens, 128, 16)
