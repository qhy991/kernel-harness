"""DSA Indexer K/V cache fused store — sglang baseline (store_kv_index, JIT CUDA).

MiniMax-M3 DSA cache write (inventory op30): a single-launch kernel that scatters the
new main K/V plus the index-K (and optional index-V) into their paged caches at the
per-token slot locations. Fuses what would otherwise be several index_put_ stores;
the inventory measures it ~12x faster than separate stores (peak 4815 GB/s @ T=16384).

  k/v -> k_cache/v_cache[loc]; idx_k -> idx_k_cache[loc]   (in place, at slots `indices`)

Config: num_kv_heads=4, head_dim=128, sparse_index_dim=128 (idx_v disabled per M3's
sparse_disable_index_value). Runs against the amd_add_m3 sglang build. reference.py IS
the correctness oracle AND the latency baseline; run() returns the mutated caches.
"""

import torch
from sglang.jit_kernel.minimax_store_kv_index import store_kv_index

NKV, HD, IDX = 4, 128, 128


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    T = axes_and_scalars["M"]            # tokens to write (sweep)
    N = axes_and_scalars["slots"]        # cache capacity
    dt = torch.bfloat16
    k = torch.randn(T, NKV * HD, device=device, dtype=dt)
    v = torch.randn(T, NKV * HD, device=device, dtype=dt)
    idx_k = torch.randn(T, IDX, device=device, dtype=dt)
    k_cache = torch.zeros(N, NKV * HD, device=device, dtype=dt)
    v_cache = torch.zeros_like(k_cache)
    idx_k_cache = torch.zeros(N, IDX, device=device, dtype=dt)
    indices = torch.randperm(N, device=device)[:T].to(torch.int64)
    return {"k": k, "v": v, "k_cache": k_cache, "v_cache": v_cache,
            "idx_k": idx_k, "idx_k_cache": idx_k_cache, "indices": indices}


@torch.no_grad()
def run(k, v, k_cache, v_cache, idx_k, idx_k_cache, indices):
    store_kv_index(k, v, k_cache, v_cache, idx_k, idx_k_cache, None, None, indices,
                   num_kv_heads=NKV, head_bytes=HD * 2)  # bf16 -> 2 bytes/elem
    return k_cache, v_cache, idx_k_cache
