"""MLA decode attention — sglang baseline (sgl_kernel.cutlass_mla_decode).

Kimi-K2.7 core Multi-head Latent Attention on the decode path. sglang's CutlassMLABackend
dispatches sgl_kernel.cutlass_mla_decode over the compressed latent KV cache (kv_lora_rank
+ qk_rope_head_dim). Input contract mirrors sgl-kernel/tests/test_cutlass_mla.py.

  o[bs, num_heads, kv_lora] = cutlass_mla_decode(q_nope, q_pe, kv_c_and_k_pe_cache,
                                                 seq_lens, block_table, workspace, scale)

Kimi-K2.7: num_heads=64, kv_lora_rank=512 (=dv), qk_rope_head_dim=64 -> head_dim d=576,
block_size=128. batch = sweep; seq_len fixed per task (paged latent KV). Only the MLA
kernel is timed (workspace + caches built untimed in get_inputs). reference.py IS the
correctness oracle AND the latency baseline.
"""

import torch
from sgl_kernel import cutlass_mla_decode, cutlass_mla_get_workspace_size

NUM_HEADS, KV_LORA, QK_ROPE = 64, 512, 64
D = KV_LORA + QK_ROPE       # 576 (latent head dim)
DV = KV_LORA                # 512 (value / q_nope dim)
BLOCK_SIZE = 128
NUM_KV_SPLITS = 1


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    bs = axes_and_scalars["M"]         # decode batch (sweep)
    seq_len = axes_and_scalars["seq_len"]
    scale = D ** -0.5
    block_num = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    pack = 128 // BLOCK_SIZE
    block_num = ((block_num + pack - 1) // pack) * pack

    q = torch.randn(bs, NUM_HEADS, D, device=device, dtype=torch.bfloat16)
    q_nope = torch.empty((NUM_HEADS, bs, DV), device=device, dtype=torch.bfloat16).transpose(0, 1)
    q_nope.copy_(q[:, :, :DV])
    q_pe = q[:, :, DV:].clone()
    block_table = torch.randint(0, bs * block_num, (bs, block_num), dtype=torch.int32, device=device)
    kv_cache = torch.randn(block_table.numel(), BLOCK_SIZE, D, device=device, dtype=torch.bfloat16)
    seq_lens = torch.full((bs,), seq_len, dtype=torch.int32, device=device)
    ws = cutlass_mla_get_workspace_size(block_num * BLOCK_SIZE, bs, num_kv_splits=NUM_KV_SPLITS)
    workspace = torch.empty(ws, device=device, dtype=torch.uint8)
    return {"q_nope": q_nope, "q_pe": q_pe, "kv_cache": kv_cache,
            "seq_lens": seq_lens, "block_table": block_table, "workspace": workspace,
            "scale": scale}


@torch.no_grad()
def run(q_nope, q_pe, kv_cache, seq_lens, block_table, workspace, scale):
    return cutlass_mla_decode(q_nope, q_pe, kv_cache, seq_lens, block_table,
                              workspace, scale, NUM_KV_SPLITS)
