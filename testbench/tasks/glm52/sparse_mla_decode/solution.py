"""GLM-5.2 Sparse MLA decode — SGLang B200 production baseline (TRT-LLM).

On Blackwell + FP8 KV, SGLang's DSA decode path dispatches
`flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(..., backend="trtllm-gen")`
(see sglang dsa_backend._forward_trtllm). That kernel is the correctness oracle
AND the latency baseline for this task — not Hopper flashmla_sparse.

FP8 path requires FP8 query + FP8 KV (BF16 query with FP8 KV has no TRTLLM-GEN
kernel). Query is pre-quantized offline in get_inputs; only the decode launch is timed.

I/O contract (TP8 local shard):
  q[B, 1, H_local=8, 576]          — absorbed (nope|rope) query, FP8_e4m3
  kv_cache[pages, 1, page=64, 576] — FP8_e4m3 packed latent KV (TRTLLM layout)
  block_tables[B, 1, topk=2048]    — selected token indices (invalid = -1)
  seq_lens[B]                      — per-seq context length
  workspace                        — flashinfer workspace buffer

  o[B, H_local, kv_lora=512]

Only the decode kernel is timed. Index / page construction is untimed in get_inputs.
"""

import os

import torch

NUM_HEADS_LOCAL = 8          # 64 / TP8
KV_LORA = 512
QK_ROPE = 64
QK_NOPE = 192
HEAD_DIM = KV_LORA + QK_ROPE  # 576
PAGE_SIZE = 64
INDEX_TOPK = 2048
# FlashInfer TRT-LLM workspace. Sparse MLA over long ctx + large batch needs more
# than the default 128 MiB; size with headroom from ctx/batch.
_DEFAULT_WS = 128 * 1024 * 1024
_MAX_WS = 1024 * 1024 * 1024


def _workspace(device: torch.device, batch: int = 1, ctx: int = 1024) -> torch.Tensor:
    env = os.environ.get("SGLANG_FLASHINFER_WORKSPACE_SIZE")
    if env:
        n = int(env)
    else:
        # Empirically: 128 MiB covers short ctx; scale up with B * ctx token traffic.
        n = max(_DEFAULT_WS, int(256 * 1024 * 1024 * (batch * max(ctx, 1) / 65536.0)))
        n = min(n, _MAX_WS)
    return torch.zeros(n, dtype=torch.uint8, device=device)  # zeros required on first use


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    B = int(axes_and_scalars["M"])
    ctx = int(axes_and_scalars["ctx"])
    topk = int(axes_and_scalars.get("topk", INDEX_TOPK))
    heads = int(axes_and_scalars.get("num_heads", NUM_HEADS_LOCAL))
    head_dim = int(axes_and_scalars.get("head_dim", HEAD_DIM))
    page = int(axes_and_scalars.get("page_size", PAGE_SIZE))
    sm_scale = float(axes_and_scalars.get("sm_scale", head_dim ** -0.5))

    # Effective selected length: min(ctx, topk). Pad remaining slots with -1.
    eff = min(ctx, topk)

    # Token pool sized to cover B sequences of `ctx`, rounded up to page boundary.
    tokens_per_seq = ((ctx + page - 1) // page) * page
    total_tokens = B * tokens_per_seq
    num_pages = total_tokens // page

    # FP8 query (B200 production path quantizes Q before TRT-LLM).
    q = (torch.randn(B, 1, heads, head_dim, device=device, dtype=torch.float32) * 0.05
         ).to(torch.float8_e4m3fn)

    # FP8 latent KV cache in TRTLLM layout: [pages, 1, page_size, head_dim].
    kv_bf16 = (torch.randn(num_pages, 1, page, head_dim, device=device,
                           dtype=torch.float32) * 0.05)
    kv_cache = kv_bf16.to(torch.float8_e4m3fn)

    # Per-seq selected indices into the flat token pool. Seeded for reproducibility.
    g = torch.Generator(device=device)
    g.manual_seed(20260714 + B * 1009 + ctx)
    block_tables = torch.full((B, 1, topk), -1, dtype=torch.int32, device=device)
    for b in range(B):
        base = b * tokens_per_seq
        # Sample `eff` distinct positions from [base, base+ctx).
        perm = torch.randperm(ctx, generator=g, device=device)[:eff]
        block_tables[b, 0, :eff] = (perm + base).to(torch.int32)

    seq_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    workspace = _workspace(device, batch=B, ctx=ctx)
    # Scalar axes carried as 0-dim tensors so they survive clone_args.
    bmm1_scale = torch.tensor(sm_scale, dtype=torch.float32, device=device)
    max_seq_len = torch.tensor(ctx, dtype=torch.int32, device=device)
    return {
        "q": q,
        "kv_cache": kv_cache,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "workspace": workspace,
        "bmm1_scale": bmm1_scale,
        "max_seq_len": max_seq_len,
    }


@torch.no_grad()
def run(q, kv_cache, block_tables, seq_lens, workspace, bmm1_scale, max_seq_len):
    import flashinfer.decode

    # Mirror dsa_backend._forward_trtllm view conventions.
    kv = kv_cache.view(-1, 1, kv_cache.shape[-2], kv_cache.shape[-1])
    msl = int(max_seq_len.item() if isinstance(max_seq_len, torch.Tensor)
              else max_seq_len)
    out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv,
        workspace_buffer=workspace,
        qk_nope_head_dim=QK_NOPE,
        kv_lora_rank=KV_LORA,
        qk_rope_head_dim=QK_ROPE,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=msl,
        sparse_mla_top_k=block_tables.shape[-1],
        bmm1_scale=bmm1_scale,
        backend="trtllm-gen",
    )
    # flashinfer returns [B, 1, H, kv_lora] or [B, H, kv_lora] depending on version;
    # normalize to [B, H, kv_lora] for a stable contract.
    if out.ndim == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    return out
