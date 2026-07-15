"""GLM-5.2 DSA sparse MLA prefill — independent chunked-PyTorch reference.

This is the correctness oracle for the task: an independent reimplementation of
the sparse latent attention computed by SGLang's ``flash_mla_sparse_fwd``, written
purely in PyTorch tensor ops. It does NOT call the production kernel.

For each query token it gathers ``topk`` latent KV vectors from a shared cache via
``indices``, scores ``Q·Kᵀ`` over the full latent width (576) scaled by
``sm_scale = 576**-0.5``, softmaxes over the selected tokens (invalid indices
masked to ``-inf``), and reduces against the first ``d_v = 512`` dims of the same
latent vectors. Only the 8 tensor-parallel-local heads are returned. All math is
accumulated in fp32; the output is cast to bfloat16.
"""

import torch

LOCAL_HEADS = 8
HEAD_DIM = 576
VALUE_DIM = 512

# Query-token chunk size: bounds the transient fp32 gather buffer
# (chunk * topk * 576 * 4 bytes) so a large M cannot exhaust device memory.
_M_CHUNK = 256


@torch.no_grad()
def run(q, kv_cache, indices):
    device = q.device
    m = q.shape[0]
    s_kv = kv_cache.shape[0]
    topk = indices.shape[-1]
    sm_scale = HEAD_DIM ** -0.5

    kv = kv_cache.reshape(s_kv, HEAD_DIM)              # [s_kv, 576]
    idx = indices.reshape(m, topk).to(torch.long)      # [M, topk]
    q_local = q[:, :LOCAL_HEADS, :]                    # [M, 8, 576]

    out = torch.empty(m, LOCAL_HEADS, VALUE_DIM, device=device, dtype=torch.bfloat16)

    for start in range(0, m, _M_CHUNK):
        end = min(start + _M_CHUNK, m)
        idx_c = idx[start:end]                         # [c, topk]
        valid = (idx_c >= 0) & (idx_c < s_kv)          # [c, topk]
        idx_safe = torch.where(valid, idx_c, torch.zeros_like(idx_c))

        gathered = kv.index_select(0, idx_safe.reshape(-1))
        gathered = gathered.reshape(end - start, topk, HEAD_DIM).float()   # [c, topk, 576]
        q_c = q_local[start:end].float()                                    # [c, 8, 576]

        scores = torch.einsum("mhd,mkd->mhk", q_c, gathered) * sm_scale     # [c, 8, topk]
        scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        # A query whose selected tokens are all invalid has an all -inf row;
        # softmax yields NaN there. The reference emits a zero output row, so
        # replace NaN weights with zero to match.
        probs = torch.nan_to_num(probs, nan=0.0)

        out[start:end] = torch.einsum(
            "mhk,mkd->mhd", probs, gathered[..., :VALUE_DIM]
        ).to(torch.bfloat16)

    return out
