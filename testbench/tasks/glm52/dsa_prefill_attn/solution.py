"""GLM-5.2 DSA sparse MLA prefill — independent Triton flash-attention kernel.

Independent reimplementation of the sparse latent attention computed by SGLang's
production sparse FlashMLA prefill op; it does NOT import or call that kernel.

Under DP=1/TP=1, GLM-5.2 exposes 64 local query heads. The Blackwell backend pads
them to 128 before dispatch, so the query tensor arrives as ``[M, 128, 576]`` with
only the first 64 heads populated; the result keeps just those 64 real heads as
``[M, 64, 512]``.

One Triton program handles a tile of query heads for one query token. All local
query heads share a single gathered set of ``topk`` latent KV vectors. The latent
width (576) is split into the value/nope part (first 512 dims) and the rope part
(last 64 dims); scores use both, while the value reduction reuses the already
gathered 512-wide part (V = first 512 latent dims). A numerically stable online
softmax runs over the selected tokens with fp32 accumulation, and the output is
stored as bfloat16 for the real heads only.
"""

import torch
import triton
import triton.language as tl

LOCAL_HEADS = 64
HEAD_DIM = 576
VALUE_DIM = 512      # value / nope latent width (also the output width)
ROPE_DIM = 64        # rope latent width; VALUE_DIM + ROPE_DIM == HEAD_DIM
HEAD_TILE = 32       # query-head rows handled per program (tensor-core friendly)
BLOCK_N = 128        # latent rows gathered per step (long coalesced bursts)
NUM_WARPS = 8
NUM_STAGES = 3


@triton.jit
def _sparse_mla_prefill_kernel(
    q_ptr, kv_ptr, idx_ptr, o_ptr,
    sm_scale,
    s_kv, topk,
    stride_qm, stride_qh, stride_qd,
    stride_kn, stride_kd,
    stride_im, stride_ik,
    stride_om, stride_oh, stride_od,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    V_DIM: tl.constexpr,
    R_DIM: tl.constexpr,
    H_KEEP: tl.constexpr,
):
    hb = tl.program_id(0)
    m = tl.program_id(1)

    offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)   # query-head rows for this tile
    offs_v = tl.arange(0, V_DIM)                    # value / nope dims
    offs_r = tl.arange(0, R_DIM)                    # rope dims
    head_store = offs_h < H_KEEP

    # Load this tile's query rows, split into value/nope and rope halves.
    q_lo_ptr = q_ptr + m * stride_qm + offs_h[:, None] * stride_qh + offs_v[None, :] * stride_qd
    q_hi_ptr = q_ptr + m * stride_qm + offs_h[:, None] * stride_qh + (V_DIM + offs_r)[None, :] * stride_qd
    q_lo = tl.load(q_lo_ptr, mask=head_store[:, None], other=0.0)   # [BLOCK_H, V_DIM] bf16
    q_hi = tl.load(q_hi_ptr, mask=head_store[:, None], other=0.0)   # [BLOCK_H, R_DIM] bf16

    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, V_DIM], dtype=tl.float32)

    for n0 in range(0, topk, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        in_range = offs_n < topk
        idx = tl.load(idx_ptr + m * stride_im + offs_n * stride_ik,
                      mask=in_range, other=s_kv)          # invalid sentinel when OOB
        valid = in_range & (idx >= 0) & (idx < s_kv)
        idx_safe = tl.where(valid, idx, 0)

        # Single gather of the 512-wide value/nope part (reused for V) plus the
        # 64-wide rope part (scores only).
        k_lo = tl.load(kv_ptr + idx_safe[:, None] * stride_kn + offs_v[None, :] * stride_kd,
                       mask=valid[:, None], other=0.0)     # [BLOCK_N, V_DIM] bf16
        k_hi = tl.load(kv_ptr + idx_safe[:, None] * stride_kn + (V_DIM + offs_r)[None, :] * stride_kd,
                       mask=valid[:, None], other=0.0)     # [BLOCK_N, R_DIM] bf16

        scores = tl.dot(q_lo, tl.trans(k_lo), out_dtype=tl.float32)
        scores += tl.dot(q_hi, tl.trans(k_hi), out_dtype=tl.float32)
        scores *= sm_scale
        scores = tl.where(valid[None, :], scores, float("-inf"))

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        active = m_new != float("-inf")
        m_exp = tl.where(active, m_new, 0.0)               # avoid -inf - (-inf)

        alpha = tl.where(active, tl.exp(m_i - m_new), 1.0)
        p = tl.exp(scores - m_exp[:, None])
        p = tl.where(valid[None, :] & active[:, None], p, 0.0)

        l_i = tl.where(active, l_i * alpha + tl.sum(p, axis=1), l_i)
        pv = tl.dot(p.to(k_lo.dtype), k_lo, out_dtype=tl.float32)
        acc = tl.where(active[:, None], acc * alpha[:, None] + pv, acc)
        m_i = tl.where(active, m_new, m_i)

    out = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)
    o_out_ptr = o_ptr + m * stride_om + offs_h[:, None] * stride_oh + offs_v[None, :] * stride_od
    tl.store(o_out_ptr, out.to(o_ptr.dtype.element_ty), mask=head_store[:, None])


@torch.no_grad()
def run(q, kv_cache, indices):
    m = q.shape[0]
    s_kv = kv_cache.shape[0]
    topk = indices.shape[-1]
    sm_scale = HEAD_DIM ** -0.5

    kv = kv_cache.reshape(s_kv, HEAD_DIM)
    idx = indices.reshape(m, topk)
    out = torch.empty(m, LOCAL_HEADS, VALUE_DIM, device=q.device, dtype=torch.bfloat16)

    # Head-tile varies fastest so all head-tiles of one query token get adjacent
    # program ids and co-schedule; the per-token KV that the first tile pulls from
    # HBM then serves the remaining tiles from L2 instead of being re-gathered.
    grid = (triton.cdiv(LOCAL_HEADS, HEAD_TILE), m)
    _sparse_mla_prefill_kernel[grid](
        q, kv, idx, out,
        sm_scale,
        s_kv, topk,
        q.stride(0), q.stride(1), q.stride(2),
        kv.stride(0), kv.stride(1),
        idx.stride(0), idx.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_N=BLOCK_N,
        BLOCK_H=HEAD_TILE,
        V_DIM=VALUE_DIM,
        R_DIM=ROPE_DIM,
        H_KEEP=LOCAL_HEADS,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out
