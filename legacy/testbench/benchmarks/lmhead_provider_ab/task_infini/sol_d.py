"""LM-head logits GEMM — bandwidth-optimized Triton kernel.

out[M, V] = hidden[M, H] @ lm_head_weight[V, H].T   (bf16)

H=7168, V=163840. The 2.35 GB bf16 weight is read once and dominates cost:
this is HBM-bandwidth-bound. W is row-major [V, H], so a [BLOCK_N, BLOCK_K] tile
is contiguous along H (fully coalesced weight loads).

Grid is linearized N-major / M-minor so that, when M is large enough to split
into several BLOCK_M row-tiles, consecutive programs share the same weight column
tile — served from L2 (~120 MB on B200) rather than re-fetched from HBM. This keeps
effective HBM weight traffic near 1x while bounding the per-block fp32 accumulator
(a single BLOCK_M=256 pass spills registers and loses at M=256).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _lmhead_kernel(
    hidden_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_hm, stride_hk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    M_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_n = pid // M_BLOCKS
    pid_m = pid % M_BLOCKS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    h_ptrs = hidden_ptr + offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    m_mask = offs_m[:, None] < M

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(h_ptrs, mask=m_mask, other=0.0)          # [BM, BK]
        b = tl.load(w_ptrs)                                  # [BN, BK] contiguous
        acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32)  # [BM, BN]
        h_ptrs += BLOCK_K * stride_hk
        w_ptrs += BLOCK_K * stride_wk

    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask)


def run(hidden_states, lm_head_weight):
    hidden_states = hidden_states.bfloat16()
    lm_head_weight = lm_head_weight.bfloat16()
    M, K = hidden_states.shape
    N, K2 = lm_head_weight.shape
    assert K == K2

    out = torch.empty((M, N), device=hidden_states.device, dtype=torch.bfloat16)

    # Per-M tile configs from a CUDA-graph bandwidth sweep on B200. Small M covers
    # all rows in one BLOCK_M pass; large M splits rows (BLOCK_M=128) with L2 reuse.
    if M <= 16:
        BLOCK_M, BLOCK_N, BLOCK_K, warps, stages = 16, 64, 128, 4, 4
    elif M <= 32:
        BLOCK_M, BLOCK_N, BLOCK_K, warps, stages = 32, 128, 128, 4, 3
    elif M <= 64:
        BLOCK_M, BLOCK_N, BLOCK_K, warps, stages = 64, 128, 128, 8, 4
    else:  # M in {128, 256}
        BLOCK_M, BLOCK_N, BLOCK_K, warps, stages = 128, 64, 128, 8, 4

    M_BLOCKS = triton.cdiv(M, BLOCK_M)
    grid = (triton.cdiv(N, BLOCK_N) * M_BLOCKS,)
    _lmhead_kernel[grid](
        hidden_states, lm_head_weight, out,
        M, N, K,
        hidden_states.stride(0), hidden_states.stride(1),
        lm_head_weight.stride(0), lm_head_weight.stride(1),
        out.stride(0), out.stride(1),
        M_BLOCKS=M_BLOCKS,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=warps, num_stages=stages,
    )
    return out
