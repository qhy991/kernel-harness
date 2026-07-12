"""Triton batched-matmul for Kimi-K2.7 MLA q_nope-absorb decode BMM.

out[Bh, M, N] = bmm(a[Bh, M, K], b[Bh, K, N]),  Bh=64, K=192, N=512, bf16.

torch.bmm is memory/launch-bound at the tiny per-head shapes (small M): the b
operand (Bh*K*N*2 = 12.6MB) dominates traffic and torch only reaches ~2 TB/s. We
stream b once with a lightweight tiled kernel and beat the launch floor there.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _bmm_kernel(a_ptr, b_ptr, c_ptr,
                M,
                stride_ab, stride_am, stride_ak,
                stride_bb, stride_bk, stride_bn,
                stride_cb, stride_cm, stride_cn,
                K: tl.constexpr, N: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_K: tl.constexpr):
    bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + bh * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + bh * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m_mask = offs_m[:, None] < M
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=m_mask, other=0.0)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(tl.bfloat16)
    c_ptrs = c_ptr + bh * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=m_mask)


def _cfg(M):
    # (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
    if M <= 16:
        return 16, 128, 64, 4, 3
    if M <= 32:
        return 32, 256, 64, 4, 4
    if M <= 64:
        return 64, 128, 64, 4, 3
    if M <= 128:
        return 128, 128, 64, 4, 3
    return 256, 256, 64, 8, 3


@torch.no_grad()
def run(a, b):
    Bh, M, K = a.shape
    N = b.shape[2]
    c = torch.empty((Bh, M, N), device=a.device, dtype=torch.bfloat16)
    BLOCK_M, BLOCK_N, BLOCK_K, nw, ns = _cfg(M)
    grid = (Bh, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _bmm_kernel[grid](
        a, b, c, M,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        K, N, BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=nw, num_stages=ns,
    )
    return c
