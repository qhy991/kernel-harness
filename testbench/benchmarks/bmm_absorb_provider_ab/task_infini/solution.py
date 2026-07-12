"""MLA weight-absorb BMM — Triton batched matmul.

out[Bh, M, N] = bmm(a[Bh, M, K], b[Bh, K, N]),  Bh=64, K=192, N=512, bf16.

At decode M sweeps small; the op is bandwidth/launch-bound on the per-head
operands at these tiny shapes. A tuned Triton kernel beats torch.bmm's
per-launch overhead there. K=192 is a multiple of every BLOCK_K used, so the
K loop needs no masking; only M is masked. The tile is chosen per M so each
regime reads the operands with minimal redundancy.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _bmm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + pid_b * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m_mask = offs_m[:, None] < M
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=m_mask, other=0.0)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.bfloat16)
    c_ptrs = c_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=offs_m[:, None] < M)


# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) chosen per M regime,
# measured via the harness's own CUPTI cold-L2 timing.
def _cfg(M):
    if M <= 16:
        return (16, 64, 32, 2, 5)
    if M <= 32:
        return (32, 64, 32, 2, 5)
    if M <= 64:
        return (64, 128, 32, 4, 5)
    if M <= 128:
        return (128, 128, 64, 8, 3)
    return (128, 256, 32, 4, 4)


@torch.no_grad()
def run(a, b):
    Bh, M, K = a.shape
    N = b.shape[2]

    # cuBLAS (nvjet) is compute-efficient for the larger token counts; the
    # Triton path only wins at the tiny, launch/bandwidth-bound decode shapes.
    if M > 32:
        return torch.bmm(a, b)

    c = torch.empty((Bh, M, N), device=a.device, dtype=torch.bfloat16)

    BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = _cfg(M)
    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M), Bh)
    _bmm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return c
