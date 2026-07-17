"""LM-head logits GEMM (decode) — bandwidth-optimized Triton kernel.

out[M, V] = hidden[M, H] @ lm_head_weight[V, H].T  in bf16.
H=7168, V=163840. The 2.35 GB bf16 weight is read exactly once, so this is
memory-bandwidth-bound. cuBLAS (torch.matmul) leaves headroom on these skinny
shapes; we win by tiling so each program owns a whole N-slab (grid over N only ->
weight read once) with large BLOCK_K for high HBM throughput, and per-M-regime
tuning of the tile / warps / pipeline depth.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _lmhead_kernel(A, W, Out, M, N, K,
                   sam, sak, swn, swk, som, son,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   NM: tl.constexpr):
    # One program per N-slab of BLOCK_N rows; iterate the full K so W is read once.
    # M is split into NM static sub-tiles of BLOCK_M that share each loaded W tile.
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, BLOCK_M)

    w_ptrs = W + offs_n[:, None] * swn + offs_k[None, :] * swk
    acc0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    ap0 = A + offs_m[:, None] * sam + offs_k[None, :] * sak
    ap1 = A + (BLOCK_M + offs_m)[:, None] * sam + offs_k[None, :] * sak
    ap2 = A + (2 * BLOCK_M + offs_m)[:, None] * sam + offs_k[None, :] * sak
    ap3 = A + (3 * BLOCK_M + offs_m)[:, None] * sam + offs_k[None, :] * sak

    for _ in range(0, K, BLOCK_K):
        w = tl.load(w_ptrs)
        wt = w.T
        acc0 = tl.dot(tl.load(ap0, mask=offs_m[:, None] < M, other=0.), wt, acc0)
        ap0 += BLOCK_K * sak
        if NM >= 2:
            acc1 = tl.dot(tl.load(ap1, mask=(BLOCK_M + offs_m)[:, None] < M, other=0.), wt, acc1)
            ap1 += BLOCK_K * sak
        if NM >= 3:
            acc2 = tl.dot(tl.load(ap2, mask=(2 * BLOCK_M + offs_m)[:, None] < M, other=0.), wt, acc2)
            ap2 += BLOCK_K * sak
        if NM >= 4:
            acc3 = tl.dot(tl.load(ap3, mask=(3 * BLOCK_M + offs_m)[:, None] < M, other=0.), wt, acc3)
            ap3 += BLOCK_K * sak
        w_ptrs += BLOCK_K * swk

    m0 = offs_m
    tl.store(Out + m0[:, None] * som + offs_n[None, :] * son, acc0.to(tl.bfloat16), mask=m0[:, None] < M)
    if NM >= 2:
        m1 = BLOCK_M + offs_m
        tl.store(Out + m1[:, None] * som + offs_n[None, :] * son, acc1.to(tl.bfloat16), mask=m1[:, None] < M)
    if NM >= 3:
        m2 = 2 * BLOCK_M + offs_m
        tl.store(Out + m2[:, None] * som + offs_n[None, :] * son, acc2.to(tl.bfloat16), mask=m2[:, None] < M)
    if NM >= 4:
        m3 = 3 * BLOCK_M + offs_m
        tl.store(Out + m3[:, None] * som + offs_n[None, :] * son, acc3.to(tl.bfloat16), mask=m3[:, None] < M)


def _config(M):
    # (BLOCK_M, BLOCK_N, BLOCK_K, NM, num_warps, num_stages), tuned per M-regime on B200.
    if M <= 16:
        return (16, 64, 128, 1, 4, 4)
    if M <= 32:
        return (32, 64, 128, 1, 4, 3)
    if M <= 64:
        return (64, 128, 64, 1, 4, 3)
    if M <= 128:
        return (128, 128, 64, 1, 8, 6)
    return None


@torch.no_grad()
def run(hidden_states, lm_head_weight):
    M, K = hidden_states.shape
    N = lm_head_weight.shape[0]
    cfg = _config(M)
    # For larger M the GEMM is balanced (not pure BW): cuBLAS tiles the large
    # output dimension of (W @ hidden.T) -> [V, M] far better than [M, V], so
    # compute that orientation and return its (identical) transpose view.
    if cfg is None:
        return torch.matmul(lm_head_weight, hidden_states.T).T
    BLOCK_M, BLOCK_N, BLOCK_K, NM, nw, ns = cfg
    if BLOCK_M * NM < M or N % BLOCK_N != 0:
        return torch.matmul(hidden_states, lm_head_weight.T)

    out = torch.empty((M, N), device=hidden_states.device, dtype=torch.bfloat16)
    _lmhead_kernel[(N // BLOCK_N,)](
        hidden_states, lm_head_weight, out, M, N, K,
        hidden_states.stride(0), hidden_states.stride(1),
        lm_head_weight.stride(0), lm_head_weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K, NM, num_warps=nw, num_stages=ns)
    return out
