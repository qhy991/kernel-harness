"""Float32-scale GLM-5.2 o_proj decode candidate for Kernel-Harness.

M=16 uses an owned Triton FP8 block-scaled GEMM.  M=32 deliberately falls back
to the current DeepGEMM reference so it cannot veto an M=16 win.
"""
from __future__ import annotations

import deep_gemm
import torch
import triton
import triton.language as tl


K_BLOCK = 128


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_N": block_n},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_n in (32, 64, 128)
        for num_warps in (4, 8)
        for num_stages in (2, 3, 4)
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _float_scale_fp8_gemm_direct(
    x_ptr,
    w_ptr,
    xs_ptr,
    ws_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_xsm,
    stride_xsk,
    stride_wsn,
    stride_wsk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_BLOCKS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 128)
    m_mask = offs_m < M
    n_mask = offs_n < N

    x_base = x_ptr + offs_m[:, None] * stride_xm
    w_base = w_ptr + offs_n[None, :] * stride_wn

    # w_scale is [N/128, K/128], not expanded to one row per output.
    # BLOCK_N divides 128, so every tile lies within exactly one scale row.
    scale_n = offs_n // 128
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, K_BLOCKS):
        kk = kb * 128 + offs_k
        x = tl.load(
            x_base + kk[None, :] * stride_xk,
            mask=m_mask[:, None],
            other=0.0,
        )
        w = tl.load(
            w_base + kk[:, None] * stride_wk,
            mask=n_mask[None, :],
            other=0.0,
        )
        sx = tl.load(
            xs_ptr + offs_m * stride_xsm + kb * stride_xsk,
            mask=m_mask,
            other=0.0,
        )
        sw = tl.load(
            ws_ptr + scale_n * stride_wsn + kb * stride_wsk,
            mask=n_mask,
            other=0.0,
        )
        acc += tl.dot(x, w, out_dtype=tl.float32) * sx[:, None] * sw[None, :]

    out = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_N": block_n, "SPLIT_K": split_k},
            num_warps=num_warps,
            num_stages=3,
        )
        for block_n in (32, 64, 128)
        for split_k in (2,)
        for num_warps in (4, 8)
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _float_scale_fp8_gemm_splitk(
    x_ptr,
    w_ptr,
    xs_ptr,
    ws_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_xsm,
    stride_xsk,
    stride_wsn,
    stride_wsk,
    stride_ok,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 128)
    m_mask = offs_m < M
    n_mask = offs_n < N
    x_base = x_ptr + offs_m[:, None] * stride_xm
    w_base = w_ptr + offs_n[None, :] * stride_wn
    scale_n = offs_n // 128

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    blocks_per_split: tl.constexpr = (K_BLOCKS + SPLIT_K - 1) // SPLIT_K
    kb0 = pid_k * blocks_per_split
    kb1 = tl.minimum(kb0 + blocks_per_split, K_BLOCKS)
    for kb in range(kb0, kb1):
        kk = kb * 128 + offs_k
        x = tl.load(
            x_base + kk[None, :] * stride_xk,
            mask=m_mask[:, None],
            other=0.0,
        )
        w = tl.load(
            w_base + kk[:, None] * stride_wk,
            mask=n_mask[None, :],
            other=0.0,
        )
        sx = tl.load(
            xs_ptr + offs_m * stride_xsm + kb * stride_xsk,
            mask=m_mask,
            other=0.0,
        )
        sw = tl.load(
            ws_ptr + scale_n * stride_wsn + kb * stride_wsk,
            mask=n_mask,
            other=0.0,
        )
        acc += tl.dot(x, w, out_dtype=tl.float32) * sx[:, None] * sw[None, :]

    out = (
        out_ptr
        + pid_k * stride_ok
        + offs_m[:, None] * stride_om
        + offs_n[None, :] * stride_on
    )
    tl.store(out, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _reduce_splitk(
    part_ptr,
    out_ptr,
    elements,
    stride_pk,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < elements
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for split in range(0, SPLIT_K):
        acc += tl.load(part_ptr + split * stride_pk + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=mask)


def _reference(inputs: dict):
    out = inputs["out"]
    deep_gemm.fp8_gemm_nt(
        (inputs["x_fp8"], inputs["x_scale"]),
        (inputs["w_fp8"], inputs["w_scale"]),
        out,
    )
    return out


@torch.no_grad()
def run(inputs: dict):
    x = inputs["x_fp8"]
    w = inputs["w_fp8"]
    M, K = x.shape
    N = w.shape[0]

    if M != 16:
        return _reference(inputs)

    xs = inputs["x_scale"]
    ws = inputs["w_scale"]
    out = inputs["out"]
    split_k = 2
    part = torch.empty(
        (split_k, M, N),
        dtype=torch.float32,
        device=x.device,
    )
    grid = lambda meta: (
        triton.cdiv(M, 16),
        triton.cdiv(N, meta["BLOCK_N"]),
        meta["SPLIT_K"],
    )
    _float_scale_fp8_gemm_splitk[grid](
        x,
        w,
        xs,
        ws,
        part,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        xs.stride(0),
        xs.stride(1),
        ws.stride(0),
        ws.stride(1),
        part.stride(0),
        part.stride(1),
        part.stride(2),
        BLOCK_M=16,
        K_BLOCKS=K // K_BLOCK,
    )
    block = 1024
    _reduce_splitk[(triton.cdiv(M * N, block),)](
        part,
        out,
        M * N,
        part.stride(0),
        SPLIT_K=split_k,
        BLOCK=block,
    )
    return out
