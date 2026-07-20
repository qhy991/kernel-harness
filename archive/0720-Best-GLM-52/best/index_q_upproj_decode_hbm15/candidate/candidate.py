"""GLM-5.2 Indexer Q Up-Projection (decode) — TMA-streamed split-K FP8 GEMM.

out[M,4096] = x_fp8[M,2048] @ w_fp8[4096,2048].T with per-token float32 block
scales (x_scale[M, K/128]) and per-block weight scales (w_scale[N/128, K/128]).

NCU on the acc-based split-K kernel showed it was *latency* bound: warps stalled
on long_scoreboard (global-load latency) at 20-40% occupancy while DRAM sat at
~10%. This kernel loads the weight tiles through the Tensor Memory Accelerator
(TMA): bulk async copies driven by the copy engine + mbarrier, so a couple of
warps keep many W loads in flight and drive a deep software pipeline. That
removes the long_scoreboard stall (measured 6.6 -> 1.2) and lifts the binding
M=32 shape from ~11.7% to ~13% HBM.

Layout: N-major transposed output. W[N,K] is row-major (K contiguous), so a TMA
tile [BLOCK_N, BLOCK_K] keeps K as the contiguous inner dim (a TMA requirement).
We compute acc[BN,BM] = dot(w_tile[BN,BK], trans(x_tile)[BK,BM]) = out[m,n]^T,
scale per K-group in the loop, and store transposed. x is tiny (M<=32) so it
uses a plain masked load. SPLIT_K=2 gives each CTA an 8-group K-loop — deep
enough to fill the TMA pipeline — while doubling the CTA count for occupancy;
the last CTA per N-tile reduces the two FP32 partials (L2-resident) in place.

Config is env-overridable for sweeps; defaults are the committed configuration.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None and v != "" else default


# ── committed configuration (env-overridable) — tuned on B200 GPU0 ──
_BLOCK_N = _env_int("IQU_BLOCK_N", 32)      # N cols per tile; TMA sweet spot at 32
_SPLIT_K = _env_int("IQU_SPLIT_K", 2)       # K-groups split factor; must divide K/128
_NUM_WARPS = _env_int("IQU_NUM_WARPS", 2)   # small CTAs -> more resident per SM
_STAGES = _env_int("IQU_STAGES", 0)         # 0 => auto per shape (M16->6, M32->4)
_GROUP = 128                                 # K quant-group / scale-block size


# Triton's on-device TMA descriptor scratch needs a registered allocator.
triton.set_allocator(
    lambda size, alignment, stream: torch.empty(size, dtype=torch.int8, device="cuda"))


def _stages(M: int) -> int:
    if _STAGES:
        return _STAGES
    return 6 if M <= 16 else 4


@triton.jit
def _tma_splitk_gemm(
    w_desc, x_ptr, sx_ptr, sw_ptr, partial_ptr, lock_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_sxm, stride_sxj,
    stride_swn, stride_swj,
    stride_ps, stride_pn, stride_pm,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUPS_PER_SPLIT: tl.constexpr, SPLIT_K: tl.constexpr, NUM_STAGES: tl.constexpr,
):
    n_tile = tl.program_id(0)
    sid = tl.program_id(1)
    n0 = n_tile * BLOCK_N
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_blk = n0 // 128  # BLOCK_N divides 128, so a tile lies within one weight-scale block

    # acc is transposed [BLOCK_N, BLOCK_M]: acc[i,j] = out[offs_m[j], offs_n[i]].
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    g0 = sid * GROUPS_PER_SPLIT
    for gg in tl.range(GROUPS_PER_SPLIT, num_stages=NUM_STAGES):
        kb = g0 + gg
        k0 = kb * BLOCK_K
        w_t = tl.load_tensor_descriptor(w_desc, [n0, k0])                    # [BN,BK] fp8
        x_t = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xk,
                      mask=m_mask[:, None], other=0.0)                       # [BM,BK] fp8
        dot = tl.dot(w_t, tl.trans(x_t), out_dtype=tl.float32)              # [BN,BM]
        sx = tl.load(sx_ptr + offs_m * stride_sxm + kb * stride_sxj, mask=m_mask, other=0.0)
        sw = tl.load(sw_ptr + n_blk * stride_swn + kb * stride_swj)
        acc += dot * (sw * sx[None, :])

    if SPLIT_K == 1:
        tl.store(out_ptr + offs_n[:, None] * stride_on + offs_m[None, :] * stride_om,
                 acc.to(tl.bfloat16), mask=m_mask[None, :])
        return

    tl.store(partial_ptr + sid * stride_ps + offs_n[:, None] * stride_pn
             + offs_m[None, :] * stride_pm, acc, mask=m_mask[None, :])
    # Last sibling for this N-tile reduces the FP32 partials, writes bf16, and
    # re-arms the semaphore in place — one launch, one reduction, no memset.
    arrived = tl.atomic_add(lock_ptr + n_tile, 1, sem="acq_rel", scope="gpu")
    if arrived == SPLIT_K - 1:
        red = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        for s in tl.static_range(SPLIT_K):
            red += tl.load(partial_ptr + s * stride_ps + offs_n[:, None] * stride_pn
                           + offs_m[None, :] * stride_pm, mask=m_mask[None, :], other=0.0)
        tl.store(out_ptr + offs_n[:, None] * stride_on + offs_m[None, :] * stride_om,
                 red.to(tl.bfloat16), mask=m_mask[None, :])
        tl.store(lock_ptr + n_tile, 0)


# Reused per-(device, stream, n_tiles) semaphore — one int32 per N-tile, re-armed
# in kernel so no per-call memset launch widens the measured device span.
_SEMAPHORES: dict[tuple[int, int, int], torch.Tensor] = {}


def _semaphore(device: torch.device, n_tiles: int) -> torch.Tensor:
    key = (device.index or 0, torch.cuda.current_stream(device).stream_id, n_tiles)
    lock = _SEMAPHORES.get(key)
    if lock is None:
        lock = torch.zeros(n_tiles, dtype=torch.int32, device=device)
        _SEMAPHORES[key] = lock
    return lock


@torch.no_grad()
def run(inputs: dict):
    x = inputs["x_fp8"]
    sx = inputs["x_scale"]
    w = inputs["w_fp8"]
    sw = inputs["w_scale"]
    M, K = x.shape
    N = w.shape[0]

    groups = K // _GROUP
    split_k = _SPLIT_K
    if groups % split_k:
        raise ValueError(f"K/128={groups} not divisible by SPLIT_K={split_k}")
    block_n = _BLOCK_N
    if N % block_n or _GROUP % block_n:
        raise ValueError(f"bad BLOCK_N={block_n} for N={N}")
    n_tiles = N // block_n
    block_m = max(16, triton.next_power_of_2(M))

    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    w_desc = TensorDescriptor.from_tensor(w, [block_n, _GROUP])
    if split_k == 1:
        partial = out
        lock = out
        ps0 = ps1 = ps2 = 0
    else:
        partial = torch.empty((split_k, N, M), dtype=torch.float32, device=x.device)
        lock = _semaphore(x.device, n_tiles)
        ps0, ps1, ps2 = partial.stride(0), partial.stride(1), partial.stride(2)

    _tma_splitk_gemm[(n_tiles, split_k)](
        w_desc, x, sx, sw, partial, lock, out,
        M, N, K,
        x.stride(0), x.stride(1),
        sx.stride(0), sx.stride(1),
        sw.stride(0), sw.stride(1),
        ps0, ps1, ps2,
        out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=_GROUP,
        GROUPS_PER_SPLIT=groups // split_k, SPLIT_K=split_k, NUM_STAGES=_stages(M),
        num_warps=_NUM_WARPS, num_stages=_stages(M),
    )
    return out
