"""B200 split-K FP8 GEMM for GLM-5.2 index_k decode.

The current harness supplies decoded float32 block scales:
``x_scale[M, K/128]`` and ``w_scale[1, K/128]``.  This is intentionally
different from the older KDA kernel's packed-int32 UE8M0 ABI.  Each of 24 CTAs
accumulates two K blocks into an FP32 partial; the last CTA performs a
deterministic FP32 reduction and writes bf16 output in the same launch.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile

import torch
import triton

_GROUP = 128
_SPLIT_K = 24
_NUM_WARPS = 16
_NUM_STAGES = 2

_KERNEL_SRC = r'''
import triton
import triton.language as tl


@triton.jit
def fused_splitk_reduce_kernel(
    x_ptr, w_ptr, sx_ptr, sw_ptr, partial_ptr, lock_ptr, out_ptr,
    M, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_sxm, stride_sxj,
    stride_swj,
    stride_ps, stride_pm, stride_pn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUPS_PER_SPLIT: tl.constexpr, SPLIT_K: tl.constexpr,
):
    sid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    g0 = sid * GROUPS_PER_SPLIT
    for gg in tl.static_range(GROUPS_PER_SPLIT):
        kb = g0 + gg
        k0 = kb * BLOCK_K
        x = tl.load(x_ptrs + k0 * stride_xk, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs + k0 * stride_wk)
        dot = tl.dot(x, w, out_dtype=tl.float32)

        sx = tl.load(
            sx_ptr + offs_m * stride_sxm + kb * stride_sxj,
            mask=m_mask,
            other=0.0,
        )
        sw = tl.load(sw_ptr + kb * stride_swj)
        acc += dot * (sx[:, None] * sw)

    p_ptrs = (
        partial_ptr
        + sid * stride_ps
        + offs_m[:, None] * stride_pm
        + offs_n[None, :] * stride_pn
    )
    tl.store(p_ptrs, acc, mask=m_mask[:, None])

    # The release publishes this CTA's partial.  The final arrival acquires all
    # sibling stores, reduces them in a fixed order, then re-arms the semaphore.
    arrived = tl.atomic_add(lock_ptr, 1, sem="acq_rel", scope="gpu")
    if arrived == SPLIT_K - 1:
        out_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for s in tl.static_range(SPLIT_K):
            p = tl.load(
                partial_ptr
                + s * stride_ps
                + offs_m[:, None] * stride_pm
                + offs_n[None, :] * stride_pn,
                mask=m_mask[:, None],
                other=0.0,
            )
            out_acc += p
        o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(o_ptrs, out_acc.to(tl.bfloat16), mask=m_mask[:, None])
        tl.store(lock_ptr, 0)
'''


def _load_kernel():
    digest = hashlib.sha1(_KERNEL_SRC.encode()).hexdigest()[:16]
    name = f"_glm52_index_k_{digest}"
    if name in sys.modules:
        return sys.modules[name]
    directory = tempfile.mkdtemp(prefix="glm52_index_k_")
    path = os.path.join(directory, name + ".py")
    with open(path, "w") as f:
        f.write(_KERNEL_SRC)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_KERNEL = _load_kernel()
_SEMAPHORES: dict[tuple[int, int], torch.Tensor] = {}


def _semaphore(device: torch.device) -> torch.Tensor:
    key = (device.index or 0, torch.cuda.current_stream(device).stream_id)
    lock = _SEMAPHORES.get(key)
    if lock is None:
        lock = torch.zeros(1, dtype=torch.int32, device=device)
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
    if groups % _SPLIT_K:
        raise ValueError(f"K/128={groups} must be divisible by split_k={_SPLIT_K}")

    partial = torch.empty((_SPLIT_K, M, N), dtype=torch.float32, device=x.device)
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    block_m = max(16, triton.next_power_of_2(M))
    _KERNEL.fused_splitk_reduce_kernel[(_SPLIT_K,)](
        x, w, sx, sw, partial, _semaphore(x.device), out,
        M, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        sx.stride(0), sx.stride(1),
        sw.stride(1),
        partial.stride(0), partial.stride(1), partial.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=N,
        BLOCK_K=_GROUP,
        GROUPS_PER_SPLIT=groups // _SPLIT_K,
        SPLIT_K=_SPLIT_K,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return out
