"""Native packed-UE8M0 Triton candidate for GLM-5.2 indexer.wq_b decode.

This is the serving-ABI adaptation of the archived split-free skinny-M kernel
under ``archive/0720-Best-GLM-52/best-hechenxi-0720/index_q_upproj_decode``.
It consumes the production int32 TMA-aligned scale tensors directly. There is
no unpack kernel, scale copy, or representation change in the timed path.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


_GROUP_K = 128
_OUTPUTS: dict[tuple[int, int, int, int], torch.Tensor] = {}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


@triton.jit
def _unpack_ue8m0(word, group):
    exponent = (word >> ((group & 3) * 8)) & 0xFF
    return (exponent << 23).to(tl.float32, bitcast=True)


@triton.jit
def _packed_ue8m0_gemm(
    weight_ptr,
    x_ptr,
    x_scale_ptr,
    weight_scale_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_wn,
    stride_wk,
    stride_xm,
    stride_xk,
    stride_sxm,
    stride_sxpack,
    stride_swn,
    stride_swpack,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    n0 = tl.program_id(0) * BLOCK_N
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for group_k in tl.range(K // BLOCK_K, num_stages=NUM_STAGES):
        k0 = group_k * BLOCK_K
        weight = tl.load(
            weight_ptr
            + offs_n[:, None] * stride_wn
            + (k0 + offs_k)[None, :] * stride_wk
        )
        x = tl.load(
            x_ptr
            + offs_m[:, None] * stride_xm
            + (k0 + offs_k)[None, :] * stride_xk,
            mask=m_mask[:, None],
            other=0.0,
        )
        dot = tl.dot(weight, tl.trans(x), out_dtype=tl.float32)

        packed_k = group_k // 4
        x_scale_word = tl.load(
            x_scale_ptr
            + offs_m * stride_sxm
            + packed_k * stride_sxpack,
            mask=m_mask,
            other=0,
        )
        # Production weight scales are replicated for every row in each
        # 128-row quantization block. BLOCK_N divides 128, so n0 is a valid
        # representative for every output row in this CTA.
        weight_scale_word = tl.load(
            weight_scale_ptr
            + n0 * stride_swn
            + packed_k * stride_swpack
        )
        x_scale = _unpack_ue8m0(x_scale_word, group_k)
        weight_scale = _unpack_ue8m0(weight_scale_word, group_k)
        acc += dot * (weight_scale * x_scale[None, :])

    tl.store(
        out_ptr
        + offs_n[:, None] * stride_on
        + offs_m[None, :] * stride_om,
        acc.to(tl.bfloat16),
        mask=m_mask[None, :],
    )


def _config(m: int) -> tuple[int, int, int]:
    block_n = _env_int("INDEXER_WQ_B_BLOCK_N", 16)
    if m <= 16:
        num_warps = _env_int("INDEXER_WQ_B_NUM_WARPS", 2)
        num_stages = _env_int("INDEXER_WQ_B_NUM_STAGES", 8)
    else:
        num_warps = _env_int("INDEXER_WQ_B_NUM_WARPS", 4)
        num_stages = _env_int("INDEXER_WQ_B_NUM_STAGES", 6)
    return block_n, num_warps, num_stages


def _output(x: torch.Tensor, m: int, n: int) -> torch.Tensor:
    stream_id = torch.cuda.current_stream(x.device).stream_id
    key = (x.device.index or 0, stream_id, m, n)
    out = _OUTPUTS.get(key)
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _OUTPUTS[key] = out
    return out


def _is_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> bool:
    m, k = x.shape
    n = weight.shape[0]
    return (
        m in (16, 32)
        and n == 4096
        and k == 2048
        and x.dtype == torch.float8_e4m3fn
        and weight.dtype == torch.float8_e4m3fn
        and x_scale.dtype == torch.int32
        and weight_scale.dtype == torch.int32
    )


@torch.no_grad()
def run_indexer_wq_b_packed(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    """Launch the rejected native-ABI experiment, or report unsupported."""
    if not _is_supported(x, weight, x_scale, weight_scale):
        return False

    m, k = x.shape
    n = weight.shape[0]
    if (
        out.shape != (m, n)
        or out.dtype != torch.bfloat16
        or out.device != x.device
    ):
        return False

    block_n, num_warps, num_stages = _config(m)
    if n % block_n != 0 or _GROUP_K % block_n != 0:
        raise ValueError(f"BLOCK_N={block_n} must divide N={n} and 128")

    block_m = max(16, triton.next_power_of_2(m))
    _packed_ue8m0_gemm[(n // block_n,)](
        weight,
        x,
        x_scale,
        weight_scale,
        out,
        m,
        N=n,
        K=k,
        stride_wn=weight.stride(0),
        stride_wk=weight.stride(1),
        stride_xm=x.stride(0),
        stride_xk=x.stride(1),
        stride_sxm=x_scale.stride(0),
        stride_sxpack=x_scale.stride(1),
        stride_swn=weight_scale.stride(0),
        stride_swpack=weight_scale.stride(1),
        stride_om=out.stride(0),
        stride_on=out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=_GROUP_K,
        NUM_STAGES=num_stages,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return True


@torch.no_grad()
def run(inputs: dict, runtime):
    x = inputs["x_fp8"]
    weight = inputs["weight_fp8"]
    x_scale = inputs["x_scale"]
    weight_scale = inputs["weight_scale"]
    m = x.shape[0]
    n = weight.shape[0]

    if not _is_supported(x, weight, x_scale, weight_scale):
        return runtime.reference(inputs)

    out = _output(x, m, n)
    if not run_indexer_wq_b_packed(
        x, weight, x_scale, weight_scale, out
    ):
        return runtime.reference(inputs)
    return out
