"""gate_v1_fusednq — GLM-5.2 norm_quant_gate / prefill.

One fused CUDA kernel replaces the region's first two stock kernels:

    sgl_kernel.fused_add_rmsnorm            (flashinfer FusedAddRMSNormKernel)
    sglang_per_token_group_quant_fp8(ue8m0) (sglang per_token_group_quant_8bit_v2)

then hands the SAME (x_fp8, packed-UE8M0 x_scale) pair to the SAME
`deep_gemm.fp8_gemm_nt` the reference calls.  No FP8 GEMM is hand-rolled; the
fusion is strictly around it (plan §3).

Why it is worth doing (measured on this box, CUDA-graph replay, cold L2):
the stock pair spills the normalized [M,K] bf16 activation to HBM and reads it
straight back, but nothing outside the region consumes it — the contract gates
only (out, residual).  Fusing removes 4*M*K bytes of HBM traffic and one graph
node.  Stock norm+quant is 37% / 45% / 50% of the graph region at M=1024/2048/4096.

Numerics: bit-exact against the stock pair, and that is mandatory rather than
nice-to-have — the harness rejected an earlier build that differed in 4 fp8 bytes
out of 6291456.  See fused_norm_quant.cu for the operation-for-operation
correspondence with flashinfer's CuTe-DSL reduction; tests/test_bitexact.py
enforces it on raw bytes and tests/solve_rms.py derives it from the reference's
own output.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_DEFAULT_BUILD = Path("/tmp") / f"kernel-harness-{os.getuid()}" / _HERE.name
_BUILD = Path(os.environ.get(
    "GATE_NQ_BUILD_DIR", _DEFAULT_BUILD))
_BUILD.mkdir(parents=True, exist_ok=True)


# Rows per block: free parameter (the 64-thread-per-row reduction shape is pinned
# by bit-exactness, how many rows share a block is not).  Swept in evidence/.
ROWS_PER_BLOCK = int(os.environ.get("GATE_NQ_ROWS", "2"))
# 1 = cp.async-into-smem staging, 0 = carry h in registers.  Swept in evidence/.
SMEM_STAGE = int(os.environ.get("GATE_NQ_SMEM", "0"))

from torch.utils.cpp_extension import load as _load  # noqa: E402

_ext = _load(
    name="gate_v1_fusednq_ext",
    sources=[str(_HERE / "fused_norm_quant.cu")],
    # NO --use_fast_math: div_mode 0 must stay IEEE div.rn.f32 (see .cu header).
    extra_cuda_cflags=["-O3", "-lineinfo", "--expt-relaxed-constexpr"],
    build_directory=str(_BUILD),
    verbose=False,
)

import deep_gemm  # noqa: E402


def _ceil_align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def _alloc_quant_outputs(M: int, K: int, device):
    """Byte-for-byte the buffers sglang's own wrapper allocates.

    Mirrors sglang.srt.layers.quantization.fp8_kernel.
    create_per_token_group_quant_fp8_output_scale(column_major_scales=True,
    scale_tma_aligned=True, scale_ue8m0=True): an int32 (aligned_k//4, aligned_mn)
    buffer viewed transposed, i.e. logical (M, K//512) with stride (1, aligned_mn).
    """
    x_q = torch.empty((M, K), device=device, dtype=torch.float8_e4m3fn)
    s_mn, s_k = M, K // 128
    aligned_mn, aligned_k = _ceil_align(s_mn, 4), _ceil_align(s_k, 4)
    x_s = torch.empty((aligned_k // 4, aligned_mn), device=device,
                      dtype=torch.int32).transpose(-1, -2)[:s_mn, :]
    return x_q, x_s


def run(inputs: dict):
    hidden, residual = inputs["hidden"], inputs["residual"]
    M, K = hidden.shape
    x_fp8, x_scale = _alloc_quant_outputs(M, K, hidden.device)
    _ext.fused_add_rmsnorm_quant_ue8m0(
        hidden, residual, inputs["norm_weight"], x_fp8, x_scale,
        inputs["eps"], ROWS_PER_BLOCK, SMEM_STAGE)
    out = inputs["out"]
    deep_gemm.fp8_gemm_nt((x_fp8, x_scale),
                          (inputs["w_fp8"], inputs["w_scale"]), out)
    return out, residual
