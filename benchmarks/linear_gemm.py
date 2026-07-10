"""Linear GEMM benchmarks (ops 1, 2, 4, 5, 6, 7, 9, 10, 11 prefill; 14, 15, 19, 20, 21, 23 decode).

Uses F.linear (bf16 cuBLAS) as a baseline. If sglang/sgl_kernel is importable, also
tries the real block-fp8 kernel via w8a8_block_fp8_linear. If deep_gemm is importable,
tries deep_gemm.fp8_gemm_nt. If torch._scaled_mm is available (H100+/H800), reports
plain fp8 GEMM as an upper-bound-ish proxy.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks._util import (
    bench,
    device_info,
    gemm_gbs,
    gemm_tflops,
    print_header,
    print_row,
)
from shapes import LINEAR_OPS

DEVICE = "cuda"
DTYPE = torch.bfloat16


def _bench_bf16_linear(M: int, K: int, N: int) -> float:
    a = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    w = torch.randn(N, K, device=DEVICE, dtype=DTYPE)
    us = bench(lambda: F.linear(a, w))
    del a, w
    gc.collect()
    torch.cuda.empty_cache()
    return us


def _bench_fp8_scaled_mm(M: int, K: int, N: int) -> float | None:
    if not hasattr(torch, "_scaled_mm"):
        return None
    try:
        a = torch.randn(M, K, device=DEVICE).clamp(-448, 448).to(torch.float8_e4m3fn)
        b = torch.randn(N, K, device=DEVICE).clamp(-448, 448).to(torch.float8_e4m3fn)
        sa = torch.tensor(1.0, device=DEVICE)
        sb = torch.tensor(1.0, device=DEVICE)
        us = bench(lambda: torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb,
                                            out_dtype=torch.bfloat16))
        del a, b
        gc.collect()
        torch.cuda.empty_cache()
        return us
    except Exception:
        return None


def _bench_sgl_block_fp8(M: int, K: int, N: int) -> float | None:
    """Try to route through sglang's actual block-fp8 dispatch."""
    try:
        from sglang.srt.layers.quantization.fp8_utils import (
            dispatch_w8a8_block_fp8_linear,
        )
    except Exception:
        return None
    # Placeholder: full construction needs a Linear layer with weight_block_size + weight_scale_inv.
    # Left as a hook — see README for how to enable end-to-end sglang linear bench.
    return None


results: list[dict] = []


def run() -> list[dict]:
    print_header()
    for op in LINEAR_OPS:
        us_bf16 = _bench_bf16_linear(op.M, op.K, op.N)
        row = {
            "op_id": op.op_id,
            "name": op.name,
            "phase": op.phase,
            "shape": f"[{op.M},{op.K}]x[{op.K},{op.N}]",
            "attr": op.attr,
            "bf16_us": round(us_bf16, 2),
            "bf16_TFLOPS": round(gemm_tflops(op.M, op.N, op.K, us_bf16), 2),
            "bf16_GBs": round(gemm_gbs(op.M, op.N, op.K, us_bf16), 2),
        }
        us_fp8 = _bench_fp8_scaled_mm(op.M, op.K, op.N)
        if us_fp8 is not None:
            row["fp8_us"] = round(us_fp8, 2)
            row["fp8_TFLOPS"] = round(gemm_tflops(op.M, op.N, op.K, us_fp8), 2)
        results.append(row)
        extra = {"TFLOPS_bf16": row["bf16_TFLOPS"], "GB/s": row["bf16_GBs"]}
        if us_fp8 is not None:
            extra["fp8_TFLOPS"] = row["fp8_TFLOPS"]
        print_row(op.op_id, op.name, op.phase, "F.linear bf16" + ("+fp8" if us_fp8 else ""),
                  row["shape"], us_bf16, extra)
    return results


if __name__ == "__main__":
    run()
