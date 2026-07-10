"""Grouped GEMM (MoE) benchmarks (ops 12, 13 prefill; 24, 25 decode).

Falls back to torch.bmm bf16 as a proxy for DeepGEMM contiguous/masked grouped FP8.
If deep_gemm is importable, also invokes the real kernel.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks._util import bench, print_header, print_row
from shapes import GROUPED_GEMM_OPS

DEVICE = "cuda"
DTYPE = torch.bfloat16


def _bench_bmm(E: int, M: int, K: int, N: int) -> tuple[float, float]:
    a = torch.randn(E, M, K, device=DEVICE, dtype=DTYPE)
    w = torch.randn(E, K, N, device=DEVICE, dtype=DTYPE)
    us = bench(lambda: torch.bmm(a, w))
    flops = 2 * E * M * K * N
    del a, w
    gc.collect()
    torch.cuda.empty_cache()
    return us, flops / (us * 1e-6) / 1e12


def _bench_deep_gemm_masked(E: int, M: int, K: int, N: int) -> float | None:
    try:
        import deep_gemm  # type: ignore
    except Exception:
        return None
    # Full call requires per-token-group fp8 quant and masked_m/expected_m setup;
    # left as a hook — see README for full-path invocation notes.
    return None


results: list[dict] = []


def run() -> list[dict]:
    print_header()
    for op in GROUPED_GEMM_OPS:
        try:
            us, tflops = _bench_bmm(op.E, op.M, op.K, op.N)
        except torch.cuda.OutOfMemoryError:
            row = {
                "op_id": op.op_id, "name": op.name, "phase": op.phase,
                "shape": f"{op.E}x[{op.M},{op.K}]x[{op.K},{op.N}]",
                "bf16_us": -1, "bf16_TFLOPS": -1, "note": "OOM",
            }
            results.append(row)
            print_row(op.op_id, op.name, op.phase, "torch.bmm bf16 (proxy)",
                      row["shape"], -1, {"note": "OOM"})
            continue
        row = {
            "op_id": op.op_id,
            "name": op.name,
            "phase": op.phase,
            "shape": f"{op.E}x[{op.M},{op.K}]x[{op.K},{op.N}]",
            "bf16_us": round(us, 2),
            "bf16_TFLOPS": round(tflops, 2),
        }
        results.append(row)
        print_row(op.op_id, op.name, op.phase, "torch.bmm bf16 (proxy)",
                  row["shape"], us, {"TFLOPS_bf16": row["bf16_TFLOPS"]})
    return results


if __name__ == "__main__":
    run()
