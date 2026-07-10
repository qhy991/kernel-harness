"""Absorb-BMM benchmarks (ops 17, 18) — decode MLA q_nope and v.

These are the ACTUAL sglang path: forward_mla.py:449-538 (q_nope) and :822-973 (v)
run torch.bmm on bf16 weights by default. FP8 variants (bmm_fp8 / grouped_gemm_masked)
are alternate paths when SGL_USE_DEEPGEMM_BMM is set at load time.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks._util import bench, print_header, print_row
from shapes import BMM_OPS

DEVICE = "cuda"
DTYPE = torch.bfloat16


def _bench_bmm(M: int, H: int, IN: int, OUT: int) -> tuple[float, float]:
    q = torch.randn(H, M, IN, device=DEVICE, dtype=DTYPE)
    w = torch.randn(H, IN, OUT, device=DEVICE, dtype=DTYPE)
    us = bench(lambda: torch.bmm(q, w))
    flops = 2 * H * M * IN * OUT
    del q, w
    gc.collect()
    torch.cuda.empty_cache()
    return us, flops / (us * 1e-6) / 1e12


results: list[dict] = []


def run() -> list[dict]:
    print_header()
    for op in BMM_OPS:
        us, tflops = _bench_bmm(op.M, op.H, op.IN, op.OUT)
        row = {
            "op_id": op.op_id,
            "name": op.name,
            "phase": op.phase,
            "shape": f"[{op.M},{op.H},{op.IN}]x[{op.H},{op.IN},{op.OUT}]",
            "bf16_us": round(us, 2),
            "bf16_TFLOPS": round(tflops, 2),
            "note": op.note,
        }
        results.append(row)
        print_row(op.op_id, op.name, op.phase, "torch.bmm bf16 (ACTUAL path)",
                  row["shape"], us, {"TFLOPS_bf16": row["bf16_TFLOPS"]})
    return results


if __name__ == "__main__":
    run()
