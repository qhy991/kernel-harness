"""SwiGLU (SiLU-and-mul) — sglang production baseline (sgl_kernel.silu_and_mul).

Kimi-K2.7 FFN activation: the exact kernel sglang dispatches in
SiluAndMul.forward_cuda (python/sglang/srt/layers/activation.py:102). Splits the
gate/up projection output [M, 2I] into gate|up along the last dim and computes
silu(gate) * up -> [M, I]. `reference.py` IS the correctness oracle AND the latency
baseline: an optimized solution.py must match this output within tolerance and run
faster.

  out[M, I] = silu(x[:, :I]) * x[:, I:],  I = (2I)//2.  Memory-bound elementwise.

Only the kernel is timed; the (untimed) get_inputs generates the gate|up tensor.
The output buffer is allocated inside run(), exactly as sglang's forward_cuda does.
"""

import torch
from sgl_kernel import silu_and_mul


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    M = axes_and_scalars["M"]
    I2 = axes_and_scalars["I2"]  # 2 * intermediate/TP (gate|up concatenated)
    x = torch.randn(M, I2, device=device, dtype=torch.bfloat16)
    return {"x": x}


@torch.no_grad()
def run(x):
    d = x.shape[-1] // 2
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    silu_and_mul(x, out)
    return out
