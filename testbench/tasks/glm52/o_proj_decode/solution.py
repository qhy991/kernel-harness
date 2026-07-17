"""FP8 blockwise GEMM — sglang production baseline (deep_gemm w8a8_block_fp8).

Shared recipe for every Kimi-K2.7 dense FP8 linear projection (O_proj, Q_a/KV_a
fused, Q_b, KV_b, Dense FFN GateUp/Down, MoE shared GateUp/Down). The op is fully
described by the (M, K, N) axes in definition.json; this file reads them and calls
the exact kernel sglang dispatches on B200. `reference.py` IS the correctness
oracle AND the latency baseline: an optimized solution.py must match this output
within the per-workload tolerance and run faster.

  out[M, N] = (x_fp8 @ w_fp8.T), 1x128 activation scales + 128x128 weight scales,
  dequantized/accumulated by deep_gemm. Weights are quantized offline (fp8 inputs);
  only the GEMM is timed, matching the kernel-harness `bench_fp8_gemm` measurement.
"""

import torch
import deep_gemm
from deep_gemm import ceil_div, get_tma_aligned_size
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.quantization.fp8_utils import requant_weight_ue8m0

BLOCK = 128


def _per_block_cast_to_fp8(x: torch.Tensor):
    m, n = x.shape
    xp = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device
    )
    xp[:m, :n] = x
    xv = xp.view(-1, 128, xp.size(1) // 128, 128)
    xa = xv.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    xs = (xv * (448.0 / xa)).to(torch.float8_e4m3fn)
    return xs.view_as(xp)[:m, :n].contiguous(), (xa / 448.0).view(
        xv.size(0), xv.size(2)
    )


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate pre-quantized FP8 inputs in the exact deep_gemm/sglang layout.

    Weight quant + ue8m0/tma requant happen here (offline-equivalent, untimed);
    run() only executes the GEMM, so the measured latency is the kernel alone.
    """
    M = axes_and_scalars["M"]
    K = axes_and_scalars["K"]
    N = axes_and_scalars["N"]

    # The ue8m0 activation scale is column-major with the token (M) dim padded to a
    # TMA-aligned stride (deep_gemm asserts As.stride(-1) == get_tma_aligned_size(M, 4)).
    # For M < 4 that padding is a non-contiguous "gap" the harness's per-iteration
    # clone_args() strips, tripping deep_gemm at M=1,2. Padding M up to the aligned
    # size makes the scale naturally contiguous (shape[0] == aligned stride), so it
    # survives the clone. M >= 4 is already aligned -> unchanged. (Same convention the
    # grouped-MoE recipes use, where M is padded to align(M, 128).)
    M = get_tma_aligned_size(M, 4)

    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    # 1/sqrt(K) weight init keeps output magnitude O(1) so tolerances are meaningful.
    w = torch.randn(N, K, device=device, dtype=torch.bfloat16) * (K ** -0.5)

    w_fp8, w_scale = _per_block_cast_to_fp8(w)
    x_fp8, x_scale = sglang_per_token_group_quant_fp8(
        x, BLOCK, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True
    )
    w_fp8, w_scale = requant_weight_ue8m0(w_fp8, w_scale, [BLOCK, BLOCK])

    return {"x_fp8": x_fp8, "x_scale": x_scale, "w_fp8": w_fp8, "w_scale": w_scale}


@torch.no_grad()
def run(x_fp8, x_scale, w_fp8, w_scale):
    # Dispatch deep_gemm.fp8_gemm_nt directly with compiled_dims="nk" so that
    # N=6144 and K=16384 become compile-time template constants (the production
    # sglang wrapper leaves them fully dynamic, compiled_dims=""). The packed
    # UE8M0 int32 scale layout (As[M, K//512], Bs[N, K//512]) is passed through
    # unchanged, exactly as the wrapper hands it to fp8_gemm_nt.
    # A num_sms sweep {32..148} was measured on both shapes: values < 48 slow the
    # kernel (fewer SMs than the 48-tile grid), and values >= 48 tie the default
    # within noise, so no set_num_sms override is applied here.
    M = x_fp8.shape[0]
    N = w_fp8.shape[0]
    out = x_fp8.new_empty((M, N), dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt((x_fp8, x_scale), (w_fp8, w_scale), out, compiled_dims="nk")
    return out
