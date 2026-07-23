"""Overrides that install huyan's tuned o_proj_prefill kernel into sglang.

Measured on the current amd/main branch:
    · M=1024  1.315x vs aiter production dispatch  (calc_diff 2.14e-10)
    · M=2048  1.196x vs aiter production dispatch  (calc_diff 1.99e-10)
    · M=4096  correctness fail (algorithm/layout bug in the tuning wrapper)

So this file gates on M — M<=2048 uses huyan's kernel, M>=4096 falls back
to sglang's aiter_w8a8_block_fp8_linear default. See
`archive/replay-20260723/results.csv` for the audit that produced these numbers.

Usage:
    ./bench_glm5_e2e.py prefill --overrides \
        benchmarks/glm5_e2e/examples/huyan_o_proj_prefill.py

Note: `huyan_o_proj_prefill` targets ONE op (o_proj at prefill). It does not
change dsa_attn, moe, or index_score — the other operators still dispatch
through sglang default. Substitute your own patches similarly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Locate the archived kernel file. Adjust if the repo is checked out somewhere
# other than /root/repos/kernel-harness.
_REPO = Path(__file__).resolve().parents[3]
_ARCHIVE = _REPO / "archive" / "0723-amd-glm52"


def _load_huyan_kernel(target_m: int):
    """Import archive/0723-amd-glm52/o_proj_prefill_m{M}.py and return its run()."""
    p = _ARCHIVE / f"o_proj_prefill_m{target_m}.py"
    if not p.is_file():
        raise FileNotFoundError(f"huyan o_proj kernel not found at {p}")
    # The wrappers `import _amd_kernels` from the same dir; make it findable.
    if str(_ARCHIVE) not in sys.path:
        sys.path.insert(0, str(_ARCHIVE))
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"huyan_o_proj_m{target_m}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run


def register():
    from operator_overrides import patch

    kernel_1024 = _load_huyan_kernel(1024)
    kernel_2048 = _load_huyan_kernel(2048)

    # Save the original for the M>=4096 fallback + for any code path we don't
    # want to touch.
    import sglang.srt.layers.quantization.fp8_utils as fp8u
    _orig = fp8u.aiter_w8a8_block_fp8_linear

    # sglang's `aiter_w8a8_block_fp8_linear` signature (from fp8_utils.py:788):
    #   fn(input, weight, block_size, weight_scale, input_scale=None, bias=None,
    #      weight_original=None) -> torch.Tensor
    # huyan's run(inputs_dict) has a different shape — repack.
    import torch

    def _o_proj_dispatch(input, weight, block_size, weight_scale,
                         input_scale=None, bias=None, weight_original=None):
        m = input.view(-1, input.shape[-1]).shape[0]
        n, k = weight.shape
        # Only o_proj matches (K, N) = (16384, 6144). Everything else goes to the default.
        is_o_proj = (k == 16384 and n == 6144)
        if not is_o_proj or m not in (1024, 2048):
            return _orig(input, weight, block_size, weight_scale,
                         input_scale, bias, weight_original)
        # Repack sglang's arguments into the frozen-inputs dict huyan expects.
        # Sglang caches the FP8 input scale already; we skip the quant round-trip.
        x_fp8 = input.view(-1, k)
        if input_scale is None:
            # sglang usually passes input already quantised; if not, use the
            # aiter per-1x128 helper (imported by fp8_utils itself).
            from sglang.srt.layers.quantization.fp8_utils import aiter_per1x128_quant
            import aiter
            x_fp8, input_scale = aiter_per1x128_quant(
                x_fp8, quant_dtype=aiter.dtypes.fp8, transpose_scale=False,
            )
        inputs = {
            "x_fp8": x_fp8,
            "x_scale": input_scale,
            "w_fp8": weight,
            "w_scale": weight_scale,
            "out": torch.empty(m, n, dtype=torch.bfloat16, device=input.device),
        }
        run = kernel_1024 if m == 1024 else kernel_2048
        out = run(inputs)
        if bias is not None:
            out += bias
        return out.view(*input.shape[:-1], n)

    return [patch(
        "sglang.srt.layers.quantization.fp8_utils.aiter_w8a8_block_fp8_linear",
        _o_proj_dispatch,
    )]
