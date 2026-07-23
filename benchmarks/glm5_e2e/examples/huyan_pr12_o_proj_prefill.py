"""Install the PR #12 tuned o_proj_prefill kernel (as inlined into
testbench/tasks/glm52_amd/o_proj_prefill/candidate.py) into sglang.

Op-level tests on this kernel (archive/replay-20260723-pr12-verify/):
  · M=1024  median 1.6x  · sp_cons 0.023x  (tail latency ~70x above median)
  · M=2048  median 1.8x  · sp_cons 0.608x  (below unity — real p90 regression)
  · M=4096  median 0.05x · sp_cons 0.019x  (**21x REGRESSION**)

So this override is expected to make e2e prefill SLOWER at M=4096, and
possibly at every shape once p90 tail hits. Gate on M and fall back for
M=4096 to avoid the known regression. M=1024/2048 still exposed to the
tail-latency risk — the e2e run measures that.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_CAND_PATH = _REPO / "testbench" / "tasks" / "glm52_amd" / "o_proj_prefill" / "candidate.py"


def _load_pr12_candidate_run():
    """Import the inlined tuned kernel from the task tree and return its run()."""
    if not _CAND_PATH.is_file():
        raise FileNotFoundError(f"PR #12 tuned o_proj candidate not found at {_CAND_PATH}")
    spec = importlib.util.spec_from_file_location("_pr12_o_proj_candidate", _CAND_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.run


def register():
    from operator_overrides import patch
    import torch

    tuned_run = _load_pr12_candidate_run()

    import sglang.srt.layers.quantization.fp8_utils as fp8u
    _orig = fp8u.aiter_w8a8_block_fp8_linear

    def _o_proj_dispatch(input, weight, block_size, weight_scale,
                         input_scale=None, bias=None, weight_original=None):
        # sglang shape: input [..., K], weight [N, K]. o_proj is K=16384, N=6144.
        # Route only o_proj through the tuned kernel; everything else stays on
        # sglang's default aiter dispatch. M=4096 also falls back — the PR #12
        # kernel has a documented 21x regression at that shape.
        m = input.view(-1, input.shape[-1]).shape[0]
        n, k = weight.shape
        is_o_proj = (k == 16384 and n == 6144)
        if not is_o_proj or m not in (1024, 2048):
            return _orig(input, weight, block_size, weight_scale,
                         input_scale, bias, weight_original)

        # Repack sglang args into the candidate's frozen-inputs dict schema.
        x_fp8 = input.view(-1, k)
        if input_scale is None:
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
        out = tuned_run(inputs)
        if bias is not None:
            out = out + bias
        return out.view(*input.shape[:-1], n)

    return [patch(
        "sglang.srt.layers.quantization.fp8_utils.aiter_w8a8_block_fp8_linear",
        _o_proj_dispatch,
    )]
