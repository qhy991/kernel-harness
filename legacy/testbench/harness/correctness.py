"""Correctness comparison — self-contained extraction of the kernel-harness method.

torch.allclose-style |a-b| <= atol + rtol*|b|, aggregated to a matched-ratio, with an
inf/nan sanity gate and an optional hard max-error cap. Returns a plain dictionary.
"""
from __future__ import annotations

import torch

DEFAULT_TOL = {"max_atol": 0.02, "max_rtol": 0.01, "required_matched_ratio": 0.999,
               "max_error_cap": None}


def _sanity(x, y):
    """Fail on inf/nan in either tensor. Returns a result dict on failure, else None."""
    x_bad = ~torch.isfinite(x)
    y_bad = ~torch.isfinite(y)
    if x_bad.any() or y_bad.any():
        has_nan = bool(torch.isnan(x).any() or torch.isnan(y).any())
        return {"max_absolute_error": None, "max_relative_error": None,
                "has_nan": has_nan, "has_inf": not has_nan}
    return None


def compute_error_stats(output: torch.Tensor, reference: torch.Tensor, tol: dict):
    """Returns (result_dict, exceeds_tol). result_dict has max_absolute_error,
    max_relative_error, has_nan, has_inf."""
    x = output.to(torch.float32)
    y = reference.to(torch.float32)

    bad = _sanity(x, y)
    if bad is not None:
        return bad, True

    abs_err = torch.abs(x - y)
    n = abs_err.numel()
    if n == 0:
        return {"max_absolute_error": 0.0, "max_relative_error": 0.0,
                "has_nan": False, "has_inf": False}, False

    max_abs = float(abs_err.max().item())
    atol = tol["max_atol"]
    rtol = tol["max_rtol"]
    exceeds_mask = (abs_err > (atol + rtol * torch.abs(y))) | ~torch.isfinite(abs_err)
    matched_ratio = max(0.0, min(1.0, 1.0 - float(exceeds_mask.sum().item()) / n))

    exceeds = matched_ratio < tol["required_matched_ratio"]
    cap = tol.get("max_error_cap")
    if cap is not None and max_abs > cap:
        exceeds = True

    rel_err = abs_err / torch.clamp(torch.abs(y), min=atol)
    return {"max_absolute_error": max_abs, "max_relative_error": float(rel_err.max().item()),
            "has_nan": False, "has_inf": False}, exceeds
