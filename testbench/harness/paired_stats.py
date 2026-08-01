"""Pure helpers for drift-resistant paired benchmark statistics.

The GPU runner deliberately keeps this module free of torch/CUDA imports so the
pairing contract can be regression-tested in the GPU-free harness selftest.
"""
from __future__ import annotations

import math
import statistics
from typing import Sequence


CONSERVATIVE_QUANTILE = 0.10
TIMING_SPREAD_LIMIT = 1.25


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (NumPy's default method), stdlib only."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize_pairs(candidate_ms: Sequence[float], reference_ms: Sequence[float]) -> dict:
    """Summarize adjacent reference/candidate pairs.

    The primary speedups are quantiles of per-pair ``reference / candidate``
    ratios.  Legacy independent-tail values remain available for archive
    comparisons, but they never decide the current gate.
    """
    if len(candidate_ms) != len(reference_ms) or not candidate_ms:
        raise ValueError("candidate/reference samples must be non-empty and aligned")
    candidate = [float(value) for value in candidate_ms]
    reference = [float(value) for value in reference_ms]
    if any(not math.isfinite(value) or value <= 0.0 for value in candidate + reference):
        raise ValueError("timing samples must be finite and positive")

    ratios = [ref / cand for ref, cand in zip(reference, candidate)]
    cand_lo, cand_hi = min(candidate), max(candidate)
    ref_lo, ref_hi = min(reference), max(reference)
    spread = max(cand_hi / cand_lo, ref_hi / ref_lo)
    q = CONSERVATIVE_QUANTILE
    return {
        "candidate_min_ms": cand_lo,
        "candidate_median_ms": statistics.median(candidate),
        "candidate_max_ms": cand_hi,
        "reference_min_ms": ref_lo,
        "reference_median_ms": statistics.median(reference),
        "reference_max_ms": ref_hi,
        "paired_ratios": ratios,
        "speedup_median": statistics.median(ratios),
        "speedup_conservative": percentile(ratios, q),
        "speedup_optimistic": percentile(ratios, 1.0 - q),
        "speedup_unpaired_median": statistics.median(reference) / statistics.median(candidate),
        "speedup_unpaired_conservative": (
            percentile(reference, q) / percentile(candidate, 1.0 - q)
        ),
        "speedup_unpaired_optimistic": (
            percentile(reference, 1.0 - q) / percentile(candidate, q)
        ),
        "timing_spread": spread,
        "timing_unstable": spread > TIMING_SPREAD_LIMIT,
    }


def balanced_orders(repeat: int) -> list[str]:
    """Return an alternating R/C, C/R capture plan for ``repeat`` pairs."""
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    return ["reference,candidate" if index % 2 == 0 else "candidate,reference"
            for index in range(repeat)]
