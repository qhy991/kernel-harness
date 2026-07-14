"""Read-only diagnostic metrics for agent feedback.

These numbers never gate WIN/lose — evaluate.py's CUPTI correctness + conservative
speedup still does. Metrics are derived from (axes, reference-built inputs, measured
latency) so a candidate cannot reward-hack them by returning forged telemetry.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from .inputs import eval_expr

# Approximate NVIDIA B200 peaks (context only; advisory profiler uses the same).
HBM_GBPS = 8000.0
FP_PEAK_TFLOPS = 2250.0


def _bytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _tensor_stats(x: torch.Tensor) -> dict[str, float]:
    x64 = x.detach().float().reshape(-1)
    if x64.numel() == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "sum": 0.0}
    return {
        "min": float(x64.min().item()),
        "max": float(x64.max().item()),
        "mean": float(x64.mean().item()),
        "std": float(x64.std(unbiased=False).item()) if x64.numel() > 1 else 0.0,
        "sum": float(x64.sum().item()),
    }


def gemm_metrics(axes: dict[str, int], latency_us: Optional[float],
                 flops: Optional[int], bytes_moved: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "kind": "gemm",
        "M": axes.get("M"),
        "K": axes.get("K"),
        "N": axes.get("N"),
        "bytes_moved": bytes_moved,
    }
    if latency_us and latency_us > 0:
        out["latency_us"] = round(latency_us, 3)
        out["us_per_token"] = round(_safe_div(latency_us, max(axes.get("M", 1), 1)), 4)
        out["tokens_per_s"] = round(_safe_div(1e6 * axes.get("M", 0), latency_us), 1)
        out["achieved_gbps"] = round(_safe_div(bytes_moved, latency_us * 1e3), 1)
        out["pct_of_hbm_peak"] = round(100.0 * out["achieved_gbps"] / HBM_GBPS, 2)
        if flops:
            tflops = flops / (latency_us * 1e-6) / 1e12
            ai = _safe_div(flops, bytes_moved)
            ridge = FP_PEAK_TFLOPS * 1e12 / (HBM_GBPS * 1e9)
            out.update({
                "flops": flops,
                "achieved_tflops": round(tflops, 2),
                "arithmetic_intensity": round(ai, 3),
                "ridge_point": round(ridge, 2),
                "pct_of_fp_peak": round(100.0 * tflops / FP_PEAK_TFLOPS, 2),
                "bound": "compute-bound" if ai >= ridge else "memory-bound",
            })
    return out


def sparse_mla_metrics(axes: dict[str, int], inputs: dict[str, torch.Tensor],
                       latency_us: Optional[float]) -> dict[str, Any]:
    """Sparse MLA: effective selected-KV traffic and cache footprint."""
    M = axes.get("M", 0)
    ctx = axes.get("ctx", axes.get("seq_len", 0))
    topk = axes.get("topk", axes.get("index_topk", 0))
    heads = axes.get("num_heads", axes.get("H", 0))
    head_dim = axes.get("head_dim", axes.get("D", 0))
    kv_lora = axes.get("kv_lora", axes.get("DV", 0))

    indices = inputs.get("block_tables")
    if indices is None:
        indices = inputs.get("indices")
    if indices is None:
        indices = inputs.get("topk_idx")

    if indices is not None and isinstance(indices, torch.Tensor):
        # Valid entries are >= 0 (invalid pad = -1).
        valid = int((indices >= 0).sum().item())
        total = int(indices.numel())
    else:
        valid = M * max(min(topk, ctx) if topk and ctx else topk, 0)
        total = M * max(topk, 0)

    eff_topk = _safe_div(valid, max(M, 1))
    # Approximate selected KV bytes: each selected token contributes head_dim (FP8)
    # plus rope bf16 if packed, but TRTLLM stores packed 576-dim FP8 → 576 bytes/token.
    bytes_per_kv = head_dim if head_dim else 576
    selected_kv_bytes = valid * bytes_per_kv
    q_bytes = M * heads * head_dim * 2  # bf16 query
    out_bytes = M * heads * (kv_lora or 512) * 2
    cache = inputs.get("kv_cache")
    cache_bytes = _bytes(cache) if isinstance(cache, torch.Tensor) else 0

    out: dict[str, Any] = {
        "kind": "sparse-mla",
        "batch": M,
        "ctx": ctx,
        "topk_nominal": topk,
        "valid_selected_kv": valid,
        "index_slots": total,
        "effective_topk": round(eff_topk, 2),
        "effective_topk_ratio": round(_safe_div(eff_topk, max(topk, 1)), 4),
        "selected_kv_bytes": selected_kv_bytes,
        "cache_footprint_bytes": cache_bytes,
        "q_bytes": q_bytes,
        "out_bytes": out_bytes,
    }
    if latency_us and latency_us > 0:
        out["latency_us"] = round(latency_us, 3)
        out["us_per_token"] = round(_safe_div(latency_us, max(M, 1)), 4)
        out["selected_kv_per_s"] = round(_safe_div(1e6 * valid, latency_us), 1)
        out["selected_token_head_per_s"] = round(
            _safe_div(1e6 * valid * max(heads, 1), latency_us), 1)
        traffic = selected_kv_bytes + q_bytes + out_bytes
        out["effective_kv_gbps"] = round(_safe_div(traffic, latency_us * 1e3), 1)
        out["pct_of_hbm_peak"] = round(100.0 * out["effective_kv_gbps"] / HBM_GBPS, 2)
    return out


def routed_expert_metrics(axes: dict[str, int], inputs: dict[str, torch.Tensor],
                          latency_us: Optional[float],
                          flops_expr: Optional[str] = None) -> dict[str, Any]:
    """Routed Gate+Up / Down: load imbalance + useful-vs-padded utilization."""
    E = axes.get("E", 0)
    M = axes.get("M", 0)   # for masked: tokens/expert pad; for contig: tokens/expert mean
    K = axes.get("K", 0)
    N = axes.get("N", 0)

    masked_m = inputs.get("masked_m")
    m_indices = inputs.get("m_indices")

    if isinstance(masked_m, torch.Tensor) and masked_m.numel() > 0:
        counts = masked_m.detach().to(torch.int64).cpu()
        active = int((counts > 0).sum().item())
        empty = int(E - active)
        total_assign = int(counts.sum().item())
        pad_cap = int(counts.numel() * max(M, 1))
        st = _tensor_stats(counts.float())
        layout = "masked"
    elif isinstance(m_indices, torch.Tensor) and m_indices.numel() > 0:
        flat = m_indices.detach().to(torch.int64).reshape(-1)
        valid = flat[flat >= 0]
        total_assign = int(valid.numel())
        if E > 0:
            counts = torch.bincount(valid, minlength=E)[:E]
        else:
            counts = torch.zeros(0, dtype=torch.int64)
        active = int((counts > 0).sum().item()) if counts.numel() else 0
        empty = int(E - active)
        pad_cap = int(flat.numel())
        st = _tensor_stats(counts.float()) if counts.numel() else {
            "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "sum": 0.0}
        layout = "contiguous"
    else:
        total_assign = M * max(E, 1)
        active, empty, pad_cap = E, 0, total_assign
        st = {"min": float(M), "max": float(M), "mean": float(M), "std": 0.0,
              "sum": float(total_assign)}
        layout = "unknown"

    cv = _safe_div(st["std"], st["mean"]) if st["mean"] else 0.0
    util = _safe_div(total_assign, pad_cap)
    useful_flops = 2 * total_assign * K * N if (K and N) else None
    padded_flops = None
    if flops_expr:
        try:
            padded_flops = eval_expr(flops_expr, axes)
        except Exception:
            padded_flops = None
    if padded_flops is None and K and N:
        # masked path pads to align(M,128) * E in the recipe; flops_expr preferred
        padded_flops = 2 * pad_cap * K * N

    out: dict[str, Any] = {
        "kind": "routed-expert",
        "layout": layout,
        "local_experts": E,
        "active_experts": active,
        "empty_experts": empty,
        "local_assignments": total_assign,
        "tokens_per_expert_min": st["min"],
        "tokens_per_expert_mean": round(st["mean"], 3),
        "tokens_per_expert_max": st["max"],
        "tokens_per_expert_cv": round(cv, 4),
        "load_imbalance": round(cv, 4),
        "pad_capacity": pad_cap,
        "padding_ratio": round(1.0 - util, 4),
        "useful_vs_padded_util": round(util, 4),
        "useful_flops": useful_flops,
        "padded_flops": padded_flops,
    }
    if latency_us and latency_us > 0:
        out["latency_us"] = round(latency_us, 3)
        out["us_per_assignment"] = round(
            _safe_div(latency_us, max(total_assign, 1)), 4)
        out["assignments_per_s"] = round(
            _safe_div(1e6 * total_assign, latency_us), 1)
        if useful_flops:
            out["useful_tflops"] = round(
                useful_flops / (latency_us * 1e-6) / 1e12, 2)
        if padded_flops:
            out["padded_tflops"] = round(
                padded_flops / (latency_us * 1e-6) / 1e12, 2)
    return out


def swiglu_metrics(axes: dict[str, int], inputs: dict[str, torch.Tensor],
                   latency_us: Optional[float]) -> dict[str, Any]:
    """Fused SwiGLU + FP8 post-quant: elementwise bandwidth + routing stats if present."""
    base = {"kind": "swiglu-fp8"}
    x = inputs.get("gate_up")
    if x is None:
        x = inputs.get("x")
    if x is None:
        x = inputs.get("gateup_output")
    if isinstance(x, torch.Tensor):
        base["input_bytes"] = _bytes(x)
        base["approx_output_bytes"] = _bytes(x) // 2  # gate|up → half + scales
    if "masked_m" in inputs or "m_indices" in inputs:
        routed = routed_expert_metrics(axes, inputs, None)
        base.update({k: v for k, v in routed.items() if k != "kind"})
        base["kind"] = "swiglu-fp8"
    if latency_us and latency_us > 0:
        moved = base.get("input_bytes", 0) + base.get("approx_output_bytes", 0)
        base["latency_us"] = round(latency_us, 3)
        base["achieved_gbps"] = round(_safe_div(moved, latency_us * 1e3), 1)
        base["pct_of_hbm_peak"] = round(100.0 * base["achieved_gbps"] / HBM_GBPS, 2)
        M = axes.get("M", 1)
        base["us_per_token"] = round(_safe_div(latency_us, max(M, 1)), 4)
    return base


def correctness_extras(cand: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    """Mean / p99 abs error + cosine distance — advisory, never gates alone."""
    a = cand.detach().float().reshape(-1)
    b = ref.detach().float().reshape(-1)
    if a.numel() == 0 or b.numel() == 0 or a.numel() != b.numel():
        return {}
    abs_err = (a - b).abs()
    mean = float(abs_err.mean().item())
    p99 = float(torch.quantile(abs_err, 0.99).item()) if abs_err.numel() > 1 else mean
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    cos = float((a @ b / denom).item())
    cos = max(-1.0, min(1.0, cos))
    return {
        "mean_abs_err": mean,
        "p99_abs_err": p99,
        "cosine_similarity": cos,
        "cosine_distance": 1.0 - cos,
    }


def compute_workload_metrics(
    definition: dict,
    axes: dict[str, int],
    inputs: dict[str, torch.Tensor],
    latency_us: Optional[float] = None,
    family: Optional[str] = None,
) -> dict[str, Any]:
    """Dispatch family-aware metrics. Safe to call with incomplete inputs."""
    family = family or definition.get("family") or (
        (definition.get("performance_model") or {}).get("family"))
    flops = None
    fexpr = definition.get("flops_expr")
    if fexpr:
        try:
            flops = eval_expr(fexpr, axes)
        except Exception:
            flops = None

    bytes_moved = sum(_bytes(t) for t in inputs.values() if isinstance(t, torch.Tensor))
    pm = definition.get("performance_model") or {}
    kind = pm.get("kind") or family or "generic"

    if kind in ("gemm", "fp8-linear-gemm", "grouped-moe", "grouped-moe-contiguous"):
        if kind.startswith("grouped") or "masked_m" in inputs or "m_indices" in inputs:
            return routed_expert_metrics(axes, inputs, latency_us, fexpr)
        return gemm_metrics(axes, latency_us, flops, bytes_moved)
    if kind in ("sparse-mla", "sparse-mla-decode", "mla-attention"):
        return sparse_mla_metrics(axes, inputs, latency_us)
    if kind in ("swiglu-fp8", "swiglu-fp8-quant", "swiglu"):
        return swiglu_metrics(axes, inputs, latency_us)
    if kind in ("routed-expert",):
        return routed_expert_metrics(axes, inputs, latency_us, fexpr)

    # Generic fallback
    out = {"kind": "generic", "bytes_moved": bytes_moved, "flops": flops}
    if latency_us and latency_us > 0:
        out["latency_us"] = round(latency_us, 3)
        out["achieved_gbps"] = round(_safe_div(bytes_moved, latency_us * 1e3), 1)
    return out
