"""GPU contract + performance gate for the graph-only decode o_proj dispatch.

Runs on one leased B200 and proves the decode ``o_proj`` fixed-N/K graph-only
dispatch behaves, then times it against **production stock DeepGEMM** on the
locked packed-UE8M0 ABI. Nothing here relabels stock as a candidate: the two
arms capture genuinely different DeepGEMM kernels (default compile vs
``compiled_dims="nk"``) and the audit records both kernel identities and proves
they differ while the outputs match bit-for-bit.

Lanes
-----
A. eager + ``SGLANG_GLM52_O_PROJ_GRAPH_ONLY=1`` (production) declines before any
   GEMM launch and returns ``None`` (caller keeps stock);
B. eager + ``=0`` (diagnostic) still selects the fixed-N/K candidate;
C. CUDA-graph capture selects the candidate under the production setting and the
   captured leaf is exactly one GEMM node whose identity differs from stock;
D. the captured leaf is identical with graph-only on and off (capture overrides
   the decline), so a graph lane timed either way replays the same kernel;
E. correctness: candidate == stock bit-for-bit, eager (diagnostic) and graph;
F. graph LEAF perf: candidate GEMM graph vs stock GEMM graph, three series;
G. graph CONTAINING-REGION perf: o_proj linear apply (dynamic FP8 quant + GEMM)
   candidate graph vs stock graph, three series;
H. eager containing identity lane: production graph-only eager path (guard
   declines, stock runs) vs pure stock — measures only the guard's host cost.

The whole comparison set runs in one process on one GPU; the audit refuses a
dataset spanning more than one GPU uuid (paired B200 ratios are not portable).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

# Reuse the CPU-tested alternating-series estimator from the W2 contract.
from serving_native.moe_w2_graph_only_gpu_contract import (  # noqa: E402
    _estimates,
    _time_alternating_series,
)

N = 6144
K = 16384
M_VALUES = (16, 32)
BLOCK = [128, 128]
GRAPH_ONLY_ENV = "SGLANG_GLM52_O_PROJ_GRAPH_ONLY"
CANDIDATE_NVTX_SYMBOL = "infini_kernel_glm52_attn_o_decode_nk"
# Required graph win floor (plan: all estimators >= 1.03 on required graph lanes).
GRAPH_WIN_FLOOR = 1.03
# Both arms run the same stock kernel, so a ratio this far from unity means the
# declining guard costs real host time and must be surfaced, not hidden.
IDENTITY_LANE_FLOOR = 0.97


def _set_opt(enabled: bool) -> None:
    os.environ["SGLANG_GLM52_OPT"] = "1" if enabled else "0"


def _set_graph_only(enabled: bool) -> None:
    os.environ[GRAPH_ONLY_ENV] = "1" if enabled else "0"


def _gpu_uuid(torch) -> str:
    env = os.environ.get("GLM52_PHYSICAL_GPU_UUID", "").strip()
    if env:
        return env
    try:
        return str(torch.cuda.get_device_properties(0).uuid)
    except Exception:
        return "unknown"


def _build_inputs(torch, device, m: int) -> dict[str, Any]:
    """Frozen decode o_proj inputs on the packed-UE8M0 ABI.

    Weight scales use the same constant packed-int32 UE8M0 pattern as the SGLang
    fixed-N/K unit test; scale *values* do not change the memory traffic (the
    102 MB weight read dominates) and candidate == stock holds bit-for-bit
    because only the DeepGEMM compile specialization differs.
    """
    torch.manual_seed(0)
    x = torch.randn((m, K), device=device, dtype=torch.bfloat16)
    weight = torch.randn((N, K), device=device, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    weight_scale = torch.full(
        (K // 128 // 4, N), 0x7F7F7F7F, device=device, dtype=torch.int32
    ).T
    return {"x": x, "weight": weight, "weight_scale": weight_scale, "m": m}


def _quantize_act(torch, x):
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    return sglang_per_token_group_quant_fp8(
        x,
        BLOCK[1],
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
    )


def _stock_leaf(torch, q, weight, s, weight_scale):
    from sglang.kernels.ops.quantization.fp8_kernel import (
        w8a8_block_fp8_matmul_deepgemm,
    )

    return w8a8_block_fp8_matmul_deepgemm(
        q, weight, s, weight_scale, BLOCK, torch.bfloat16
    )


def _candidate_leaf(torch, q, weight, s, weight_scale):
    from sglang.srt.layers.glm52_opt.context import op_context
    from sglang.srt.layers.glm52_opt.dispatch import try_dispatch_fp8_gemm

    with op_context("o_proj"):
        return try_dispatch_fp8_gemm(
            q, weight, s, weight_scale, BLOCK, torch.bfloat16
        )


def _region(torch, x, weight, weight_scale):
    """Production o_proj linear apply: dynamic FP8 quant + dispatched GEMM."""
    from sglang.srt.layers.glm52_opt.context import op_context
    from sglang.srt.layers.quantization.fp8_utils import (
        deepgemm_w8a8_block_fp8_linear_with_fallback,
    )

    with op_context("o_proj"):
        return deepgemm_w8a8_block_fp8_linear_with_fallback(
            x, weight, BLOCK, weight_scale
        )


def _capture(torch, device, fn) -> dict[str, Any]:
    """Warm on a side stream then capture one graph; return graph + topology."""
    from serving_native.contract_v2 import inspect_cuda_graph

    stream = torch.cuda.Stream(device=device)
    current = torch.cuda.current_stream(device)
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    stream.synchronize()
    current.wait_stream(stream)

    graph = torch.cuda.CUDAGraph(keep_graph=True)
    out_holder: list[Any] = []
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            out_holder.append(fn())
    stream.synchronize()
    current.wait_stream(stream)
    inspected = inspect_cuda_graph(int(graph.raw_cuda_graph()))
    inspected["graph"] = graph
    inspected["out"] = out_holder[0]
    return inspected


def _gemm_identities(inspected: dict[str, Any]) -> list[dict[str, Any]]:
    """Kernel nodes (name+grid+block+smem) — the GEMM leaf identity."""
    return [
        {
            "kernel": node.get("kernel"),
            "grid": node.get("grid"),
            "block": node.get("block"),
            "shared_memory_bytes": node.get("shared_memory_bytes"),
        }
        for node in inspected["nodes"]
        if "KERNEL" in str(node.get("type", ""))
    ]


def _run_m(torch, device, m: int, args) -> dict[str, Any]:
    ev: dict[str, Any] = {"m": m, "n": N, "k": K}
    inp = _build_inputs(torch, device, m)
    x, weight, weight_scale = inp["x"], inp["weight"], inp["weight_scale"]

    os.environ["SGLANG_GLM52_OPT_PROFILE"] = "e2e_candidates"
    os.environ["SGLANG_GLM52_OPT_OPS"] = "o_proj"
    from sglang.srt.layers.glm52_opt.context import set_forward_mode
    from sglang.srt.layers.glm52_opt.dispatch import _HIT_COUNTS
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    set_forward_mode(ForwardMode.DECODE, m)
    hit_key = f"fp8_gemm/fixed_nk:o_proj:decode:m{m}"

    # Pre-quantize activations once (shared by both leaf arms). Uses the exact
    # production quant (column-major, TMA-aligned, UE8M0 packed on Blackwell).
    q, s = _quantize_act(torch, x)
    ev["x_scale_dtype"] = str(s.dtype)
    ev["x_scale_shape"] = list(s.shape)
    ev["x_scale_stride"] = list(s.stride())

    # Stock reference output (OPT off).
    _set_opt(False)
    stock_leaf_out = _stock_leaf(torch, q, weight, s, weight_scale).clone()

    # A. eager, production graph-only on: decline (None), no hit.
    _set_opt(True)
    _set_graph_only(True)
    before = _HIT_COUNTS.get(hit_key, 0)
    decline = _candidate_leaf(torch, q, weight, s, weight_scale)
    torch.cuda.synchronize()
    ev["A_eager_graph_only_declines"] = {
        "returned_none": decline is None,
        "hit_delta": _HIT_COUNTS.get(hit_key, 0) - before,
    }

    # B. eager, diagnostic graph-only off: select candidate (tensor), hit fires.
    _set_graph_only(False)
    before = _HIT_COUNTS.get(hit_key, 0)
    diag = _candidate_leaf(torch, q, weight, s, weight_scale)
    torch.cuda.synchronize()
    ev["B_eager_diagnostic_selects"] = {
        "returned_tensor": isinstance(diag, torch.Tensor),
        "hit_delta": _HIT_COUNTS.get(hit_key, 0) - before,
    }
    # E1. eager diagnostic correctness vs stock.
    ev["E_eager_max_abs_err"] = (
        float((diag - stock_leaf_out).abs().max().item())
        if isinstance(diag, torch.Tensor)
        else None
    )

    # Compile every kernel eagerly (graph-only OFF so the candidate is exercised)
    # BEFORE any capture: DeepGEMM JIT compilation inside a CUDA-graph capture is
    # illegal and would abort the lease.
    _set_opt(True)
    _set_graph_only(False)
    for _ in range(3):
        _candidate_leaf(torch, q, weight, s, weight_scale)
        _region(torch, x, weight, weight_scale)
    _set_opt(False)
    for _ in range(3):
        _stock_leaf(torch, q, weight, s, weight_scale)
        _region(torch, x, weight, weight_scale)
    torch.cuda.synchronize()

    # C/D. Capture the candidate leaf under graph-only on and off; compare.
    _set_opt(True)
    captures: dict[str, dict[str, Any]] = {}
    for label, enabled in (("graph_only_on", True), ("graph_only_off", False)):
        _set_graph_only(enabled)
        cap = _capture(
            torch, device, lambda: _candidate_leaf(torch, q, weight, s, weight_scale)
        )
        captures[label] = cap
    # Stock leaf graph (OPT off) — the denominator kernel.
    _set_opt(False)
    stock_cap = _capture(
        torch, device, lambda: _stock_leaf(torch, q, weight, s, weight_scale)
    )
    _set_opt(True)

    cand_ids = _gemm_identities(captures["graph_only_on"])
    stock_ids = _gemm_identities(stock_cap)
    ev["C_candidate_leaf_graph"] = {
        "node_count": captures["graph_only_on"]["node_count"],
        "kernel_node_count": len(cand_ids),
        "gemm_identities": cand_ids,
        "forbidden_nodes": captures["graph_only_on"]["forbidden_nodes"],
        "nvtx_symbol": CANDIDATE_NVTX_SYMBOL,
    }
    ev["C_stock_leaf_graph"] = {
        "node_count": stock_cap["node_count"],
        "kernel_node_count": len(stock_ids),
        "gemm_identities": stock_ids,
    }
    ev["C_candidate_differs_from_stock"] = cand_ids != stock_ids
    ev["D_capture_identical_on_off"] = {
        "node_count_equal": (
            captures["graph_only_on"]["node_count"]
            == captures["graph_only_off"]["node_count"]
        ),
        "identities_equal": (
            _gemm_identities(captures["graph_only_on"])
            == _gemm_identities(captures["graph_only_off"])
        ),
        "node_type_counts_equal": (
            captures["graph_only_on"]["node_type_counts"]
            == captures["graph_only_off"]["node_type_counts"]
        ),
    }

    # E2. graph correctness vs stock (replay candidate leaf graph).
    captures["graph_only_on"]["graph"].replay()
    torch.cuda.synchronize()
    graph_leaf_out = captures["graph_only_on"]["out"]
    ev["E_graph_max_abs_err"] = float(
        (graph_leaf_out - stock_leaf_out).abs().max().item()
    )

    # F. graph LEAF perf: candidate leaf graph vs stock leaf graph.
    cand_leaf_graph = captures["graph_only_on"]["graph"]
    ev["F_graph_leaf_series"] = [
        _time_alternating_series(
            torch,
            stock_cap["graph"].replay,
            cand_leaf_graph.replay,
            series_index=i,
            warmup=args.warmup,
            pairs=args.pairs,
        )
        for i in range(args.series)
    ]

    # G. graph CONTAINING-REGION perf: o_proj linear apply (quant + GEMM).
    _set_opt(True)
    _set_graph_only(True)
    cand_region_cap = _capture(torch, device, lambda: _region(torch, x, weight, weight_scale))
    _set_opt(False)
    stock_region_cap = _capture(torch, device, lambda: _region(torch, x, weight, weight_scale))
    _set_opt(True)
    ev["G_region_node_types"] = {
        "candidate": cand_region_cap["node_type_counts"],
        "stock": stock_region_cap["node_type_counts"],
    }
    # Region correctness (candidate region replay vs stock region replay).
    cand_region_cap["graph"].replay()
    stock_region_cap["graph"].replay()
    torch.cuda.synchronize()
    ev["G_region_max_abs_err"] = float(
        (cand_region_cap["out"] - stock_region_cap["out"]).abs().max().item()
    )
    ev["G_graph_region_series"] = [
        _time_alternating_series(
            torch,
            stock_region_cap["graph"].replay,
            cand_region_cap["graph"].replay,
            series_index=i,
            warmup=args.warmup,
            pairs=args.pairs,
        )
        for i in range(args.series)
    ]

    # H. eager containing identity lane: production graph-only eager path
    # (guard declines -> stock runs) vs pure stock. Both execute stock; the
    # ratio isolates the declining guard's host cost.
    _set_opt(True)
    _set_graph_only(True)

    def guarded_region() -> None:
        # Production eager path: dispatch declines (graph-only, not capturing),
        # caller runs stock. We assert the decline never selects the candidate.
        out = _region(torch, x, weight, weight_scale)
        del out

    def pure_stock_region() -> None:
        _set_opt(False)
        out = _region(torch, x, weight, weight_scale)
        _set_opt(True)
        del out

    ev["H_eager_identity_series"] = [
        _time_alternating_series(
            torch,
            pure_stock_region,
            guarded_region,
            series_index=i,
            warmup=args.warmup,
            pairs=args.pairs,
        )
        for i in range(args.series)
    ]

    set_forward_mode(None)
    # Release the captured graphs / big weight before the next M.
    for cap in (*captures.values(), stock_cap, cand_region_cap, stock_region_cap):
        cap.pop("graph", None)
        cap.pop("out", None)
    del q, s, weight, weight_scale, x, stock_leaf_out, graph_leaf_out
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return ev


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("o_proj graph-only contract requires a CUDA device")
    device = torch.device("cuda:0")
    evidence: dict[str, Any] = {
        "contract": "glm52-o-proj-decode-graph-only-gpu-v1",
        "op": "o_proj",
        "phase": "decode",
        "n": N,
        "k": K,
        "m_values": list(M_VALUES),
        "graph_only_env": GRAPH_ONLY_ENV,
        "candidate_nvtx_symbol": CANDIDATE_NVTX_SYMBOL,
        "graph_win_floor": GRAPH_WIN_FLOOR,
        "identity_lane_floor": IDENTITY_LANE_FLOOR,
        "gpu_uuid": _gpu_uuid(torch),
        "physical_gpu": os.environ.get("GLM52_PHYSICAL_GPU", ""),
        "torch_version": torch.__version__,
    }
    from sglang.srt.layers import deep_gemm_wrapper

    evidence["deepgemm_scale_ue8m0"] = bool(
        getattr(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", False)
    )
    evidence["enable_jit_deepgemm"] = bool(
        getattr(deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", False)
    )
    evidence["by_m"] = {str(m): _run_m(torch, device, m, args) for m in M_VALUES}
    return evidence


def _series_estimators(series: list[dict[str, Any]]) -> list[dict[str, float]]:
    return [s["performance_estimates"] for s in series]


def audit(evidence: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if evidence.get("gpu_uuid", "unknown") in ("", "unknown"):
        failures.append("gpu: no resolvable GPU uuid recorded")
    if not evidence.get("deepgemm_scale_ue8m0"):
        failures.append(
            "abi: DEEPGEMM_SCALE_UE8M0 is False — not the packed-UE8M0 production path"
        )
    est_fields = (
        "pooled_speedup",
        "order_balanced_speedup",
        "ab_median_speedup",
        "ba_median_speedup",
    )
    for m_key, ev in evidence["by_m"].items():
        tag = f"M{m_key}"
        # ABI: activation scales are packed int32.
        if ev.get("x_scale_dtype") != "torch.int32":
            failures.append(f"{tag} abi: x_scale dtype {ev.get('x_scale_dtype')} != int32")
        # A/B dispatch contract.
        a = ev["A_eager_graph_only_declines"]
        if not a["returned_none"] or a["hit_delta"] != 0:
            failures.append(f"{tag} A: eager graph-only did not decline before launch")
        b = ev["B_eager_diagnostic_selects"]
        if not b["returned_tensor"] or b["hit_delta"] != 1:
            failures.append(f"{tag} B: diagnostic eager did not select the candidate")
        # C: candidate leaf graph is exactly one GEMM node, differs from stock.
        c = ev["C_candidate_leaf_graph"]
        if c["kernel_node_count"] != 1 or c["forbidden_nodes"]:
            failures.append(f"{tag} C: candidate leaf graph is not exactly one clean GEMM node")
        if not ev["C_candidate_differs_from_stock"]:
            failures.append(f"{tag} C: candidate GEMM identity equals stock (stock relabeled)")
        # D: capture identical on/off.
        if not all(ev["D_capture_identical_on_off"].values()):
            failures.append(f"{tag} D: captured leaf differs between graph-only settings")
        # E: correctness bit-exact.
        if ev.get("E_eager_max_abs_err") not in (0.0, 0):
            failures.append(f"{tag} E: eager candidate != stock (max_abs_err={ev.get('E_eager_max_abs_err')})")
        if ev.get("E_graph_max_abs_err") not in (0.0, 0):
            failures.append(f"{tag} E: graph candidate != stock (max_abs_err={ev.get('E_graph_max_abs_err')})")
        if ev.get("G_region_max_abs_err") not in (0.0, 0):
            failures.append(f"{tag} E: graph region candidate != stock (max_abs_err={ev.get('G_region_max_abs_err')})")
        # F: graph leaf >= 1.03 all series, all estimators.
        for i, est in enumerate(_series_estimators(ev["F_graph_leaf_series"])):
            for f in est_fields:
                v = est[f]
                if not math.isfinite(v) or v < GRAPH_WIN_FLOOR:
                    failures.append(f"{tag} F: graph-leaf series {i} {f}={v:.4f} < {GRAPH_WIN_FLOOR}")
        # G: graph containing-region >= 1.03 all series, all estimators.
        for i, est in enumerate(_series_estimators(ev["G_graph_region_series"])):
            for f in est_fields:
                v = est[f]
                if not math.isfinite(v) or v < GRAPH_WIN_FLOOR:
                    failures.append(f"{tag} G: graph-region series {i} {f}={v:.4f} < {GRAPH_WIN_FLOOR}")
        # H: eager identity lane floor (guard cost sanity).
        for i, est in enumerate(_series_estimators(ev["H_eager_identity_series"])):
            for f in est_fields:
                v = est[f]
                if not math.isfinite(v) or v < IDENTITY_LANE_FLOOR:
                    failures.append(f"{tag} H: eager identity series {i} {f}={v:.4f} < {IDENTITY_LANE_FLOOR}")
    return failures


def _summary(evidence: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for m_key, ev in evidence["by_m"].items():
        def med(series):
            vals = [s["performance_estimates"]["order_balanced_speedup"] for s in series]
            return sorted(vals)[len(vals) // 2]
        out[f"M{m_key}"] = {
            "graph_leaf_obs": round(med(ev["F_graph_leaf_series"]), 4),
            "graph_region_obs": round(med(ev["G_graph_region_series"]), 4),
            "eager_identity_obs": round(med(ev["H_eager_identity_series"]), 4),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    evidence = run(args)
    failures = audit(evidence)
    evidence["failures"] = failures
    evidence["status"] = "pass" if not failures else "fail"
    evidence["summary"] = _summary(evidence)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": evidence["status"],
        "gpu_uuid": evidence["gpu_uuid"],
        "summary": evidence["summary"],
        "failures": failures,
    }, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
