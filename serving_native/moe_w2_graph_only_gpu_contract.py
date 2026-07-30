"""GPU contract for the graph-only MoE W2 hotspot dispatch policy.

Runs on one leased B200 and proves the four properties the W2 graph-only
integration claims, without asserting any performance result:

A. eager + ``SGLANG_GLM52_W2_GRAPH_ONLY=1`` declines before any provider
   launch, leaving the caller-owned output untouched;
B. eager + ``SGLANG_GLM52_W2_GRAPH_ONLY=0`` still selects the candidate, so
   the diagnostic eager leaf remains reachable;
C. CUDA graph capture selects the candidate under the production setting and
   yields exactly one W2 kernel node carrying the candidate symbol;
D. the captured graph is identical with graph-only on and off, so a graph lane
   timed under either setting measures the same replayed kernel.

It then times the eager containing region as a stock-fallback identity lane:
both arms execute the same stock W2 kernel, and the ratio measures only the
declining dispatch guard's host cost.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
CANDIDATE_PATH = HERE / "candidates" / "moe_w2_hotspot_bm16_auto.py"
CANDIDATE_SYMBOL = "infini_kernel_glm52_moe_w2_decode_bm16_auto"
GRAPH_ONLY_ENV = "SGLANG_GLM52_W2_GRAPH_ONLY"
# Both identity-lane arms run the same stock kernel, so a ratio this far from
# unity means the declining guard costs real time and must be reported.
IDENTITY_LANE_FLOOR = 0.97


def _load_candidate_module():
    spec = importlib.util.spec_from_file_location(
        "moe_w2_graph_only_contract_candidate", CANDIDATE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_runtime(workload) -> Any:
    return SimpleNamespace(workload=workload, local_rank=0)


class _ProviderProbe:
    """Snapshot/diff the provider's own launch counters."""

    def __init__(self, provider_module) -> None:
        self._provider = provider_module
        self._mark = self.counts()

    def counts(self) -> dict[str, int]:
        return dict(self._provider.call_counts())

    def reset(self) -> None:
        self._mark = self.counts()

    def attempted_since_mark(self) -> int:
        return self.counts()["attempted"] - self._mark["attempted"]


def _set_graph_only(enabled: bool) -> None:
    os.environ[GRAPH_ONLY_ENV] = "1" if enabled else "0"


def _poisoned_everywhere(torch, out) -> bool:
    return bool(torch.isnan(out).all().item())


def _dispatch_once(dispatch, inputs, *, out) -> bool:
    from sglang.srt.layers.glm52_opt.context import op_context

    with op_context("moe_down_proj"):
        return bool(
            dispatch.try_dispatch_moe_masked(
                (inputs["down_input"], inputs["down_input_scale"]),
                (inputs["w2_weight_fp8"], inputs["w2_weight_scale"]),
                out,
                inputs["masked_m"],
                inputs["expected_m"],
            )
        )


def _capture_dispatch(
    torch, dispatch, inputs, *, out, device, probe
) -> dict[str, Any]:
    """Capture one W2 dispatch and return the graph's inspected topology."""
    from serving_native.contract_v2 import inspect_cuda_graph

    stream = torch.cuda.Stream(device=device)
    current = torch.cuda.current_stream(device)
    stream.wait_stream(current)
    # Side-stream warmup so nothing lazy is left to allocate inside capture.
    with torch.cuda.stream(stream):
        for _ in range(3):
            _dispatch_once(dispatch, inputs, out=out)
    stream.synchronize()
    current.wait_stream(stream)
    # Count only what capture itself records.
    probe.reset()

    graph = torch.cuda.CUDAGraph(keep_graph=True)
    selected: list[bool] = []
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            selected.append(_dispatch_once(dispatch, inputs, out=out))
    stream.synchronize()
    current.wait_stream(stream)
    inspected = inspect_cuda_graph(int(graph.raw_cuda_graph()))
    inspected["selected"] = bool(selected and selected[0])
    inspected["graph"] = graph
    return inspected


def _region_inputs(runtime) -> dict[str, Any]:
    """Build the containing-region inputs and its pre-W2 activation."""
    from sglang.srt.layers.deep_gemm_wrapper.entrypoint import (
        grouped_gemm_nt_f8f8bf16_masked,
    )
    from sglang.srt.layers.glm52_opt.context import op_context
    from sglang.srt.layers.moe.moe_runner.deep_gemm import (
        _varlen_deep_gemm_silu_mul_quant,
    )

    inputs = runtime.build_inputs()
    experts, slab, _hidden = inputs["activation_fp8"].shape
    gate_up = inputs["w13_weight_fp8"].shape[1]
    gateup_output = runtime.torch.empty(
        (experts, slab, gate_up), device=runtime.device, dtype=runtime.torch.bfloat16
    )
    with op_context("moe_gate_proj"):
        grouped_gemm_nt_f8f8bf16_masked(
            (inputs["activation_fp8"], inputs["activation_scale"]),
            (inputs["w13_weight_fp8"], inputs["w13_weight_scale"]),
            gateup_output,
            inputs["masked_m"],
            inputs["expected_m"],
        )
    down_input, down_input_scale = _varlen_deep_gemm_silu_mul_quant(
        gateup_output,
        inputs["masked_m"],
        group_size=inputs["group_size"],
        topk=inputs["topk"],
    )
    inputs["down_input"] = down_input
    inputs["down_input_scale"] = down_input_scale
    return inputs


def _time_alternating_series(
    torch,
    reference_fn: Callable[[], None],
    candidate_fn: Callable[[], None],
    *,
    series_index: int,
    warmup: int,
    pairs: int,
) -> dict[str, Any]:
    """One independent alternating AB/BA series with retained ordered samples."""
    start_order = "AB" if series_index % 2 == 0 else "BA"
    functions = {"reference": reference_fn, "candidate": candidate_fn}
    for index in range(warmup):
        order = start_order if index % 2 == 0 else _flip(start_order)
        for name in _order_names(order):
            functions[name]()
    torch.cuda.synchronize()

    raw: list[dict[str, Any]] = []
    for index in range(pairs):
        order = start_order if index % 2 == 0 else _flip(start_order)
        for name in _order_names(order):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            functions[name]()
            end.record()
            end.synchronize()
            raw.append(
                {
                    "order": order,
                    "implementation": name,
                    "latency_ms": float(start.elapsed_time(end)),
                }
            )
    return {
        "series_index": series_index,
        "start_order": start_order,
        "warmup_pairs": warmup,
        "pairs": pairs,
        "raw_ordered_samples": raw,
        "performance_estimates": _estimates(raw),
    }


def _flip(order: str) -> str:
    return "BA" if order == "AB" else "AB"


def _order_names(order: str) -> tuple[str, str]:
    return ("reference", "candidate") if order == "AB" else ("candidate", "reference")


def _estimates(raw: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute the four required estimators from raw ordered pairs."""
    reference_values: list[float] = []
    candidate_values: list[float] = []
    by_order: dict[str, list[float]] = {"AB": [], "BA": []}
    for offset in range(0, len(raw), 2):
        pair = raw[offset : offset + 2]
        order = pair[0]["order"]
        if pair[1]["order"] != order:
            raise RuntimeError("identity-lane pair order is malformed")
        values = {item["implementation"]: item["latency_ms"] for item in pair}
        if set(values) != {"reference", "candidate"}:
            raise RuntimeError("identity-lane pair is incomplete")
        reference_values.append(values["reference"])
        candidate_values.append(values["candidate"])
        by_order[order].append(values["reference"] / values["candidate"])
    ab = statistics.median(by_order["AB"])
    ba = statistics.median(by_order["BA"])
    pooled = statistics.median(reference_values) / statistics.median(candidate_values)
    estimates = {
        "pair_count": len(raw) // 2,
        "pooled_speedup": pooled,
        "order_balanced_speedup": math.sqrt(ab * ba),
        "ab_median_speedup": ab,
        "ba_median_speedup": ba,
        "reference_p50_ms": statistics.median(reference_values),
        "candidate_p50_ms": statistics.median(candidate_values),
    }
    if not all(
        math.isfinite(value) and value > 0.0
        for key, value in estimates.items()
        if key != "pair_count"
    ):
        raise RuntimeError("identity-lane estimator is non-finite")
    return estimates


def run(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(HERE.parent))
    candidate_module = _load_candidate_module()
    candidate_module.prepare_process()

    from serving_native.runner import Runtime
    from serving_native.workloads import get_workload

    workload = get_workload(args.region_workload)
    runtime = Runtime(workload)
    candidate_module.prepare_runtime(_fake_runtime(workload))

    torch = runtime.torch
    device = runtime.device
    dispatch = candidate_module._STATE["dispatch"]
    provider_module = candidate_module._STATE["provider_module"]
    probe = _ProviderProbe(provider_module)
    evidence: dict[str, Any] = {
        "contract": "glm52-moe-w2-graph-only-gpu-v1",
        "region_workload": workload.name,
        "candidate_symbol": CANDIDATE_SYMBOL,
        "graph_only_env": GRAPH_ONLY_ENV,
    }

    inputs = _region_inputs(runtime)
    experts, slab, hidden = inputs["out"].shape
    scratch = torch.empty(
        (experts, slab, hidden), device=device, dtype=torch.bfloat16
    )

    # A. Production setting, eager: decline before any provider launch.
    _set_graph_only(True)
    scratch.fill_(float("nan"))
    probe.reset()
    selected_eager_graph_only = _dispatch_once(dispatch, inputs, out=scratch)
    torch.cuda.synchronize()
    evidence["A_eager_graph_only"] = {
        "selected": selected_eager_graph_only,
        "provider_attempts": probe.attempted_since_mark(),
        "output_untouched": _poisoned_everywhere(torch, scratch),
    }

    # B. Diagnostic override, eager: the candidate is still reachable.
    _set_graph_only(False)
    scratch.fill_(float("nan"))
    probe.reset()
    selected_eager_forced = _dispatch_once(dispatch, inputs, out=scratch)
    torch.cuda.synchronize()
    evidence["B_eager_graph_only_disabled"] = {
        "selected": selected_eager_forced,
        "provider_attempts": probe.attempted_since_mark(),
        "output_untouched": _poisoned_everywhere(torch, scratch),
    }

    # C/D. Capture under both settings and compare the recorded topology.
    captures: dict[str, dict[str, Any]] = {}
    for label, enabled in (("graph_only_on", True), ("graph_only_off", False)):
        _set_graph_only(enabled)
        scratch.fill_(float("nan"))
        inspected = _capture_dispatch(
            torch, dispatch, inputs, out=scratch, device=device, probe=probe
        )
        graph = inspected.pop("graph")
        inspected["provider_attempts"] = probe.attempted_since_mark()
        captures[label] = inspected
        del graph
    evidence["C_capture"] = captures
    evidence["D_capture_identical"] = {
        "node_count_equal": (
            captures["graph_only_on"]["node_count"]
            == captures["graph_only_off"]["node_count"]
        ),
        "kernel_identities_equal": (
            captures["graph_only_on"]["kernel_identities"]
            == captures["graph_only_off"]["kernel_identities"]
        ),
        "node_type_counts_equal": (
            captures["graph_only_on"]["node_type_counts"]
            == captures["graph_only_off"]["node_type_counts"]
        ),
    }

    # E. Eager containing-region stock-fallback identity lane.
    _set_graph_only(True)
    stock_w2 = candidate_module_stock_w2(runtime)

    def guarded_w2(lhs, rhs, out, masked_m, expected_m) -> None:
        """Production graph-only eager path: guard declines, stock runs."""
        if dispatch.try_dispatch_moe_masked(lhs, rhs, out, masked_m, expected_m):
            raise RuntimeError(
                "graph-only dispatch selected the candidate in an eager lane"
            )
        stock_w2(lhs, rhs, out, masked_m, expected_m)

    probe.reset()
    series: list[dict[str, Any]] = []
    for series_index in range(args.series):
        series.append(
            _time_alternating_series(
                torch,
                lambda: runtime.run_moe_compute_region(inputs, w2_gemm=stock_w2),
                lambda: runtime.run_moe_compute_region(inputs, w2_gemm=guarded_w2),
                series_index=series_index,
                warmup=args.warmup,
                pairs=args.pairs,
            )
        )
    evidence["E_eager_containing_identity_lane"] = {
        "policy": "both_arms_execute_stock_w2_kernel",
        "provider_attempts_during_lane": probe.attempted_since_mark(),
        "declared_floor": IDENTITY_LANE_FLOOR,
        "series": series,
    }
    evidence["provider_call_counts_final"] = probe.counts()
    # The harness candidate wrapper is deliberately unused here: this contract
    # drives SGLang's dispatch entry point directly.
    evidence["candidate_wrapper_invocations"] = candidate_module._STATE[
        "selected_invocations"
    ]
    return evidence


def candidate_module_stock_w2(runtime) -> Callable[..., None]:
    """The same-DSO stock denominator used by the harness region."""
    return type(runtime)._stock_w2_gemm


def audit(evidence: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    a = evidence["A_eager_graph_only"]
    if a["selected"] or a["provider_attempts"] != 0 or not a["output_untouched"]:
        failures.append("A: eager graph-only did not decline before provider launch")
    b = evidence["B_eager_graph_only_disabled"]
    if not b["selected"] or b["provider_attempts"] != 1 or b["output_untouched"]:
        failures.append("B: diagnostic eager override did not select the candidate")
    for label, capture in evidence["C_capture"].items():
        kernels = [
            name for name in capture["kernel_identities"] if CANDIDATE_SYMBOL in name
        ]
        if (
            not capture["selected"]
            or capture["provider_attempts"] != 1
            or capture["node_count"] != 1
            or len(kernels) != 1
            or capture["forbidden_nodes"]
        ):
            failures.append(f"C: {label} capture is not exactly one candidate node")
    if not all(evidence["D_capture_identical"].values()):
        failures.append("D: captured graphs differ between graph-only settings")
    lane = evidence["E_eager_containing_identity_lane"]
    if lane["provider_attempts_during_lane"] != 0:
        failures.append("E: the eager identity lane launched the provider")
    for item in lane["series"]:
        for field in (
            "pooled_speedup",
            "order_balanced_speedup",
            "ab_median_speedup",
            "ba_median_speedup",
        ):
            value = item["performance_estimates"][field]
            if not math.isfinite(value) or value < IDENTITY_LANE_FLOOR:
                failures.append(
                    f"E: series {item['series_index']} {field}={value:.6f} "
                    f"below the declared {IDENTITY_LANE_FLOOR} identity floor"
                )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--region-workload", default="moe_w2_hotspot_region_decode_em4"
    )
    parser.add_argument("--series", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    evidence = run(args)
    failures = audit(evidence)
    evidence["failures"] = failures
    evidence["status"] = "pass" if not failures else "fail"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": evidence["status"], "failures": failures}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
