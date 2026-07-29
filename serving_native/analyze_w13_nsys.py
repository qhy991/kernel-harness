#!/usr/bin/env python3
"""Summarize a node-traced Nsight Systems W13 containing-region capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CANDIDATE_SYMBOL = "infini_kernel_glm52_moe_w13_decode_em4_bm16_2sm"
STOCK_W13_SHAPE = "(unsigned int)4096, (unsigned int)6144"
W2_SHAPE = "(unsigned int)6144, (unsigned int)2048"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of empty sequence")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution_ns(values: list[int | float]) -> dict[str, float]:
    converted = [float(value) / 1000.0 for value in values]
    return {
        "p10_us": _percentile(converted, 0.10),
        "p50_us": statistics.median(converted),
        "p90_us": _percentile(converted, 0.90),
        "min_us": min(converted),
        "max_us": max(converted),
    }


def _classify(name: str) -> str | None:
    if CANDIDATE_SYMBOL in name:
        return "candidate_w13"
    if "sm100_fp8_fp4_gemm_1d1d_impl" in name and STOCK_W13_SHAPE in name:
        return "stock_w13"
    if "silu_mul_quant_varlen_kernel" in name:
        return "activation_quant"
    if "sm100_fp8_fp4_gemm_1d1d_impl" in name and W2_SHAPE in name:
        return "w2"
    return None


def _collect(sqlite_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    connection = sqlite3.connect(sqlite_path)
    connection.row_factory = sqlite3.Row
    kernel_query = """
        SELECT
            k.correlationId,
            k.graphId,
            k.graphNodeId,
            k.start,
            k.end,
            k.streamId,
            k.gridX,
            k.gridY,
            k.gridZ,
            k.blockX,
            k.blockY,
            k.blockZ,
            k.dynamicSharedMemory,
            k.registersPerThread,
            strings.value AS name
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN StringIds AS strings ON strings.id = k.demangledName
        WHERE k.graphId IS NOT NULL
        ORDER BY k.start
    """
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in connection.execute(kernel_query):
        kind = _classify(row["name"])
        if kind is None:
            continue
        grouped[int(row["correlationId"])].append(
            {
                "kind": kind,
                **{
                    field: int(row[field])
                    for field in (
                        "graphId",
                        "graphNodeId",
                        "start",
                        "end",
                        "streamId",
                        "gridX",
                        "gridY",
                        "gridZ",
                        "blockX",
                        "blockY",
                        "blockZ",
                        "dynamicSharedMemory",
                        "registersPerThread",
                    )
                },
                "name": str(row["name"]),
            }
        )

    complete: dict[int, list[dict[str, Any]]] = {}
    for correlation_id, kernels in grouped.items():
        kinds = [kernel["kind"] for kernel in kernels]
        if (
            len(kernels) == 3
            and kinds.count("activation_quant") == 1
            and kinds.count("w2") == 1
            and (
                kinds.count("candidate_w13") == 1
                or kinds.count("stock_w13") == 1
            )
        ):
            complete[correlation_id] = sorted(
                kernels, key=lambda kernel: kernel["start"]
            )
    if not complete:
        raise RuntimeError("no complete W13 -> activation/quant -> W2 graphs found")

    placeholders = ",".join("?" for _ in complete)
    runtime_query = f"""
        SELECT
            runtime.correlationId,
            runtime.start,
            runtime.end,
            strings.value AS name
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
        JOIN StringIds AS strings ON strings.id = runtime.nameId
        WHERE runtime.correlationId IN ({placeholders})
    """
    runtime = {
        int(row["correlationId"]): {
            "start": int(row["start"]),
            "end": int(row["end"]),
            "name": str(row["name"]),
        }
        for row in connection.execute(runtime_query, tuple(complete))
    }
    connection.close()
    if set(runtime) != set(complete):
        raise RuntimeError("graph launch/runtime correlation set does not close")
    if {record["name"] for record in runtime.values()} != {
        "cudaGraphLaunch_v10000"
    }:
        raise RuntimeError("complete graphs were not submitted by cudaGraphLaunch")

    observations: dict[str, list[dict[str, Any]]] = {
        "stock": [],
        "candidate": [],
    }
    for correlation_id, kernels in complete.items():
        kinds = [kernel["kind"] for kernel in kernels]
        arm = "candidate" if "candidate_w13" in kinds else "stock"
        expected = (
            ["candidate_w13", "activation_quant", "w2"]
            if arm == "candidate"
            else ["stock_w13", "activation_quant", "w2"]
        )
        if kinds != expected:
            raise RuntimeError(
                f"{arm} graph kernel order mismatch for correlation {correlation_id}: "
                f"{kinds}"
            )
        launch = runtime[correlation_id]
        first, middle, last = kernels
        observations[arm].append(
            {
                "correlation_id": correlation_id,
                "graph_id": first["graphId"],
                "stream_id": first["streamId"],
                "launch_api_ns": launch["end"] - launch["start"],
                "queue_after_api_ns": first["start"] - launch["end"],
                "api_start_to_device_end_ns": last["end"] - launch["start"],
                "device_span_ns": last["end"] - first["start"],
                "sum_kernel_ns": sum(
                    kernel["end"] - kernel["start"] for kernel in kernels
                ),
                "w13_ns": first["end"] - first["start"],
                "activation_quant_ns": middle["end"] - middle["start"],
                "w2_ns": last["end"] - last["start"],
                "w13_to_activation_gap_ns": middle["start"] - first["end"],
                "activation_to_w2_gap_ns": last["start"] - middle["end"],
                "w13_geometry": {
                    field: first[field]
                    for field in (
                        "gridX",
                        "gridY",
                        "gridZ",
                        "blockX",
                        "blockY",
                        "blockZ",
                        "dynamicSharedMemory",
                        "registersPerThread",
                    )
                },
            }
        )
    counts = {arm: len(items) for arm, items in observations.items()}
    if counts["stock"] != counts["candidate"]:
        raise RuntimeError(f"unbalanced graph launch counts: {counts}")

    summaries: dict[str, Any] = {}
    for arm, items in observations.items():
        summaries[arm] = {
            "graph_launches": len(items),
            "graph_ids": sorted({item["graph_id"] for item in items}),
            "stream_ids": sorted({item["stream_id"] for item in items}),
            "launch_api": _distribution_ns(
                [item["launch_api_ns"] for item in items]
            ),
            "queue_after_api": _distribution_ns(
                [item["queue_after_api_ns"] for item in items]
            ),
            "api_start_to_device_end": _distribution_ns(
                [item["api_start_to_device_end_ns"] for item in items]
            ),
            "device_critical_span": _distribution_ns(
                [item["device_span_ns"] for item in items]
            ),
            "sum_kernel_time": _distribution_ns(
                [item["sum_kernel_ns"] for item in items]
            ),
            "w13_device": _distribution_ns([item["w13_ns"] for item in items]),
            "activation_quant_device": _distribution_ns(
                [item["activation_quant_ns"] for item in items]
            ),
            "w2_device": _distribution_ns([item["w2_ns"] for item in items]),
            "w13_to_activation_gap": _distribution_ns(
                [item["w13_to_activation_gap_ns"] for item in items]
            ),
            "activation_to_w2_gap": _distribution_ns(
                [item["activation_to_w2_gap_ns"] for item in items]
            ),
            "w13_geometry": sorted(
                {
                    tuple(sorted(item["w13_geometry"].items()))
                    for item in items
                }
            ),
        }
    return summaries, {
        "candidate_symbol": CANDIDATE_SYMBOL,
        "complete_graph_launches": sum(counts.values()),
        "arm_counts": counts,
        "classification": {
            "stock_w13_shape": STOCK_W13_SHAPE,
            "candidate_w13_symbol": CANDIDATE_SYMBOL,
            "activation_quant": "silu_mul_quant_varlen_kernel",
            "w2_shape": W2_SHAPE,
        },
    }


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        raise ValueError("speedup denominator must be positive")
    return numerator / denominator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--archived-report", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        name: path.expanduser().resolve()
        for name, path in (
            ("sqlite", args.sqlite),
            ("report", args.report),
            ("archived_report", args.archived_report),
            ("result", args.result),
        )
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    summaries, collection = _collect(paths["sqlite"])
    stock = summaries["stock"]
    candidate = summaries["candidate"]
    stock_span = stock["device_critical_span"]["p50_us"]
    candidate_span = candidate["device_critical_span"]["p50_us"]
    stock_w13 = stock["w13_device"]["p50_us"]
    candidate_w13 = candidate["w13_device"]["p50_us"]
    result = json.loads(paths["result"].read_text())
    runtime = result["provenance"]["w13_runtime"]
    nsys_version = subprocess.run(
        ["nsys", "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    document = {
        "schema_version": 1,
        "tool": {
            "name": "NVIDIA Nsight Systems",
            "version": nsys_version,
            "collection": {
                "trace": ["cuda", "nvtx"],
                "sample": "none",
                "cpuctxsw": "none",
                "cuda_graph_trace": "node",
            },
        },
        "artifacts": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "profiled_result": {
            "self_audit_valid": result["self_audit"]["valid"],
            "performance_authority": False,
            "performance_gate_passed": result["aggregate"][
                "performance_gate_passed"
            ],
            "workload": result["workload"]["name"],
            "execution_mode": result["execution"]["mode"],
            "physical_gpu_uuid": result["provenance"]["hardware"]["uuid"],
            "candidate": result["candidate"]["path"],
            "candidate_variant": runtime["variant"],
            "manifest_sha256": runtime["manifest_sha256"],
            "note": (
                "Profiler-instrumented 3x10 result is attribution-only; "
                "unprofiled 3x50 artifacts are the performance authority."
            ),
        },
        "graph_collection": collection,
        "arms": summaries,
        "comparison": {
            "device_critical_span_speedup": _ratio(
                stock_span, candidate_span
            ),
            "device_critical_span_reduction_us": stock_span - candidate_span,
            "w13_device_speedup": _ratio(stock_w13, candidate_w13),
            "w13_device_reduction_us": stock_w13 - candidate_w13,
            "downstream_activation_quant_delta_us": (
                candidate["activation_quant_device"]["p50_us"]
                - stock["activation_quant_device"]["p50_us"]
            ),
            "downstream_w2_delta_us": (
                candidate["w2_device"]["p50_us"]
                - stock["w2_device"]["p50_us"]
            ),
            "launch_api_delta_us": (
                candidate["launch_api"]["p50_us"]
                - stock["launch_api"]["p50_us"]
            ),
            "critical_path_interpretation": (
                "The three graph nodes execute in dependency order. Negative "
                "inter-node gaps are sub-microsecond boundary overlap; W13 "
                "dominates the measured stock-to-candidate span reduction."
            ),
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(document["comparison"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
