#!/usr/bin/env python3
"""Close the GLM-5.2 W13 16-lane fair-performance evidence matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving_native.audit_result import audit_path


RESULT_PATTERN = re.compile(
    r"fair_bm16_2sm_(leaf|region)_(eager|graph)_em(4|5|8|9)\.json$"
)
CONTROL_PATTERN = re.compile(r"a0_(leaf|region)_(eager|graph)_em4\.json$")
ESTIMATORS = (
    "pooled_speedup",
    "order_balanced_speedup",
    "ab_median_speedup",
    "ba_median_speedup",
)


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


def _distribution(values: list[float], *, scale: float = 1.0) -> dict[str, float]:
    scaled = [value * scale for value in values]
    return {
        "p10": _percentile(scaled, 0.10),
        "p50": statistics.median(scaled),
        "p90": _percentile(scaled, 0.90),
        "min": min(scaled),
        "max": max(scaled),
    }


def _samples(document: dict[str, Any], implementation: str) -> list[float]:
    return [
        float(sample["latency_ms"])
        for series in document["series"]
        for sample in series["raw_ordered_samples"]
        if sample["implementation"] == implementation
    ]


def _speedups(document: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for series in document["series"]:
        pairs: dict[int, dict[str, float]] = {}
        for sample in series["raw_ordered_samples"]:
            pairs.setdefault(int(sample["pair_index"]), {})[
                str(sample["implementation"])
            ] = float(sample["latency_ms"])
        values.extend(
            pair["reference"] / pair["candidate"]
            for _, pair in sorted(pairs.items())
        )
    return values


def _clock_range(document: dict[str, Any]) -> dict[str, list[int]]:
    samples = document["provenance"]["hardware"]["clock_samples"]
    return {
        field: [
            min(int(sample[field]) for sample in samples),
            max(int(sample[field]) for sample in samples),
        ]
        for field in ("sm_clock_mhz", "memory_clock_mhz")
    }


def _candidate_nodes(document: dict[str, Any]) -> list[list[str]] | None:
    if document["execution"]["mode"] != "cuda_graph":
        return None
    signatures = {
        tuple(
            node["kernel"]
            for node in capture["nodes"]
        )
        for series in document["series"]
        for capture in series["graph"]["captures"]
        if capture["implementation"] == "candidate"
    }
    return [list(signature) for signature in sorted(signatures)]


def _candidate_profile(document: dict[str, Any]) -> list[str] | None:
    profiles = document["execution"]["kernel_profiles"]
    if profiles is None:
        return None
    return list(profiles["candidate"]["kernel_identities"])


def _lane(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    match = RESULT_PATTERN.match(path.name)
    if match is None:
        raise ValueError(path)
    scope, execution, expected_m_text = match.groups()
    document = json.loads(path.read_text())
    audit = audit_path(path)
    if not audit["valid"] or audit["errors"]:
        raise RuntimeError(f"{path} failed independent audit: {audit}")
    if audit["performance_gate_passed"] is not True:
        raise RuntimeError(f"{path} did not pass the fair performance gate")
    series_records = []
    weakest = {
        "value": math.inf,
        "series": None,
        "estimator": None,
    }
    for series in document["series"]:
        estimates = series["performance_estimates"]
        for estimator in ESTIMATORS:
            value = float(estimates[estimator])
            if value < weakest["value"]:
                weakest = {
                    "value": value,
                    "series": int(series["series_index"]) + 1,
                    "estimator": estimator,
                }
        series_records.append(
            {
                "series": int(series["series_index"]) + 1,
                "start_order": series["start_order"],
                "pairs": int(series["repeat"]),
                "estimators": {
                    estimator: float(estimates[estimator])
                    for estimator in ESTIMATORS
                },
                "reference_latency_us": _distribution(
                    [
                        float(sample["latency_ms"])
                        for sample in series["raw_ordered_samples"]
                        if sample["implementation"] == "reference"
                    ],
                    scale=1000.0,
                ),
                "candidate_latency_us": _distribution(
                    [
                        float(sample["latency_ms"])
                        for sample in series["raw_ordered_samples"]
                        if sample["implementation"] == "candidate"
                    ],
                    scale=1000.0,
                ),
            }
        )
    runtime = document["provenance"]["w13_runtime"]
    provider = runtime["provider"]
    candidate_module = runtime["modules"]["candidate"]
    candidate_cubins = {
        relative: digest
        for relative, digest in candidate_module["jit_artifacts"].items()
        if relative.endswith(".cubin")
        and f"_em{expected_m_text}_" in relative
        and "_bm16_2sm." in relative
    }
    if len(candidate_cubins) != 1:
        raise RuntimeError(
            f"{path} expected one keyed candidate cubin, got {candidate_cubins}"
        )
    accounting = document["implementations"]["candidate"]
    record = {
        "scope": scope,
        "execution": execution,
        "expected_m": int(expected_m_text),
        "workload": document["workload"]["name"],
        "result": {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "self_audit_valid": document["self_audit"]["valid"],
            "independent_audit_valid": audit["valid"],
        },
        "physical_gpu_uuid": document["provenance"]["hardware"]["uuid"],
        "clocks": _clock_range(document),
        "series_count": len(document["series"]),
        "pairs_per_series": [series["repeat"] for series in document["series"]],
        "series": series_records,
        "weakest_estimator": weakest,
        "all_reference_latency_us": _distribution(
            _samples(document, "reference"), scale=1000.0
        ),
        "all_candidate_latency_us": _distribution(
            _samples(document, "candidate"), scale=1000.0
        ),
        "all_paired_speedups": _distribution(_speedups(document)),
        "candidate_hits": accounting["hit_count"],
        "fallbacks": accounting["fallback_count"],
        "reference_delegations": accounting["reference_delegations"],
        "candidate_profile_kernel_identities": _candidate_profile(document),
        "candidate_graph_node_signatures": _candidate_nodes(document),
        "identity": {
            "manifest_sha256": runtime["manifest_sha256"],
            "variant": runtime["variant"],
            "config": runtime["config"],
            "provider_path": provider["path"],
            "provider_sha256": provider["sha256"],
            "provider_name": provider["state"]["provider_info"]["name"],
            "candidate_dso_sha256": candidate_module["shared_object_sha256"],
            "candidate_cubin": candidate_cubins,
        },
    }
    return record, document


def _control(path: Path) -> dict[str, Any]:
    match = CONTROL_PATTERN.match(path.name)
    if match is None:
        raise ValueError(path)
    scope, execution = match.groups()
    document = json.loads(path.read_text())
    audit = audit_path(path)
    if not audit["valid"] or audit["errors"]:
        raise RuntimeError(f"{path} failed independent audit: {audit}")
    if document["aggregate"]["identity_control_forced_non_win"] is not True:
        raise RuntimeError(f"{path} was not forced to a non-win")
    estimators = [
        float(series["performance_estimates"][estimator])
        for series in document["series"]
        for estimator in ESTIMATORS
    ]
    return {
        "scope": scope,
        "execution": execution,
        "result": {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
        },
        "physical_gpu_uuid": document["provenance"]["hardware"]["uuid"],
        "series_count": len(document["series"]),
        "pairs_per_series": [series["repeat"] for series in document["series"]],
        "estimator_range": [min(estimators), max(estimators)],
        "max_absolute_deviation_from_one": max(
            abs(value - 1.0) for value in estimators
        ),
        "identity_control_forced_non_win": True,
        "performance_gate_passed": document["aggregate"][
            "performance_gate_passed"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence_dir.expanduser().resolve()
    result_paths = sorted(evidence.glob("fair_bm16_2sm_*_em*.json"))
    control_paths = sorted(evidence.glob("a0_*_em4.json"))
    if len(result_paths) != 16:
        raise RuntimeError(f"expected 16 fair results, found {len(result_paths)}")
    if len(control_paths) != 4:
        raise RuntimeError(f"expected 4 A0 controls, found {len(control_paths)}")
    lanes = []
    documents = []
    for path in result_paths:
        lane, document = _lane(path)
        lanes.append(lane)
        documents.append(document)
    expected_keys = {
        (scope, execution, expected_m)
        for scope in ("leaf", "region")
        for execution in ("eager", "graph")
        for expected_m in (4, 5, 8, 9)
    }
    actual_keys = {
        (lane["scope"], lane["execution"], lane["expected_m"])
        for lane in lanes
    }
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"fair lane matrix mismatch: missing={expected_keys - actual_keys} "
            f"extra={actual_keys - expected_keys}"
        )
    shared_identities = []
    for lane in lanes:
        identity = dict(lane["identity"])
        identity.pop("candidate_cubin")
        shared_identities.append(identity)
    identities = {
        json.dumps(identity, sort_keys=True) for identity in shared_identities
    }
    if len(identities) != 1:
        raise RuntimeError("fair results do not share one candidate identity")
    gpu_uuids = sorted({lane["physical_gpu_uuid"] for lane in lanes})
    weakest_lane = min(lanes, key=lambda lane: lane["weakest_estimator"]["value"])
    document = {
        "schema_version": 1,
        "contract": {
            "candidate": "bm16_2sm",
            "config": [16, 128, 128, 12, 2],
            "expected_m": [4, 5, 8, 9],
            "scopes": ["leaf", "region"],
            "execution_modes": ["eager", "graph"],
            "required_series": 3,
            "required_pairs_per_series": 50,
            "required_estimators": list(ESTIMATORS),
            "threshold": 1.03,
        },
        "summary": {
            "lane_count": len(lanes),
            "lanes_passed": sum(
                document["aggregate"]["performance_gate_passed"]
                for document in documents
            ),
            "all_lanes_passed": all(
                document["aggregate"]["performance_gate_passed"]
                for document in documents
            ),
            "physical_gpu_uuids": gpu_uuids,
            "weakest_lane": {
                "scope": weakest_lane["scope"],
                "execution": weakest_lane["execution"],
                "expected_m": weakest_lane["expected_m"],
                **weakest_lane["weakest_estimator"],
            },
            "total_pairs": sum(
                sum(lane["pairs_per_series"]) for lane in lanes
            ),
            "total_candidate_hits": sum(
                lane["candidate_hits"] for lane in lanes
            ),
            "total_fallbacks": sum(lane["fallbacks"] for lane in lanes),
            "total_reference_delegations": sum(
                lane["reference_delegations"] for lane in lanes
            ),
        },
        "identity": shared_identities[0],
        "controls": [_control(path) for path in control_paths],
        "lanes": sorted(
            lanes,
            key=lambda lane: (
                lane["expected_m"],
                lane["scope"],
                lane["execution"],
            ),
        ),
    }
    if document["summary"]["lanes_passed"] != 16:
        raise RuntimeError("not all fair lanes passed")
    if document["summary"]["total_fallbacks"] != 0:
        raise RuntimeError("a fair lane recorded candidate fallback")
    if document["summary"]["total_reference_delegations"] != 0:
        raise RuntimeError("a fair lane delegated to the reference")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(document["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
