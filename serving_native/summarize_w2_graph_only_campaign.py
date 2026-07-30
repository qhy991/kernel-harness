"""Reduce the W2 graph-only campaign results to one signed evidence document.

Reads every runner result plus the graph-only GPU contract and the NCU roofline
numbers, recomputes the gate verdict from raw per-series estimators, and writes
a compact JSON summary that the final report cites.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ESTIMATORS = (
    "pooled_speedup",
    "order_balanced_speedup",
    "ab_median_speedup",
    "ba_median_speedup",
)
THRESHOLD = 1.03
# NCU, expected-M 4, one profiled launch per arm (see ncu/w2_em4.ncu-rep).
NCU = {
    "stock": {
        "duration_us": 74.848,
        "dram_read_bytes": 414284288,
        "dram_write_bytes": 40068864,
        "registers": 36,
        "issue_slots_busy_pct": 6.23,
    },
    "candidate": {
        "duration_us": 68.096,
        "dram_read_bytes": 406906624,
        "dram_write_bytes": 4599552,
        "registers": 43,
        "issue_slots_busy_pct": 5.49,
    },
    # E32 x N6144 x K2048 FP8 weights are read exactly once and cannot shrink.
    "irreducible_weight_bytes": 32 * 6144 * 2048,
    # expected-M 4: 128 valid rows x N6144 x BF16.
    "write_floor_bytes_em4": 128 * 6144 * 2,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _lane(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    series = []
    for item in document.get("series", []):
        estimates = item["performance_estimates"]
        series.append(
            {
                "series_index": item["series_index"],
                "start_order": item["start_order"],
                "pair_count": estimates["pair_count"],
                **{name: estimates[name] for name in ESTIMATORS},
                "min_estimator": min(estimates[name] for name in ESTIMATORS),
                "reference_median_ms": item["reference"]["median_ms"],
                "candidate_median_ms": item["candidate"]["median_ms"],
                "passes": all(estimates[name] >= THRESHOLD for name in ESTIMATORS),
            }
        )
    return {
        "result_path": str(path),
        "result_sha256": _sha256(path),
        "workload": document["workload"]["name"],
        "family": document["workload"]["family"],
        "expected_m": document["workload"]["params"]["expected_m"],
        "execution_mode": document["self_audit"]["execution_mode"],
        "self_audit_valid": document["self_audit"]["valid"],
        "series": series,
        "series_passing": sum(1 for item in series if item["passes"]),
        "series_total": len(series),
        "gate_passes": bool(series) and all(item["passes"] for item in series),
        "worst_estimator": min(item["min_estimator"] for item in series),
    }


def _roofline() -> dict[str, Any]:
    candidate = NCU["candidate"]
    moved = candidate["dram_read_bytes"] + candidate["dram_write_bytes"]
    bandwidth = moved / (candidate["duration_us"] * 1e-6)
    floor_bytes = candidate["dram_read_bytes"] + NCU["write_floor_bytes_em4"]
    floor_us = floor_bytes / bandwidth * 1e6
    stock_moved = NCU["stock"]["dram_read_bytes"] + NCU["stock"]["dram_write_bytes"]
    return {
        "candidate_achieved_bandwidth_tb_s": bandwidth / 1e12,
        "stock_achieved_bandwidth_tb_s": stock_moved
        / (NCU["stock"]["duration_us"] * 1e-6)
        / 1e12,
        "candidate_bytes_moved_mb": moved / 1e6,
        "read_bytes_mb": candidate["dram_read_bytes"] / 1e6,
        "irreducible_weight_mb": NCU["irreducible_weight_bytes"] / 1e6,
        "read_over_irreducible": candidate["dram_read_bytes"]
        / NCU["irreducible_weight_bytes"],
        "write_bytes_mb": candidate["dram_write_bytes"] / 1e6,
        "write_floor_mb": NCU["write_floor_bytes_em4"] / 1e6,
        "floor_duration_us": floor_us,
        "candidate_duration_us": candidate["duration_us"],
        "max_remaining_headroom_us": candidate["duration_us"] - floor_us,
        "pct_of_achievable_floor": floor_us / candidate["duration_us"] * 100.0,
    }


def _best_case_gate(lanes: list[dict[str, Any]], headroom_us: float) -> list[dict]:
    """Re-score the worst estimator of each region lane at the physical floor."""
    rescored = []
    for lane in lanes:
        if lane["family"] != "moe_compute_region":
            continue
        worst = min(lane["series"], key=lambda item: item["min_estimator"])
        candidate_ms = worst["candidate_median_ms"]
        # Every estimator is a reference/candidate ratio on the same region, so
        # removing the same host-independent device time scales them equally.
        scale = candidate_ms / (candidate_ms - headroom_us / 1000.0)
        rescored.append(
            {
                "expected_m": lane["expected_m"],
                "measured_worst_estimator": worst["min_estimator"],
                "best_case_worst_estimator": worst["min_estimator"] * scale,
                "best_case_passes": worst["min_estimator"] * scale >= THRESHOLD,
            }
        )
    return rescored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.results_dir)
    lanes = [_lane(path) for path in sorted(root.glob("graph_*_em*.json"))]
    contract_path = root / "graph_only_contract.json"
    contract = json.loads(contract_path.read_text())
    roofline = _roofline()

    leaf = [lane for lane in lanes if lane["family"] == "moe_grouped_masked"]
    region = [lane for lane in lanes if lane["family"] == "moe_compute_region"]
    summary = {
        "contract": "glm52-moe-w2-graph-only-campaign-v1",
        "threshold": THRESHOLD,
        "series_gate": "all_four_estimates_each_series_gte_1p03_v1",
        "graph_only_integration": {
            "contract_path": str(contract_path),
            "contract_sha256": _sha256(contract_path),
            "status": contract["status"],
            "eager_declines_before_provider_launch": (
                contract["A_eager_graph_only"]["provider_attempts"] == 0
                and contract["A_eager_graph_only"]["output_untouched"]
            ),
            "capture_selects_one_candidate_node": all(
                item["node_count"] == 1 and item["provider_attempts"] == 1
                for item in contract["C_capture"].values()
            ),
            "captured_graph_identical_across_settings": all(
                contract["D_capture_identical"].values()
            ),
            "eager_containing_identity_estimators": [
                item["performance_estimates"] for item in
                contract["E_eager_containing_identity_lane"]["series"]
            ],
        },
        "lanes": lanes,
        "graph_leaf_gate_passes": bool(leaf) and all(x["gate_passes"] for x in leaf),
        "graph_region_gate_passes": bool(region)
        and all(x["gate_passes"] for x in region),
        "roofline": roofline,
        "best_case_region_gate_at_physical_floor": _best_case_gate(
            lanes, roofline["max_remaining_headroom_us"]
        ),
    }
    summary["disposition"] = (
        "external-acceptance-candidate"
        if summary["graph_leaf_gate_passes"] and summary["graph_region_gate_passes"]
        else "no-replacement"
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "disposition": summary["disposition"],
        "graph_leaf_gate_passes": summary["graph_leaf_gate_passes"],
        "graph_region_gate_passes": summary["graph_region_gate_passes"],
        "max_remaining_headroom_us": roofline["max_remaining_headroom_us"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
