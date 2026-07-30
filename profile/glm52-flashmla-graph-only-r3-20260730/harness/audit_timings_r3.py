#!/usr/bin/env python3
"""Audit round-3 timing evidence against the graph-only promotion policy.

Round-2's audit script is kept unmodified as the record of what round 2 did. It
gated all four lanes and asserted ``no-replacement``; neither is right here.

Round-3 policy:

| lane              | requirement                                          |
|-------------------|------------------------------------------------------|
| containing_graph  | every estimator, every series, >= 1.03 versus stock  |
| leaf_graph        | every estimator, every series, >= 1.03 versus stock  |
| containing_eager  | stock fallback: zero provider launches, no speedup   |
| leaf_eager        | diagnostic only                                      |

Every estimator is recomputed here from the raw ordered pairs and then compared
against the stored summary, so a corrupted or hand-edited summary cannot pass
the gate. The terminal disposition is derived from the audit rather than
asserted, and the same-binary null spread measured this round is reported beside
the margins so the reader can see what the instrument could actually resolve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


GATED_LANES = ("containing_graph", "leaf_graph")
FALLBACK_LANE = "containing_eager"
ESTIMATORS = (
    "pooled_ratio_of_medians",
    "ab_ratio_of_medians",
    "ba_ratio_of_medians",
    "order_balanced_sqrt_ab_ba",
)
GATE = 1.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio_of_medians(rows) -> float:
    return statistics.median(float(r["a_us"]) for r in rows) / statistics.median(
        float(r["b_us"]) for r in rows
    )


def recompute(rows) -> dict[str, float]:
    ab = [r for r in rows if r["order"] == "AB"]
    ba = [r for r in rows if r["order"] == "BA"]
    if not ab or not ba:
        raise AssertionError("both AB and BA observations are required")
    ab_m = ratio_of_medians(ab)
    ba_m = ratio_of_medians(ba)
    return {
        "pooled_ratio_of_medians": ratio_of_medians(rows),
        "ab_ratio_of_medians": ab_m,
        "ba_ratio_of_medians": ba_m,
        "order_balanced_sqrt_ab_ba": math.sqrt(ab_m * ba_m),
    }


def audit_file(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    timing = data["timing"]
    if (
        timing["series"] < 3
        or timing["pairs_per_series"] < 100
        or timing["ab_pairs_per_series"] < 50
        or timing["ba_pairs_per_series"] < 50
    ):
        raise AssertionError(f"{path}: insufficient fairness observations")

    lanes: dict[str, object] = {}
    for lane, lane_data in data["lanes"].items():
        values: list[float] = []
        failures: list[dict[str, object]] = []
        drift: list[dict[str, object]] = []
        for series in lane_data["series"]:
            rows = series["raw_pairs"]
            summary = series["summary"]
            if (
                summary["pairs"] != timing["pairs_per_series"]
                or summary["ab_pairs"] != timing["ab_pairs_per_series"]
                or summary["ba_pairs"] != timing["ba_pairs_per_series"]
                or len(rows) != timing["pairs_per_series"]
            ):
                raise AssertionError(f"{path}:{lane}: pair count drift")
            mine = recompute(rows)
            for estimator in ESTIMATORS:
                value = mine[estimator]
                stored = float(summary["estimators"][estimator])
                if not math.isfinite(value):
                    raise AssertionError(f"{path}:{lane}:{estimator} not finite")
                # Recomputed and stored must agree to floating-point noise.
                if abs(value - stored) > 1e-9:
                    drift.append(
                        {
                            "series": series["series"],
                            "estimator": estimator,
                            "recomputed": value,
                            "stored": stored,
                        }
                    )
                values.append(value)
                if value < GATE:
                    failures.append(
                        {
                            "series": series["series"],
                            "estimator": estimator,
                            "value": value,
                        }
                    )
        if drift:
            raise AssertionError(f"{path}:{lane}: stored summary disagrees: {drift}")
        lanes[lane] = {
            "recomputed_from_raw_pairs": True,
            "estimator_count": len(values),
            "minimum_estimator": min(values),
            "maximum_estimator": max(values),
            "meets_1_03_everywhere": not failures,
            "failed_observations": failures,
            "provider_launches_during_lane": lane_data.get(
                "provider_launches_during_lane"
            ),
        }

    gated = [lane for lane in GATED_LANES if lane in lanes]
    result: dict[str, object] = {
        "path": str(path),
        "sha256": sha256(path),
        "m": data["m"],
        "comparison": data["comparison"],
        "candidate": data["b"],
        "reference": data["a"],
        "gpu_uuid": data["gpu"]["uuid"],
        "series": timing["series"],
        "pairs_per_series": timing["pairs_per_series"],
        "replays_per_observation": timing["replays_per_observation"],
        "lanes": lanes,
        "gated_lanes_present": gated,
    }
    if gated:
        result["all_gated_lanes_pass"] = all(
            lanes[lane]["meets_1_03_everywhere"] for lane in gated
        )
    if FALLBACK_LANE in lanes:
        launches = lanes[FALLBACK_LANE]["provider_launches_during_lane"]
        result["eager_containing_is_stock_fallback"] = launches == 0
    return result


def main() -> int:
    args = parse_args()
    evidence_dir = args.evidence_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {output}")

    files = sorted(
        p
        for p in evidence_dir.glob("*_m*.json")
        if "graph_gate_k20" in p.name
        or "eager_k20" in p.name
        or "null_stock_stock" in p.name
    )
    if not files:
        raise AssertionError("no paired timing evidence found")

    audits = [audit_file(path) for path in files]

    nulls = [a for a in audits if a["comparison"] == "stock_stock"]
    null_values: list[float] = []
    for a in nulls:
        for lane in a["lanes"].values():
            null_values += [lane["minimum_estimator"], lane["maximum_estimator"]]
    null_halfwidth_pct = (
        max(abs(1.0 - min(null_values)), abs(max(null_values) - 1.0)) * 100.0
        if null_values
        else None
    )

    # A bucket is promotable when both graph lanes clear 1.03 against stock.
    promotable: dict[str, list[int]] = {}
    for a in audits:
        if a["comparison"] != "stock_provider":
            continue
        if not a.get("gated_lanes_present"):
            continue
        if a.get("all_gated_lanes_pass"):
            promotable.setdefault(str(a["candidate"]), []).append(int(a["m"]))

    evidence = {
        "schema_version": 1,
        "policy": {
            "gated_lanes": list(GATED_LANES),
            "gate": GATE,
            "requirement": (
                "every estimator in every one of three independent 50-AB/50-BA "
                "series must be finite and >= 1.03 against installed stock"
            ),
            "containing_eager": (
                "stock fallback required (zero provider launches); round-3 "
                "policy does not require a speedup on this lane"
            ),
            "leaf_eager": "diagnostic only",
        },
        "instrument_null_this_round": {
            "files": [a["path"] for a in nulls],
            "min": min(null_values) if null_values else None,
            "max": max(null_values) if null_values else None,
            "halfwidth_pct": null_halfwidth_pct,
            "note": (
                "same installed stock binary in both arms; anything inside this "
                "band is instrument noise and cannot support a claim of either "
                "presence or absence of an effect"
            ),
        },
        "audits": audits,
        "promotable_buckets_by_candidate": promotable,
        "promotable_bucket_count": sum(len(v) for v in promotable.values()),
    }
    evidence["terminal_disposition"] = (
        "external-acceptance-candidate"
        if evidence["promotable_bucket_count"] > 0
        else "no-replacement"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "files_audited": len(audits),
                "null_halfwidth_pct": null_halfwidth_pct,
                "promotable_buckets_by_candidate": promotable,
                "terminal_disposition": evidence["terminal_disposition"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
