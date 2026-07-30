#!/usr/bin/env python3
"""Audit combine-campaign timing evidence against this round's promotion policy.

The prior rounds' audit scripts are kept unmodified as the record of what they
did. This round adds one requirement they did not have.

| lane / comparison                        | requirement                          |
|------------------------------------------|--------------------------------------|
| containing_graph vs installed stock      | every estimator, every series >=1.03 |
| leaf_graph vs installed stock            | every estimator, every series >=1.03 |
| containing_graph vs P1 + stock combine   | every estimator, every series >=1.03 |
| leaf_graph vs P1 + stock combine         | every estimator, every series >=1.03 |
| containing_eager                         | stock fallback, zero provider launch |
| leaf_eager                               | diagnostic only                      |

The second pair of rows is plan hypothesis 1: a combine candidate may not bank
the main-kernel win that round 2 already earned. Both denominators must clear the
gate independently, so a bucket is promotable only if it appears in the
stock-relative *and* the P1-relative pass set.

Every estimator is recomputed here from the raw ordered pairs and then compared
against the stored summary, so a corrupted or hand-edited summary cannot pass
the gate. The terminal disposition is derived from the audit rather than
asserted, and both same-binary null spreads measured this round -- the
stock-versus-stock null and the identity-versus-identity provider-pair null --
are reported beside the margins so the reader can see what each instrument could
actually resolve.
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
    parser.add_argument(
        "--name-filter",
        default="",
        help="only audit evidence files whose name contains this substring",
    )
    parser.add_argument(
        "--require-single-gpu",
        action="store_true",
        help="fail unless every audited file was measured on one physical GPU",
    )
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
        or "pair_" in p.name
    )
    if args.name_filter:
        files = [p for p in files if args.name_filter in p.name]
    if not files:
        raise AssertionError("no paired timing evidence found")

    audits = [audit_file(path) for path in files]

    # A ratio is only portable across files that share a physical GPU: the
    # installed-stock arm's absolute leaf time differs about 7 percent between
    # this host's B200s at identical nominal clocks.
    gpu_uuids = sorted({str(a["gpu_uuid"]) for a in audits})
    if args.require_single_gpu and len(gpu_uuids) != 1:
        raise AssertionError(
            f"decision dataset spans more than one physical GPU: {gpu_uuids}"
        )

    def null_band(selected):
        values: list[float] = []
        for a in selected:
            for lane in a["lanes"].values():
                values += [lane["minimum_estimator"], lane["maximum_estimator"]]
        if not values:
            return None
        return {
            "files": [a["path"] for a in selected],
            "min": min(values),
            "max": max(values),
            "halfwidth_pct": max(abs(1.0 - min(values)), abs(max(values) - 1.0)) * 100.0,
            "note": (
                "the identical binary in both arms; anything inside this band is "
                "instrument noise and cannot support a claim of either presence "
                "or absence of an effect"
            ),
        }

    stock_nulls = [a for a in audits if a["comparison"] == "stock_stock"]
    pair_nulls = [
        a
        for a in audits
        if a["comparison"] == "provider_provider" and a["candidate"] == a["reference"]
    ]
    stock_null = null_band(stock_nulls)
    pair_null = null_band(pair_nulls)
    # A pooled null band misrepresents both buckets when they differ. The claim
    # for a bucket may only be read against that bucket's own null.
    null_by_bucket = {
        f"{label}_m{m}": null_band([a for a in selected if int(a["m"]) == m])
        for label, selected in (
            ("stock_vs_stock", stock_nulls),
            ("provider_pair", pair_nulls),
        )
        for m in (16, 32)
    }
    null_by_bucket = {k: v for k, v in null_by_bucket.items() if v}
    null_halfwidth_pct = None if stock_null is None else stock_null["halfwidth_pct"]

    # Both denominators must clear the gate independently for the same bucket.
    def pass_set(comparison, exclude_null):
        found: dict[str, list[int]] = {}
        for a in audits:
            if a["comparison"] != comparison or not a.get("gated_lanes_present"):
                continue
            if exclude_null and a["candidate"] == a["reference"]:
                continue
            if a.get("all_gated_lanes_pass"):
                found.setdefault(str(a["candidate"]), []).append(int(a["m"]))
        return found

    vs_stock = pass_set("stock_provider", exclude_null=False)
    vs_p1 = pass_set("provider_provider", exclude_null=True)
    promotable = {
        candidate: sorted(set(buckets) & set(vs_p1.get(candidate, [])))
        for candidate, buckets in vs_stock.items()
    }
    promotable = {k: v for k, v in promotable.items() if v}

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
                "stock fallback required (zero provider launches); the graph-only "
                "policy does not require a speedup on this lane"
            ),
            "leaf_eager": "diagnostic only",
            "second_denominator": (
                "plan hypothesis 1: the same two graph lanes must also clear 1.03 "
                "in the directly paired comparison against P1 plus the stock "
                "combine, so a combine candidate cannot bank round 2's "
                "main-kernel win"
            ),
        },
        "instrument_null_stock_vs_stock": stock_null,
        "instrument_null_provider_pair": pair_null,
        "instrument_null_by_bucket": null_by_bucket,
        "gpu_uuids": gpu_uuids,
        "single_physical_gpu": len(gpu_uuids) == 1,
        "audits": audits,
        "buckets_passing_vs_installed_stock": vs_stock,
        "buckets_passing_vs_p1_stock_combine": vs_p1,
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
                "pair_null_halfwidth_pct": (
                    None if pair_null is None else pair_null["halfwidth_pct"]
                ),
                "null_halfwidth_pct_by_bucket": {
                    k: round(v["halfwidth_pct"], 3) for k, v in null_by_bucket.items()
                },
                "buckets_passing_vs_installed_stock": vs_stock,
                "buckets_passing_vs_p1_stock_combine": vs_p1,
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
