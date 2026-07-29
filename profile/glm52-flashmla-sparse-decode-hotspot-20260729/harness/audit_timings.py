#!/usr/bin/env python3
"""Audit every retained paired-series timing file against the promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


FILES = {
    "b1_coord_ready": ("b1_paired_m16.json", "b1_paired_m32.json"),
    "b2_stagger_nope": ("b2_paired_m16.json", "b2_paired_m32.json"),
    "b3_native_fp8_bf16": ("b3_paired_m16.json", "b3_paired_m32.json"),
    "b4_evict_first": ("b4a_paired_m16.json", "b4a_paired_m32.json"),
    "b4_rope_early_warp": ("b4b_paired_m16.json", "b4b_paired_m32.json"),
    "b5_exact_no_extra": ("b5_paired_m16.json", "b5_paired_m32.json"),
    "b3_b5_native_exact": (
        "b3_b5_paired_m16.json",
        "b3_b5_paired_m32.json",
    ),
}
LANES = ("leaf_eager", "leaf_graph", "containing_eager", "containing_graph")
ESTIMATORS = (
    "pooled_ratio_of_medians",
    "ab_ratio_of_medians",
    "ba_ratio_of_medians",
    "order_balanced_sqrt_ab_ba",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    evidence_dir = args.evidence_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {output}")

    candidates: list[dict[str, object]] = []
    for variant, filenames in FILES.items():
        buckets: list[dict[str, object]] = []
        for filename in filenames:
            path = evidence_dir / filename
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
            for lane in LANES:
                lane_data = data["lanes"][lane]
                values: list[float] = []
                failures: list[dict[str, object]] = []
                for series in lane_data["series"]:
                    summary = series["summary"]
                    if (
                        summary["pairs"] != timing["pairs_per_series"]
                        or summary["ab_pairs"] != timing["ab_pairs_per_series"]
                        or summary["ba_pairs"] != timing["ba_pairs_per_series"]
                    ):
                        raise AssertionError(f"{path}:{lane}: pair count drift")
                    for estimator in ESTIMATORS:
                        value = float(summary["estimators"][estimator])
                        if not math.isfinite(value):
                            raise AssertionError(
                                f"{path}:{lane}:{estimator} is not finite"
                            )
                        values.append(value)
                        if value < 1.03:
                            failures.append(
                                {
                                    "series": series["series"],
                                    "estimator": estimator,
                                    "value": value,
                                }
                            )
                passes = not failures
                if passes != lane_data[
                    "passes_every_series_every_estimator_1_03"
                ]:
                    raise AssertionError(f"{path}:{lane}: stored gate flag mismatch")
                lanes[lane] = {
                    "passes": passes,
                    "minimum_estimator": min(values),
                    "maximum_estimator": max(values),
                    "failed_observations": failures,
                }
            buckets.append(
                {
                    "m": data["m"],
                    "path": str(path),
                    "sha256": sha256(path),
                    "candidate_extension_sha256": data["candidate_extension"][
                        "sha256"
                    ],
                    "gpu_uuid": data["gpu"]["uuid"],
                    "series": timing["series"],
                    "pairs_per_series": timing["pairs_per_series"],
                    "ab_pairs_per_series": timing["ab_pairs_per_series"],
                    "ba_pairs_per_series": timing["ba_pairs_per_series"],
                    "lanes": lanes,
                    "passes_all_four_lanes": all(
                        lane["passes"] for lane in lanes.values()
                    ),
                }
            )
        candidates.append(
            {
                "variant": variant,
                "buckets": buckets,
                "promotable_bucket_count": sum(
                    bucket["passes_all_four_lanes"] for bucket in buckets
                ),
            }
        )

    evidence = {
        "schema_version": 1,
        "gate": (
            "every estimator in every one of three independent 50-AB/50-BA "
            "series must be finite and >=1.03 for all four lanes"
        ),
        "candidates": candidates,
        "candidate_count": len(candidates),
        "bucket_count": sum(len(item["buckets"]) for item in candidates),
        "promotable_bucket_count": sum(
            item["promotable_bucket_count"] for item in candidates
        ),
        "terminal_disposition": "no-replacement",
    }
    if evidence["promotable_bucket_count"] != 0:
        raise AssertionError("a promotable bucket exists; no-replacement is invalid")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "candidate_count": evidence["candidate_count"],
                "bucket_count": evidence["bucket_count"],
                "promotable_bucket_count": 0,
                "terminal_disposition": "no-replacement",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
