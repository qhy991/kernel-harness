"""Run every benchmark, collect results, and write JSON + CSV.

Usage:
    python run_all.py [--out <dir>]  [--only linear|grouped|bmm|indexer|attention]

Env vars:
    SGLANG_HARNESS_WARMUP  (default 5)
    SGLANG_HARNESS_ITERS   (default 20)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks import linear_gemm, grouped_gemm_moe, bmm_absorb, indexer_score, attention
from benchmarks._util import device_info


BENCH_MODULES = {
    "linear":    linear_gemm,
    "grouped":   grouped_gemm_moe,
    "bmm":       bmm_absorb,
    "indexer":   indexer_score,
    "attention": attention,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="logs",
                    help="output dir for JSON+CSV (default: ./logs)")
    ap.add_argument("--only", type=str, default=None,
                    help=f"run one bench only: {'|'.join(BENCH_MODULES)}")
    args = ap.parse_args()

    print(device_info())
    print()

    all_results: dict[str, list[dict]] = {}
    for name, mod in BENCH_MODULES.items():
        if args.only and args.only != name:
            continue
        print(f"\n### {name} ".ljust(148, "#"))
        all_results[name] = mod.run()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "results.json"
    csv_path = out_dir / "results.csv"

    with json_path.open("w") as f:
        json.dump(all_results, f, indent=2)

    all_rows: list[dict] = []
    for group, rows in all_results.items():
        for r in rows:
            all_rows.append({"group": group, **r})
    if all_rows:
        fields = ["group"] + sorted({k for r in all_rows for k in r if k != "group"})
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in all_rows:
                w.writerow(r)

    print(f"\n# JSON: {json_path}")
    print(f"# CSV:  {csv_path}")


if __name__ == "__main__":
    main()
