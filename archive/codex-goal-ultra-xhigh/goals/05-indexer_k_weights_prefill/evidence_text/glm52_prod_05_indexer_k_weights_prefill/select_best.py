#!/usr/bin/env python3
"""Select the strongest successful isolated candidate without inventing data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.results_dir.glob("isolated_*_sweep.json")):
        data = json.loads(path.read_text())
        candidate = data.get("candidate")
        if not candidate:
            continue
        metadata = candidate.get("metadata") or {}
        rows.append(
            {
                "result": str(path.resolve()),
                "candidate_path": candidate["path"],
                "backend": metadata.get("backend", path.stem),
                "speedup": candidate["speedup"],
                "paired_p10_speedup": candidate["paired_p10_speedup"],
                "paired_p90_speedup": candidate["paired_p90_speedup"],
                "passes_3pct_median_gate": candidate["passes_3pct_median_gate"],
            }
        )
    if not rows:
        raise SystemExit("no successful isolated candidate sweep found")
    rows.sort(key=lambda row: row["speedup"], reverse=True)
    selection = {"selected": rows[0], "ranked_candidates": rows}
    args.output.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    print(json.dumps(selection, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
