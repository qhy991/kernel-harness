#!/usr/bin/env python3
"""Merge bench outputs into llm_flops-style comparison CSVs."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "results"

SHARED_OPS = ("o_proj", "index_k_proj", "moe_up_proj", "moe_down_proj")

FIELDNAMES = [
    "run_ts", "phase", "op", "category", "M", "S",
    "stock_ms", "ours_ms", "speedup", "impl", "archive",
    "protocol", "same_inputs", "shape",
]


def _load_perf(path: Path, phase: str, m_col: str) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with path.open() as f:
        for r in csv.DictReader(f):
            avg = float(r.get("avg_ms", 0) or 0)
            if avg <= 0:
                continue
            stock = float(r["stock_ms"])
            ours = avg
            rows.append({
                "phase": phase,
                "op": r["name"],
                "category": r.get("category", ""),
                "M": r[m_col],
                "S": r["S"],
                "stock_ms": stock,
                "ours_ms": ours,
                "speedup": stock / ours if ours else 0.0,
                "impl": r.get("impl", ""),
                "archive": r.get("archive") or "",
                "protocol": r.get("protocol", ""),
                "same_inputs": r.get("same_inputs", "True"),
                "shape": r.get("shape", ""),
            })
    return rows


def _layer_rows(body: list[dict], phase: str) -> list[dict]:
    extra: list[dict] = []
    for M in sorted({int(r["M"]) for r in body}):
        sub = [r for r in body if int(r["M"]) == M]
        stock_sum = sum(r["stock_ms"] for r in sub)
        ours_sum = sum(r["ours_ms"] for r in sub)
        extra.append({
            "phase": phase,
            "op": "__LAYER__",
            "category": "layer",
            "M": str(M),
            "S": sub[0]["S"],
            "stock_ms": stock_sum,
            "ours_ms": ours_sum,
            "speedup": stock_sum / ours_sum if ours_sum else 0.0,
            "impl": "sum",
            "archive": "",
            "protocol": "cuda_graph",
            "same_inputs": "True",
            "shape": "",
        })
    return extra


def _write_csv(path: Path, rows: list[dict], run_ts: str) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow({
                "run_ts": run_ts,
                "phase": r["phase"],
                "op": r["op"],
                "category": r["category"],
                "M": r["M"],
                "S": r["S"],
                "stock_ms": f"{r['stock_ms']:.6f}",
                "ours_ms": f"{r['ours_ms']:.6f}",
                "speedup": f"{r['speedup']:.4f}",
                "impl": r["impl"],
                "archive": r["archive"],
                "protocol": r["protocol"],
                "same_inputs": r["same_inputs"],
                "shape": r.get("shape", ""),
            })


def main() -> int:
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    decode = _load_perf(OUT / "glm5_decode_swapped_perf.csv", "decode", "batch")
    prefill = _load_perf(OUT / "glm5_prefill_swapped_perf.csv", "prefill", "M")

    decode_all = decode + _layer_rows(decode, "decode")
    prefill_all = prefill + _layer_rows(prefill, "prefill")

    _write_csv(OUT / "comparison_decode.csv", decode_all, run_ts)
    _write_csv(OUT / "comparison_prefill.csv", prefill_all, run_ts)
    _write_csv(OUT / "comparison_all.csv", decode_all + prefill_all, run_ts)

    shared = [
        r for r in decode + prefill
        if r["op"] in SHARED_OPS
    ]
    _write_csv(OUT / "comparison_shared_decode_prefill.csv", shared, run_ts)

    meta = {
        "run_ts": run_ts,
        "protocol": "cuda_graph_drop_in",
        "warmup": 5,
        "runs": 20,
        "decode_ops": len(decode),
        "prefill_ops": len(prefill),
        "shared_ops": len(shared),
        "files": [
            "comparison_decode.csv",
            "comparison_prefill.csv",
            "comparison_all.csv",
            "comparison_shared_decode_prefill.csv",
        ],
    }
    (OUT / "comparison_run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
