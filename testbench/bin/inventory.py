#!/usr/bin/env python3
"""Print the authoritative task inventory from task.json files.

With --headroom, annotate each task with agent-facing difficulty / integration
contract so sessions pick high-leverage work first.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_BIN = Path(__file__).resolve().parent
_TASKS = _BIN.parent / "tasks"

# Heuristic headroom for agent routing (not a measured SOL; from AGENTS.md + WIN rates).
# high  = fusion / memory-bound wrappers with launch/copy headroom
# medium = sparse / dispatch / layout tricks sometimes win
# low   = production DeepGEMM / tightly-tuned GEMMs near the floor
HIGH_FAMILIES = {
    "swiglu", "swiglu-fp8-quant", "embedding", "rmsnorm", "gemma-rmsnorm",
    "fused-add-rmsnorm", "gemma-fused-add-rmsnorm", "rope", "act-fp8-quant",
    "moe-combine",
}
MEDIUM_FAMILIES = {
    "sparse-mla-decode", "grouped-moe", "grouped-moe-contiguous", "bmm",
    "dsa-qknorm-rope", "dsa-decode-topk", "dsa-store-kv-index",
    "dsa-decode-attn", "dsa-prefill-attn", "dsa-prefill-topk", "moe-gate",
}
LOW_FAMILIES = {
    "fp8-linear-gemm", "bf16-linear", "router-gemm", "lm-head", "nvfp4-moe",
}


def _headroom(family: str) -> str:
    if family in HIGH_FAMILIES:
        return "high"
    if family in MEDIUM_FAMILIES:
        return "medium"
    if family in LOW_FAMILIES:
        return "low"
    return "unknown"


def _contract(task_dir: Path) -> dict:
    sys.path.insert(0, str(_BIN))
    import integrate as ig  # noqa: WPS001

    meta = json.loads((task_dir / "task.json").read_text())
    family = meta.get("family")
    model = meta.get("model") or task_dir.parent.name
    if family in ig.FUSED_ONLY:
        return "fused-only"
    if ig._resolve_recipe(family, model) is None:
        return "unsupported"
    return "drop-in"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headroom", action="store_true",
                    help="annotate headroom + integration contract per task")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("model", nargs="?", help="optional model filter, e.g. glm52")
    args = ap.parse_args()

    rows = []
    unique_families = set()
    total = 0
    for model_dir in sorted(p for p in _TASKS.iterdir() if p.is_dir()):
        if args.model and model_dir.name != args.model:
            continue
        families = Counter()
        for task_file in sorted(model_dir.glob("*/task.json")):
            meta = json.loads(task_file.read_text())
            family = meta["family"]
            families[family] += 1
            task_dir = task_file.parent
            row = {
                "task": f"{model_dir.name}/{task_dir.name}",
                "model": model_dir.name,
                "family": family,
                "phase": meta.get("phase"),
                "op": meta.get("op"),
                "headroom": _headroom(family),
            }
            if args.headroom:
                try:
                    row["integration_contract"] = _contract(task_dir)
                except Exception as e:
                    row["integration_contract"] = f"error:{e}"
            rows.append(row)
        if not families:
            continue
        count = sum(families.values())
        total += count
        unique_families.update(families)
        if not args.headroom and not args.json:
            print(f"{model_dir.name}: {count} tasks, {len(families)} families")
            for family, family_count in sorted(families.items()):
                print(f"  {family}: {family_count}")

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    if args.headroom:
        print(f"{'task':45s} {'headroom':8s} {'contract':12s} family")
        # Prefer high headroom first for agent routing.
        order = {"high": 0, "medium": 1, "unknown": 2, "low": 3}
        for r in sorted(rows, key=lambda x: (order.get(x["headroom"], 9), x["task"])):
            print(f"{r['task']:45s} {r['headroom']:8s} "
                  f"{r.get('integration_contract', ''):12s} {r['family']}")
        by_h = Counter(r["headroom"] for r in rows)
        print(f"\n{len(rows)} tasks — headroom "
              + ", ".join(f"{k}={by_h[k]}" for k in ("high", "medium", "low", "unknown")
                          if by_h[k]))
        return

    print(f"total: {total} tasks, {len(unique_families)} unique families")


if __name__ == "__main__":
    main()
