#!/usr/bin/env python3
"""Print per-task SGLang drop-in integration contract for agent routing.

Helps agents know BEFORE optimizing whether a WIN will require integrate.py,
whether the family is fused-only (no isolated symbol), or which recipe applies.

  python3 testbench/bin/integration_status.py [task_or_model/task ...]
  python3 testbench/bin/integration_status.py --json glm52
  python3 testbench/bin/integration_status.py   # all tasks
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import contract tables from integrate without running GPU code.
_BIN = Path(__file__).resolve().parent
sys.path.insert(0, str(_BIN))
import integrate as ig  # noqa: E402

_TASKS = _BIN.parent / "tasks"


def _contract(task_dir: Path) -> dict:
    meta = json.loads((task_dir / "task.json").read_text())
    family = meta.get("family")
    model = meta.get("model") or task_dir.parent.name
    name = f"{model}/{task_dir.name}"

    if family in ig.FUSED_ONLY:
        return {
            "task": name,
            "family": family,
            "model": model,
            "integration_contract": "fused-only",
            "integrate_expected": "no-recipe",
            "drop_in_site": None,
            "note": ig.FUSED_ONLY[family],
        }

    recipe = ig._resolve_recipe(family, model)
    if recipe is None:
        return {
            "task": name,
            "family": family,
            "model": model,
            "integration_contract": "unsupported",
            "integrate_expected": "not-run",
            "drop_in_site": None,
            "note": f"no integration recipe for family={family!r}",
        }

    site = "model-specific" if (family, model) in ig.MODEL_RECIPES else "family-default"
    return {
        "task": name,
        "family": family,
        "model": model,
        "integration_contract": "drop-in",
        "integrate_expected": "pass-or-fail",
        "drop_in_site": site,
        "note": f"WIN should run integrate.py; patches real sglang dispatch ({site})",
    }


def _iter_tasks(selectors: list[str]):
    if not selectors:
        for model_dir in sorted(_TASKS.iterdir()):
            if not model_dir.is_dir():
                continue
            for task_dir in sorted(model_dir.iterdir()):
                if (task_dir / "task.json").exists():
                    yield task_dir
        return

    for sel in selectors:
        p = Path(sel)
        if p.is_dir() and (p / "task.json").exists():
            yield p.resolve()
            continue
        if "/" in sel:
            model, task = sel.split("/", 1)
            td = _TASKS / model / task
            if (td / "task.json").exists():
                yield td
            else:
                raise SystemExit(f"unknown task: {sel}")
            continue
        model_dir = _TASKS / sel
        if model_dir.is_dir():
            for task_dir in sorted(model_dir.iterdir()):
                if (task_dir / "task.json").exists():
                    yield task_dir
        else:
            raise SystemExit(f"unknown selector: {sel}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("selectors", nargs="*", help="task dir, model/task, or model")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = [_contract(td) for td in _iter_tasks(args.selectors)]
    if args.json:
        print(json.dumps(rows, indent=2))
        return

    for r in rows:
        print(f"{r['task']:45s}  {r['integration_contract']:12s}  {r['integrate_expected']}")
    print(f"\n{len(rows)} tasks")


if __name__ == "__main__":
    main()
