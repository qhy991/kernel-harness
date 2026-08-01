#!/usr/bin/env python3
"""List, inspect, and validate one-at-a-time experiment recipes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RECIPES = ROOT / "testbench" / "knowledge" / "recipes"
INDEX = RECIPES / "index.json"


def load_catalog() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    catalog = json.loads(INDEX.read_text())
    if catalog.get("schema_version") != 1:
        raise ValueError("recipe index schema_version must be 1")
    entries = catalog.get("recipes")
    if not isinstance(entries, list) or not entries:
        raise ValueError("recipe index must contain recipes")

    recipes: dict[str, dict[str, Any]] = {}
    seen_files: set[str] = set()
    orders: list[int] = []
    for entry in entries:
        recipe_id = entry.get("id")
        filename = entry.get("file")
        order = entry.get("order")
        if not isinstance(recipe_id, str) or not recipe_id:
            raise ValueError("every index entry needs a non-empty id")
        if recipe_id in recipes:
            raise ValueError(f"duplicate recipe id: {recipe_id}")
        if not isinstance(filename, str) or filename in seen_files:
            raise ValueError(f"duplicate or invalid recipe file: {filename!r}")
        if not isinstance(order, int):
            raise ValueError(f"{recipe_id}: order must be an integer")
        path = RECIPES / filename
        if path.parent != RECIPES or not path.is_file():
            raise ValueError(f"{recipe_id}: recipe file missing or outside catalog: {filename}")
        recipe = json.loads(path.read_text())
        if recipe.get("schema_version") != 1:
            raise ValueError(f"{recipe_id}: schema_version must be 1")
        if recipe.get("kind") != "atomic_experiment_recipe":
            raise ValueError(f"{recipe_id}: kind must be atomic_experiment_recipe")
        if recipe.get("id") != recipe_id or path.stem != recipe_id:
            raise ValueError(f"{recipe_id}: id, filename, and catalog disagree")
        if "workflow" in recipe:
            raise ValueError(f"{recipe_id}: bundled workflow is forbidden")
        for key in ("title", "objective", "stop_rule"):
            if not isinstance(recipe.get(key), str) or not recipe[key].strip():
                raise ValueError(f"{recipe_id}: {key} must be one non-empty string")
        for key in ("preconditions", "steps", "artifacts", "origin_evidence"):
            values = recipe.get(key)
            if not isinstance(values, list) or not values or not all(
                isinstance(value, str) and value.strip() for value in values
            ):
                raise ValueError(f"{recipe_id}: {key} must be a non-empty string list")
        if not isinstance(recipe.get("acceptance"), dict) or not recipe["acceptance"]:
            raise ValueError(f"{recipe_id}: acceptance must be a non-empty object")
        recipes[recipe_id] = recipe
        seen_files.add(filename)
        orders.append(order)
    if orders != sorted(orders):
        raise ValueError("recipe index entries must be sorted by order")

    indexed = {entry["file"] for entry in entries}
    unindexed = {
        path.name for path in RECIPES.glob("*.json") if path.name != INDEX.name
    } - indexed
    if unindexed:
        raise ValueError(f"unindexed recipe files: {', '.join(sorted(unindexed))}")
    return catalog, recipes


def render(recipe: dict[str, Any]) -> str:
    lines = [
        f"# {recipe['id']} — {recipe['title']}",
        "",
        f"Objective: {recipe['objective']}",
        "",
        "Preconditions:",
        *[f"  - {value}" for value in recipe["preconditions"]],
        "",
        "Steps:",
        *[f"  {index}. {value}" for index, value in enumerate(recipe["steps"], 1)],
        "",
        "Acceptance:",
        *[f"  - {state}: {value}" for state, value in recipe["acceptance"].items()],
        "",
        f"STOP: {recipe['stop_rule']}",
        "",
        "Artifacts:",
        *[f"  - {value}" for value in recipe["artifacts"]],
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--tag")
    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("id")
    show_parser.add_argument("--json", action="store_true")
    subparsers.add_parser("check")
    args = parser.parse_args()

    try:
        catalog, recipes = load_catalog()
    except Exception as exc:
        parser.error(str(exc))

    if args.command == "check":
        print(f"atomic recipe catalog OK: {len(recipes)} individually runnable recipes")
        return 0
    if args.command == "list":
        for entry in catalog["recipes"]:
            if args.tag and args.tag not in entry.get("tags", []):
                continue
            recipe = recipes[entry["id"]]
            print(
                f"{entry['id']:38s} {entry['category']:15s} {recipe['title']}"
            )
        return 0

    recipe = recipes.get(args.id)
    if recipe is None:
        parser.error(f"unknown recipe id: {args.id}")
    print(json.dumps(recipe, indent=2, sort_keys=True) if args.json else render(recipe))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
