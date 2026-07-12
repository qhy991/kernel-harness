#!/usr/bin/env python3
"""Print the authoritative task inventory from task.json files."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def main() -> None:
    tasks_root = Path(__file__).resolve().parent.parent / "tasks"
    unique_families = set()
    total = 0
    for model_dir in sorted(path for path in tasks_root.iterdir() if path.is_dir()):
        families = Counter()
        for task_file in sorted(model_dir.glob("*/task.json")):
            metadata = json.loads(task_file.read_text())
            families[metadata["family"]] += 1
        if not families:
            continue
        count = sum(families.values())
        total += count
        unique_families.update(families)
        print(f"{model_dir.name}: {count} tasks, {len(families)} families")
        for family, family_count in sorted(families.items()):
            print(f"  {family}: {family_count}")
    print(f"total: {total} tasks, {len(unique_families)} unique families")


if __name__ == "__main__":
    main()
