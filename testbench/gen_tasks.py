#!/usr/bin/env python3
"""Generate standalone FlashInfer-Bench-format testbench tasks (Kimi-K2.7 + MiniMax-M3).

Thin runner over the `taskgen` package: every kernel task is a declarative `TaskSpec`
(taskgen/families/*.py, grouped by kernel family) written by one canonical writer
(taskgen/spec.py:write_task). Adding a kernel = declare a spec, not copy boilerplate.

    python gen_tasks.py                 # regenerate all tasks under tasks/
    python gen_tasks.py --out /tmp/x    # generate into a scratch dir (for diffing)
"""
import argparse
from collections import Counter
from pathlib import Path

from taskgen import all_specs, write_task
from taskgen.config import SOLEXEC

ROOT = Path(__file__).resolve().parent

RUN_SH = (
    "#!/usr/bin/env bash\n"
    "# Self-contained entrypoint: evaluate THIS task folder against the sglang\n"
    "# baseline. Forwards args to bin/evaluate.py (e.g. ./run.sh --repeat 3).\n"
    "set -euo pipefail\n"
    'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
    'exec python3 "$HERE/../../../bin/evaluate.py" "$HERE" "$@"\n'
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "tasks",
                    help="tasks root to write into (default: testbench/tasks)")
    args = ap.parse_args()
    tasks_root = args.out

    specs = all_specs()
    per_family = Counter()
    for spec in specs:
        write_task(spec, tasks_root, SOLEXEC)
        per_family[(spec.model, spec.family)] += 1

    for (model, family), n in sorted(per_family.items()):
        print(f"generated {n} {model}/{family} tasks")

    # per-task run.sh so each folder is self-describing
    count = 0
    for d in sorted(tasks_root.glob("*/*")):
        if (d / "definition.json").exists():
            p = d / "run.sh"
            p.write_text(RUN_SH)
            p.chmod(0o755)
            count += 1
    print(f"\ntotal: {len(specs)} tasks across {len(per_family)} families; "
          f"wrote run.sh into {count} dirs")


if __name__ == "__main__":
    main()
