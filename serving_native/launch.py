"""Launch one serving-native task locally or with torchrun."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving_native.workloads import get_workload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    workload = get_workload(args.task)
    # Keep the venv entry point instead of resolving its symlink to the base
    # interpreter.  ``python -m torch.distributed.run`` then preserves the
    # exact package set used by local tasks on every spawned rank.
    python = Path(sys.executable)
    runner = HERE / "runner.py"
    base = [str(runner), "--task", workload.name, *args.runner_args]

    if workload.world_size == 1:
        command = [str(python), *base]
    else:
        command = [
            str(python),
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={workload.world_size}",
            *base,
        ]

    if args.dry_run:
        print(" ".join(command))
        return 0

    if workload.world_size > 1:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            visible_count = len([item for item in visible.split(",") if item.strip()])
            if visible_count < workload.world_size:
                raise RuntimeError(
                    f"{workload.name} requires {workload.world_size} visible GPUs; "
                    f"CUDA_VISIBLE_DEVICES exposes {visible_count}"
                )
    return subprocess.call(command, cwd=REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
