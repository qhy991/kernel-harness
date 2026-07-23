#!/usr/bin/env bash
# Multi-process entry point for this communication task.
# Every rank runs its own Python; the process group is the collective.
#
#   ./run.sh --describe                   # what is this problem? (single process; no dist)
#   ./run.sh                              # full sweep (WORLD_SIZE default 8)
#   ./run.sh --M 1024                      # one shape
#   ./run.sh --repeat 1                   # fast probe
#   KERNEL_HARNESS_COMM_WORLD_SIZE=4 ./run.sh   # 4-GPU smoke on a partial node
#   ./run.sh --candidate ~/my_kernel.py   # any .py defining run(inputs)
#
# Exit: 0 correct+fast · 1 correct+not-faster · 2 incorrect · 3 infra
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTBENCH="$(cd "$HERE/../../.." && pwd)"
REPO="$(cd "$TESTBENCH/.." && pwd)"
PYTHON="${REPO}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

WORLD_SIZE="${KERNEL_HARNESS_COMM_WORLD_SIZE:-8}"

# --describe short-circuits the multi-process launch — the problem statement
# is single-process readable.
if [[ "${1:-}" == "--describe" ]]; then
  exec "$PYTHON" "$TESTBENCH/harness/evaluate_comm_task.py" "$HERE" "$@"
fi

# All ranks return 0 to torchrun (avoids ChildFailedError banner); rank 0
# writes runs/comm/<task>/last_exit_code with the real 0/1/2/3 verdict.
TASK_NAME="$(basename "$HERE")"
LAST_EXIT="$REPO/runs/comm/$TASK_NAME/last_exit_code"
rm -f "$LAST_EXIT"

"$PYTHON" -m torch.distributed.run \
  --standalone --nproc-per-node="$WORLD_SIZE" \
  "$TESTBENCH/harness/evaluate_comm_task.py" "$HERE" "$@"

# Reconstitute the verdict from the file rank 0 wrote.
if [[ -f "$LAST_EXIT" ]]; then
  exit "$(cat "$LAST_EXIT")"
fi
echo "[run.sh] rank 0 did not write $LAST_EXIT — treating as infra error" >&2
exit 3
