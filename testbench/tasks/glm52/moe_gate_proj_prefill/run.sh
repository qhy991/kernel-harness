#!/usr/bin/env bash
# The single entry point for this task.
#
#   ./run.sh --describe      # what is this problem? (generated from glm52_ops)
#   ./run.sh                 # full sweep, 3 samples per shape — the gate
#   ./run.sh --M 1024          # one shape
#   ./run.sh --repeat 1      # fast probe. CANNOT gate a win.
#
# Exit: 0 correct+fast · 1 correct+not-faster · 2 incorrect · 3 infra/contract error
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTBENCH="$(cd "$HERE/../../.." && pwd)"
REPO="$(cd "$TESTBENCH/.." && pwd)"
PYTHON="${REPO}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi
exec "$PYTHON" "$TESTBENCH/harness/evaluate_task.py" "$HERE" "$@"
