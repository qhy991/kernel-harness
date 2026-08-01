#!/usr/bin/env bash
# The single entry point for this task.
#
#   ./run.sh --describe          # what is this problem? (generated from glm52_ops)
#   ./run.sh --describe --json   # ...the same thing, machine-readable (== problem.json)
#   ./run.sh                 # full sweep; defaults warmup=8, repeat=10
#   ./run.sh --M 1024         # one shape
#   ./run.sh --repeat 1      # fast probe. CANNOT gate a win.
#
# To test a kernel that is NOT this directory's candidate.py — the usual case, since
# nothing should have to edit the task to be measured:
#
#   ./run.sh --candidate ~/my_kernels/o_proj.py    # any .py defining run(inputs)
#   ./run.sh --candidate ~/my_kernels/             # or a dir holding candidate.py
#
# Exit: 0 gate-eligible correct+fast · 1 no-win/probe/unstable · 2 incorrect/invalid
#       3 infrastructure, timeout, or contract error
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTBENCH="$(cd "$HERE/../../.." && pwd)"
REPO="$(cd "$TESTBENCH/.." && pwd)"
PYTHON="${REPO}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi
TIMEOUT_SECONDS="${KERNEL_HARNESS_TIMEOUT_SECONDS:-1800}"
exec "$PYTHON" "$TESTBENCH/bin/supervise.py" --timeout "$TIMEOUT_SECONDS" --   "$PYTHON" "$TESTBENCH/harness/evaluate_task.py" "$HERE" "$@"
