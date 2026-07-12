#!/usr/bin/env bash
# Run one task through kernel-harness's low-level driver in the unified venv.
#
#   run.sh <task_dir> [solution_name] [out_dir] [iterations]
#
# solution_name defaults to solution.py (the candidate an agent edits).
# Pass reference.py to measure the sglang baseline. Correctness is always checked
# against definition.reference (the sglang kernel). Prints traces.json location.
set -euo pipefail

# Resolve paths through bin/config.py (env var -> harness.env -> portable default).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eval "$(python3 "$HERE/config.py")"
export PATH="$VENV/bin:$PATH"
export SGLANG_DIR
export PYTHONPATH="$SGLANG_DIR/python:${PYTHONPATH:-}"
export CUDA_HOME

TASK="${1:?usage: run.sh <task_dir> [solution_name] [out_dir] [iterations]}"
SOL="${2:-solution.py}"
OUT="${3:-/tmp/kernel-harness/$(basename "$TASK")-$SOL}"
ITERS="${4:-50}"

mkdir -p "$OUT"
"$VENV/bin/python" "$HERE/../harness/driver.py" "$TASK" \
  --solution-name "$SOL" --iterations "$ITERS" -o "$OUT"
echo "TRACES=$OUT"
