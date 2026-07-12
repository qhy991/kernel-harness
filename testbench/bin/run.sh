#!/usr/bin/env bash
# Run one testbench task through the sol-execbench harness in the unified sglang venv.
#
#   run.sh <task_dir> [solution_name] [out_dir] [iterations]
#
# solution_name defaults to solution.py (the candidate an agent edits).
# Pass reference.py to measure the sglang baseline. Correctness is always checked
# against definition.reference (the sglang kernel). Prints traces.json location.
set -euo pipefail

# Resolve paths through bin/config.py (env var -> harness.env -> built-in default),
# so this script has no hardcoded machine paths.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eval "$(python3 "$HERE/config.py")"   # sets VENV SOLEXEC SGLANG_DIR CUDA_HOME
export PATH="$VENV/bin:$PATH"
export SGLANG_DIR
export PYTHONPATH="$SGLANG_DIR/python:${PYTHONPATH:-}"
export CUDA_HOME

TASK="${1:?usage: run.sh <task_dir> [solution_name] [out_dir] [iterations]}"
SOL="${2:-solution.py}"
OUT="${3:-/tmp/kersor-tb/$(basename "$TASK")-$SOL}"
ITERS="${4:-50}"

mkdir -p "$OUT"
cd "$SOLEXEC"
python scripts/run_dataset.py "$TASK" \
  --solution-name "$SOL" --iterations "$ITERS" --rerun -o "$OUT"
echo "TRACES=$OUT"
