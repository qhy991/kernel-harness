#!/usr/bin/env bash
# The single entry point for this task.
#
#   ./run.sh --describe          # what is this problem? (generated from glm52_ops)
#   ./run.sh --describe --json   # ...the same thing, machine-readable (== problem.json)
#   ./run.sh                 # full sweep; defaults warmup=3, repeat=10
#   ./run.sh --M 1024         # one shape
#   ./run.sh --repeat 1      # fast probe. CANNOT gate a win.
#
# To test a kernel that is NOT this directory's candidate.py — the usual case, since
# nothing should have to edit the task to be measured:
#
#   ./run.sh --candidate ~/my_kernels/o_proj.py    # any .py defining run(inputs)
#   ./run.sh --candidate ~/my_kernels/             # or a dir holding candidate.py
#
# Exit: 0 correct+fast · 1 correct+not-faster · 2 incorrect · 3 infra/contract error
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTBENCH="$(cd "$HERE/../../.." && pwd)"
REPO="$(cd "$TESTBENCH/.." && pwd)"
PYTHON="${ROCM_TORCH_PYTHON:-}"
if [[ -z "$PYTHON" && -n "${ROCM_TORCH_VENV:-}" && -x "${ROCM_TORCH_VENV}/bin/python" ]]; then
  PYTHON="${ROCM_TORCH_VENV}/bin/python"
fi
if [[ -z "$PYTHON" ]]; then
  PYTHON="${REPO}/.venv/bin/python"
fi
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi
export KERNEL_HARNESS_PLATFORM="${KERNEL_HARNESS_PLATFORM:-rocm}"
export KERNEL_HARNESS_PROFILE="${KERNEL_HARNESS_PROFILE:-amd-mi300x}"
export KERNEL_HARNESS_PROVIDER="${KERNEL_HARNESS_PROVIDER:-aiter-torch-reference}"
export KERNEL_HARNESS_TIMER="${KERNEL_HARNESS_TIMER:-event}"
export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
exec "$PYTHON" "$TESTBENCH/harness/evaluate_task.py" "$HERE" "$@"
