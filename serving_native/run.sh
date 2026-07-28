#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${KERNEL_HARNESS_PYTHON:-${ROOT}/.venv/bin/python}"
PAIRED_SGLANG="$(cd "${ROOT}/.." && pwd)/sglang"

if [[ ! -x "${PY}" ]]; then
  echo "Python environment not found: ${PY}" >&2
  exit 3
fi

if [[ -d "${PAIRED_SGLANG}/python/sglang" ]]; then
  export SGLANG_ROOT="${SGLANG_ROOT:-${PAIRED_SGLANG}}"
else
  export SGLANG_ROOT="${SGLANG_ROOT:-/home/qinhaiyan/sglang}"
fi
export PYTHONPATH="${SGLANG_ROOT}/python:${ROOT}:${PYTHONPATH:-}"

if [[ $# -eq 0 ]]; then
  exec "${PY}" "${ROOT}/serving_native/runner.py" --list
fi

if [[ "$1" == "--list" ]]; then
  exec "${PY}" "${ROOT}/serving_native/runner.py" --list
fi

if [[ "$1" == "--describe" ]]; then
  [[ $# -eq 2 ]] || { echo "usage: run.sh --describe TASK" >&2; exit 3; }
  exec "${PY}" "${ROOT}/serving_native/runner.py" --describe "$2"
fi

TASK="$1"
shift
exec "${PY}" "${ROOT}/serving_native/launch.py" "${TASK}" "$@"
