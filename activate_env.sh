#!/usr/bin/env bash
# Source after running ./testbench/setup_env.sh.

_KH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eval "$(python3 "$_KH_ROOT/testbench/bin/config.py")"
export VIRTUAL_ENV="$VENV"
export SGLANG_DIR CUDA_HOME
export PYTHON="$VIRTUAL_ENV/bin/python"
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="${SGLANG_DIR}/python:${PYTHONPATH:-}"
unset _KH_ROOT VENV MM_M3_SGLANG_DIR
