#!/usr/bin/env bash
# Source after running ./testbench/setup_env.sh.

_KH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eval "$(python3 "$_KH_ROOT/testbench/bin/config.py")"
export VIRTUAL_ENV="$VENV"
export SGLANG_DIR CUDA_HOME AITER_PATH
export KERNEL_HARNESS_PLATFORM KERNEL_HARNESS_PROFILE
export KERNEL_HARNESS_PROVIDER KERNEL_HARNESS_TIMER
export PYTHON="$VIRTUAL_ENV/bin/python"
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="${SGLANG_DIR}/python:${PYTHONPATH:-}"
unset _KH_ROOT VENV MM_M3_SGLANG_DIR
