#!/usr/bin/env bash
# Source before running kernel-harness or official sglang tests:
#   source /home/qinhaiyan/kernel-harness/activate_env.sh

export SGLANG_DIR="${SGLANG_DIR:-/home/qinhaiyan/sglang}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export VIRTUAL_ENV="/home/qinhaiyan/sglang-exp/glm52-kernel-opt/.venv"
export PYTHON="${VIRTUAL_ENV}/bin/python3.12"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONPATH="${SGLANG_DIR}/python:${PYTHONPATH:-}"
