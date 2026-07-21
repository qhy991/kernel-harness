#!/usr/bin/env bash
# Rebuild the AMD MI300X GLM-5.2 benchmark environment.
#
# The venv lives on /opt/mizar (a 1.5 TB tmpfs — NOT persistent across reboots), because
# it is the only large writable mount on this node (home is a 15 GB overlay; /mnt/public
# and /opt/aeon are read-only). Re-run this after a reboot to recreate it.
#
# Target (aligned with the B-card server, adjusted to what exists for ROCm):
#   python 3.12  |  torch 2.11.0.dev+rocm7.0  |  pytorch-triton-rocm  |  sglang 0.5.15
# ROCm 7.0 is already installed at /opt/rocm (see ~/.bashrc ROCm block).
set -euo pipefail

VENV=/opt/mizar/huyan/venvs/amd-glm52
export UV_CACHE_DIR=/opt/mizar/huyan/.uv-cache
UV=/opt/conda/bin/uv
PYBASE=/opt/conda/envs/sol-execbench/bin/python3.12   # an existing cpython 3.12 (avoids a slow standalone download)
USTC=https://mirrors.ustc.edu.cn/pypi/web/simple      # domestic mirror (fast); already the default pip index

echo ">>> [1/4] creating venv (python 3.12) at $VENV"
mkdir -p "$UV_CACHE_DIR" "$(dirname "$VENV")"
"$UV" venv --python "$PYBASE" "$VENV"

echo ">>> [2/4] installing torch 2.11 dev + rocm7.0 + triton-rocm (nightly channel, ~4.5 GB)"
"$UV" pip install --python "$VENV/bin/python" \
  --index-url https://download.pytorch.org/whl/nightly/rocm7.0 --pre \
  torch pytorch-triton-rocm

echo ">>> [3/4] installing numeric + sglang runtime deps (USTC mirror)"
"$UV" pip install --python "$VENV/bin/python" --index-url "$USTC" \
  numpy pandas orjson pyzmq requests tqdm packaging pillow psutil \
  transformers pydantic uvicorn fastapi aiohttp interegular

echo ">>> [4/4] installing sglang 0.5.15 (no-deps, keep our rocm torch)"
"$UV" pip install --python "$VENV/bin/python" --index-url "$USTC" "sglang==0.5.15" --no-deps

echo ">>> verify"
"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "| hip", torch.version.hip, "| GPUs", torch.cuda.device_count())
try:
    import sglang; print("sglang", sglang.__version__)
except Exception as e:
    print("sglang import note:", str(e)[:80])
PY
echo ">>> done. activate with:  source $VENV/bin/activate"
echo ">>> run baseline:         export HIP_VISIBLE_DEVICES=0; python bench_AMD_GLM5_ops_prefill.py"
