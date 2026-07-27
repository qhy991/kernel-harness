#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT=/home/qinhaiyan/glm52-goal-runs/16-indexer_score_decode/kernel-harness
SGLANG_TREE=/home/qinhaiyan/glm52-goal-runs/16-indexer_score_decode/sglang
TASK_PY="$HARNESS_ROOT/.venv/bin/python"
DRIVER="$HARNESS_ROOT/evidence/glm52_prod_16_indexer_score_decode/profile_driver.py"
SMOKE_DIR="$(mktemp -d)"

export SGLANG_ROOT="$SGLANG_TREE"
export PYTHONPATH="$SGLANG_TREE/python:$HARNESS_ROOT:${PYTHONPATH:-}"

nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  -o "$SMOKE_DIR/nsys-smoke" \
  "$TASK_PY" "$DRIVER" --batch 16 --backend deepgemm

ncu \
  --set basic \
  --nvtx \
  --nvtx-include 'indexer_score_topk_deepgemm_m16/' \
  --launch-count 1 \
  --force-overwrite \
  -o "$SMOKE_DIR/ncu-smoke" \
  "$TASK_PY" "$DRIVER" --batch 16 --backend deepgemm

test -s "$SMOKE_DIR/nsys-smoke.nsys-rep"
test -s "$SMOKE_DIR/ncu-smoke.ncu-rep"
ncu --import "$SMOKE_DIR/ncu-smoke.ncu-rep" --page details
printf 'profiler_smoke_dir=%s\n' "$SMOKE_DIR"
