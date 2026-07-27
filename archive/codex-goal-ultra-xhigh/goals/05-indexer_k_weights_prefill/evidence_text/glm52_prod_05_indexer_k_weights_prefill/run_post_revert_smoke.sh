#!/usr/bin/env bash
# Final stock-ABI smoke test. Invoke only through with_flexible_gpu.sh.
set -euo pipefail

KH_ROOT=/home/qinhaiyan/glm52-goal-runs/05-indexer_k_weights_prefill/kernel-harness
SGLANG_ROOT=/home/qinhaiyan/glm52-goal-runs/05-indexer_k_weights_prefill/sglang
EVIDENCE="$KH_ROOT/evidence/glm52_prod_05_indexer_k_weights_prefill"
PY="$KH_ROOT/.venv/bin/python"

export SGLANG_ROOT
export PYTHONPATH="$SGLANG_ROOT/python:$KH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_GLM52_OPT=0
unset SGLANG_DISABLE_DSA_INDEXER_FUSION

{
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  printf 'kh_sha=%s\n' "$(git -C "$KH_ROOT" rev-parse HEAD)"
  printf 'sglang_sha=%s\n' "$(git -C "$SGLANG_ROOT" rev-parse HEAD)"
  printf 'dsa_indexer_sha256='
  sha256sum "$SGLANG_ROOT/python/sglang/srt/layers/attention/dsa/dsa_indexer.py"
  nvidia-smi --query-gpu=index,uuid,name,pstate,clocks.current.sm,clocks.current.memory,power.draw \
    --format=csv
} > "$EVIDENCE/post_revert_environment.txt" 2>&1

"$PY" "$KH_ROOT/testbench/bin/check_env.py" \
  > "$EVIDENCE/post_revert_check_env.txt" 2>&1

"$KH_ROOT/serving_native/run.sh" indexer_wk_weights_prefill_m4096 \
  --warmup 3 --repeat 10 --output "$EVIDENCE/post_revert_isolated_stock.json" \
  > "$EVIDENCE/post_revert_isolated_stock.log" 2>&1

"$KH_ROOT/serving_native/run.sh" \
  indexer_fused_prepare_store_prefill_m4096_eager_dual_stream \
  --warmup 3 --repeat 10 --output "$EVIDENCE/post_revert_region_stock.json" \
  > "$EVIDENCE/post_revert_region_stock.log" 2>&1
