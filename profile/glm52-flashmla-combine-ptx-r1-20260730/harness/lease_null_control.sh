#!/usr/bin/env bash
# One GPU lease: revalidate the instrument on this round's host and GPUs.
#
# Every round-3 conclusion is a ratio compared against 1.03, and the
# R3-A-versus-P1 comparison is a claim that a difference is *absent*, so the
# instrument's own same-binary null spread bounds what may be concluded. Both
# arms are the identical installed stock binary; anything the null resolves is
# instrument noise, not signal.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${KERNEL_HARNESS_PYTHON:?}"
export SGLANG_ROOT=/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/flashmla-sparse-decode/sglang
export GLM52_FLASHMLA_COMBINE_VARIANT=combine_identity

for m in 16 32; do
  echo "### stock-vs-stock null, K=20, m=${m} ###"
  "$PY" harness/measure_paired.py \
    --m "$m" \
    --comparison stock_stock \
    --lanes graph_gate \
    --series 3 \
    --pairs 100 \
    --replays-per-observation 20 \
    --output "evidence/null_stock_stock_k20_m${m}.json"
done

echo "### current API-v1 dispatch stage cost (graph-only early return) ###"
"$PY" harness/time_dispatch_stages.py \
  --output evidence/dispatch_stage_times_graph_only.json || true
