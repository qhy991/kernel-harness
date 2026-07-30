#!/usr/bin/env bash
# One GPU lease: remaining evidence for the selected candidate.
#
#   1. the round-3 graph-only selection contract (eager must be a stock
#      fallback, capture must select the candidate);
#   2. the eager lanes, recorded as diagnostics -- round-3 does not gate on
#      them, and the eager containing arm is expected to be stock;
#   3. the Nsys device chain proving one prefixed main followed by one
#      unchanged stock combine in eager and in each graph replay.
set -euo pipefail

VARIANT="${1:?usage: lease_evidence.sh VARIANT}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${KERNEL_HARNESS_PYTHON:?}"
export SGLANG_ROOT=/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/flashmla-sparse-decode/sglang
export GLM52_FLASHMLA_COMBINE_VARIANT="$VARIANT"

SYMBOL="infini_kernel_glm52_flashmla_sparse_decode_${VARIANT}_main"

for m in 16 32; do
  echo "### ${VARIANT} m=${m} graph-only selection contract ###"
  "$PY" harness/probe_graph_only.py --m "$m" \
    --output "evidence/${VARIANT}_graph_only_contract_m${m}.json"
done

for m in 16 32; do
  echo "### ${VARIANT} m=${m} eager lanes (diagnostic) ###"
  "$PY" harness/measure_paired.py \
    --m "$m" \
    --comparison stock_provider \
    --lanes eager \
    --series 3 \
    --pairs 100 \
    --replays-per-observation 20 \
    --output "evidence/${VARIANT}_eager_k20_m${m}.json"
done

for m in 16 32; do
  echo "### ${VARIANT} m=${m} nsys device chain ###"
  rep="evidence/${VARIANT}_chain_m${m}"
  rm -f "${rep}.nsys-rep" "${rep}.sqlite"
  nsys profile \
    --trace=cuda,nvtx \
    --cuda-graph-trace=node \
    --force-overwrite=true \
    --output "$rep" \
    "$PY" harness/trace_chain.py --m "$m" --graph-replays 5 \
    >"${rep}.trace_stdout.log" 2>&1
  "$PY" harness/summarize_nsys.py \
    --m "$m" \
    --report "${rep}.nsys-rep" \
    --candidate-main-symbol "$SYMBOL" \
    --output "evidence/${VARIANT}_chain_m${m}_summary.json"
done
