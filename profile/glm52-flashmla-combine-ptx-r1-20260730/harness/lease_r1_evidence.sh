#!/usr/bin/env bash
# One GPU lease: graph-only selection contract, diagnostic eager lanes, and the
# Nsys device chain for one combine variant.
#
# The Nsys capture is what separates two very different explanations of a flat
# chain measurement: a combine kernel that did not get faster, versus a combine
# kernel that got faster but was already hidden behind the main kernel by PDL.
# It reports per-kernel durations for the stock arm and the candidate arm in the
# same report, so main, combine and overlap are all directly observable.
set -euo pipefail

VARIANT="${1:?usage: lease_r1_evidence.sh VARIANT [--skip-eager]}"
SKIP_EAGER="${2:-}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${KERNEL_HARNESS_PYTHON:?}"
export SGLANG_ROOT=/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/flashmla-sparse-decode/sglang
export GLM52_FLASHMLA_COMBINE_VARIANT="$VARIANT"

# The combine campaign pins the main kernel to the round-2 survivor P1 and
# replaces only the second kernel of the chain.
MAIN_SYMBOL="infini_kernel_glm52_flashmla_sparse_decode_p1_consumer_scale_main"
COMBINE_SYMBOL="infini_kernel_glm52_flashmla_sparse_decode_${VARIANT}"

for m in 16 32; do
  echo "### ${VARIANT} m=${m} graph-only selection contract ###"
  "$PY" harness/probe_graph_only.py --m "$m" \
    --output "evidence/${VARIANT}_graph_only_contract_m${m}.json"
done

if [[ "$SKIP_EAGER" != "--skip-eager" ]]; then
  for m in 16 32; do
    echo "### ${VARIANT} m=${m} eager lanes (diagnostic under graph-only) ###"
    "$PY" harness/measure_paired.py \
      --m "$m" \
      --comparison stock_provider \
      --lanes eager \
      --series 3 \
      --pairs 100 \
      --replays-per-observation 20 \
      --output "evidence/${VARIANT}_eager_k20_m${m}.json"
  done
fi

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
    --candidate-main-symbol "$MAIN_SYMBOL" \
    --candidate-combine-symbol "$COMBINE_SYMBOL" \
    --output "evidence/${VARIANT}_chain_m${m}_summary.json"
done
