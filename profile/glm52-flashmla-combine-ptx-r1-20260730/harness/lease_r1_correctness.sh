#!/usr/bin/env bash
# One GPU lease: complete correctness for one combine variant at both buckets.
#
# SGLANG_GLM52_FLASHMLA_GRAPH_ONLY=0 is deliberate. Production selection is
# graph-only, so with the default the eager boundaries would fall back to stock
# and the eager correctness comparison would silently become stock-versus-stock.
# The kernel must be validated everywhere it can execute, so correctness forces
# selection on every boundary; performance runs use the production default.
set -euo pipefail

VARIANT="${1:?usage: lease_r1_correctness.sh VARIANT}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${KERNEL_HARNESS_PYTHON:?}"
export SGLANG_ROOT=/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/flashmla-sparse-decode/sglang
export GLM52_FLASHMLA_COMBINE_VARIANT="$VARIANT"
export SGLANG_GLM52_FLASHMLA_GRAPH_ONLY=0

for m in 16 32; do
  echo "### ${VARIANT} m=${m} boundary correctness ###"
  "$PY" harness/validate_a0.py --m "$m" \
    --output "evidence/${VARIANT}_correctness_m${m}.json"
  echo "### ${VARIANT} m=${m} adversarial matrix ###"
  "$PY" harness/validate_matrix.py --m "$m" \
    --output "evidence/${VARIANT}_matrix_m${m}.json"
done
