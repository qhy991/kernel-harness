#!/usr/bin/env bash
# One GPU lease: preflight plus complete correctness for one variant.
#
# SGLANG_GLM52_FLASHMLA_GRAPH_ONLY=0 is deliberate here. Round-3 selects the
# provider only under graph capture, so with the production default the eager
# containing region would fall back to stock and the eager containing
# correctness comparison would silently degrade into stock-versus-stock. The
# kernel must be validated everywhere it can execute, so correctness forces
# selection on every boundary. Performance runs use the production default,
# where the eager containing lane is required to be a stock fallback.
set -euo pipefail

VARIANT="${1:?usage: lease_correctness.sh VARIANT [--with-preflight]}"
WITH_PREFLIGHT="${2:-}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${KERNEL_HARNESS_PYTHON:?}"
export SGLANG_ROOT=/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/flashmla-sparse-decode/sglang

export GLM52_FLASHMLA_VARIANT="$VARIANT"
export SGLANG_GLM52_FLASHMLA_GRAPH_ONLY=0

if [[ "$WITH_PREFLIGHT" == "--with-preflight" ]]; then
  "$PY" harness/preflight.py --output evidence/preflight.json
fi

for m in 16 32; do
  echo "### ${VARIANT} m=${m} boundary correctness ###"
  "$PY" harness/validate_a0.py --m "$m" \
    --output "evidence/${VARIANT}_correctness_m${m}.json"
  echo "### ${VARIANT} m=${m} adversarial matrix ###"
  "$PY" harness/validate_matrix.py --m "$m" \
    --output "evidence/${VARIANT}_matrix_m${m}.json"
done
