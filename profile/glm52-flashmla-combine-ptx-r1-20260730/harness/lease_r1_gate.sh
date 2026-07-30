#!/usr/bin/env bash
# One GPU lease: revalidate the instrument, then run the mandatory graph gate for
# the combine identity control and the combine candidate.
#
# Everything is in one lease so the null control, the identity denominator and
# the candidate are all measured on the same physical GPU under the same clocks.
#
# Order follows plan section 6: CUDA Graph containing region and CUDA Graph leaf
# first, reject before wider profiling. The identity arm is measured against
# installed stock as well, because that ratio must reproduce round-2's P1 number
# (~1.06) -- if it does not, this provider is not a faithful P1 carrier and no
# candidate number from it means anything.
#
# K=20 replays per observation is the round-2 instrument, validated on a
# stock-versus-stock null to +/-0.28 percent. The 1.03 gate is unchanged.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${KERNEL_HARNESS_PYTHON:?}"
export SGLANG_ROOT=/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/flashmla-sparse-decode/sglang

export GLM52_FLASHMLA_COMBINE_VARIANT=combine_identity
"$PY" harness/preflight.py --output evidence/preflight.json

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

for variant in combine_identity combine_c1_stage8; do
  export GLM52_FLASHMLA_COMBINE_VARIANT="$variant"
  for m in 16 32; do
    echo "### ${variant} m=${m} graph gate vs installed stock (containing + leaf) ###"
    "$PY" harness/measure_paired.py \
      --m "$m" \
      --comparison stock_provider \
      --lanes graph_gate \
      --series 3 \
      --pairs 100 \
      --replays-per-observation 20 \
      --output "evidence/${variant}_graph_gate_k20_m${m}.json"
  done
done
