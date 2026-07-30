#!/usr/bin/env bash
# One GPU lease: the directly paired P1-relative denominator.
#
# Plan hypothesis 1 requires the combine candidate to improve on the current
# P1-plus-stock-combine chain, not merely on ancient installed stock. This lease
# measures that as a single alternating AB/BA comparison with the combine
# identity control in the A arm, plus an identity-versus-identity null so the
# instrument's own spread on this harness is known before any ratio is read.
set -euo pipefail

CANDIDATE="${1:?usage: lease_r1_pair.sh CANDIDATE_VARIANT}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${KERNEL_HARNESS_PYTHON:?}"
export SGLANG_ROOT=/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/flashmla-sparse-decode/sglang

for m in 16 32; do
  echo "### provider-pair null: combine_identity vs combine_identity, m=${m} ###"
  "$PY" harness/measure_provider_pair.py \
    --m "$m" \
    --a-variant combine_identity \
    --b-variant combine_identity \
    --series 3 --pairs 100 --replays-per-observation 20 \
    --output "evidence/pair_null_identity_identity_k20_m${m}.json"
done

for m in 16 32; do
  echo "### provider-pair: combine_identity vs ${CANDIDATE}, m=${m} ###"
  "$PY" harness/measure_provider_pair.py \
    --m "$m" \
    --a-variant combine_identity \
    --b-variant "$CANDIDATE" \
    --series 3 --pairs 100 --replays-per-observation 20 \
    --output "evidence/pair_identity_vs_${CANDIDATE}_k20_m${m}.json"
done
