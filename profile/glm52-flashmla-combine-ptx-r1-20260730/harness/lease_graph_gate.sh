#!/usr/bin/env bash
# One GPU lease: the round-3 mandatory graph gate for one variant.
#
# Plan order runs the CUDA Graph containing region and the CUDA Graph leaf
# first, and rejects before any wider profiling if an estimator misses 1.03.
# Both buckets stay inside this single lease so every AB/BA series and its
# paired stock arm are measured on the same physical GPU.
#
# The production graph-only default is left in place: the selection probe
# recorded by measure_paired.py must show zero provider launches in the eager
# containing region and non-zero under graph replay.
#
# K=20 replays per observation is the round-2 instrument, validated on a
# stock-versus-stock null to +/-0.28 percent. The 1.03 gate is unchanged.
set -euo pipefail

VARIANT="${1:?usage: lease_graph_gate.sh VARIANT}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${KERNEL_HARNESS_PYTHON:?}"
export SGLANG_ROOT=/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/flashmla-sparse-decode/sglang

export GLM52_FLASHMLA_COMBINE_VARIANT="$VARIANT"

for m in 16 32; do
  echo "### ${VARIANT} m=${m} graph gate (containing + leaf) ###"
  "$PY" harness/measure_paired.py \
    --m "$m" \
    --comparison stock_provider \
    --lanes graph_gate \
    --series 3 \
    --pairs 100 \
    --replays-per-observation 20 \
    --output "evidence/${VARIANT}_graph_gate_k20_m${m}.json"
done
