#!/usr/bin/env bash
# One GPU lease, one physical GPU: the complete final decision dataset.
#
# Why this exists. The first two leases were served different physical B200s, and
# the installed-stock arm's absolute leaf time differs about 7 percent between
# them (M16 leaf 28.86 us on one, 26.85 us on the other) while the candidate arm
# is unchanged. The shared contract forbids comparing unpaired runs from
# different GPUs, so every ratio that will be quoted is regenerated here inside a
# single lease from the final committed binaries.
#
# The asymmetry itself is disclosed rather than hidden: the provider arm reduces
# into workspaces preallocated once by initialize(), while the installed stock
# path takes its o_accum from the caching allocator, so the stock arm's buffer
# placement is a per-process nuisance the candidate arm does not have. That is
# exactly why the P1-relative comparison below is a directly paired
# provider-versus-provider measurement: both arms then use identically shaped
# workspaces allocated by the same code in the same process, so the placement
# term cancels instead of being argued about.
#
# Order: the two same-binary nulls first (they bound what anything else can
# resolve), then the stock-relative graph gate for every arm, then the paired
# P1-relative graph gate for every candidate.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${KERNEL_HARNESS_PYTHON:?}"
export SGLANG_ROOT=/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/flashmla-sparse-decode/sglang

CANDIDATES=(combine_c1_stage8 combine_c2_bucket_stages)

echo "### same-binary null 1 of 2: installed stock in both arms ###"
export GLM52_FLASHMLA_COMBINE_VARIANT=combine_identity
for m in 16 32; do
  "$PY" harness/measure_paired.py --m "$m" --comparison stock_stock --lanes graph_gate \
    --series 3 --pairs 100 --replays-per-observation 20 \
    --output "evidence/final_null_stock_stock_k20_m${m}.json"
done

echo "### same-binary null 2 of 2: combine_identity in both provider arms ###"
for m in 16 32; do
  "$PY" harness/measure_provider_pair.py --m "$m" \
    --a-variant combine_identity --b-variant combine_identity \
    --series 3 --pairs 100 --replays-per-observation 20 \
    --output "evidence/final_pair_null_identity_identity_k20_m${m}.json"
done

echo "### stock-relative graph gate, every arm, same GPU, final binaries ###"
for variant in combine_identity "${CANDIDATES[@]}"; do
  export GLM52_FLASHMLA_COMBINE_VARIANT="$variant"
  for m in 16 32; do
    "$PY" harness/measure_paired.py --m "$m" --comparison stock_provider --lanes graph_gate \
      --series 3 --pairs 100 --replays-per-observation 20 \
      --output "evidence/final_${variant}_graph_gate_k20_m${m}.json"
  done
done

echo "### P1-relative paired graph gate: identity versus each candidate ###"
for variant in "${CANDIDATES[@]}"; do
  for m in 16 32; do
    "$PY" harness/measure_provider_pair.py --m "$m" \
      --a-variant combine_identity --b-variant "$variant" \
      --series 3 --pairs 100 --replays-per-observation 20 \
      --output "evidence/final_pair_identity_vs_${variant}_k20_m${m}.json"
  done
done
