#!/usr/bin/env bash
set -euo pipefail

KH=/home/qinhaiyan/glm52-goal-runs/22-dsa_flashmla_kv_production/kernel-harness
SG=/home/qinhaiyan/glm52-goal-runs/22-dsa_flashmla_kv_production/sglang
BASE="$KH/profile/dsa-flashmla-kv-stock-b200-20260722"
CAMPAIGN_ID=${1:-flex-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ ! "$CAMPAIGN_ID" =~ ^flex-[0-9]{8}T[0-9]{6}Z[a-z]?$ ]]; then
  echo "campaign id must be flex-YYYYMMDDTHHMMSSZ" >&2
  exit 64
fi
OUT="$BASE/campaigns/$CAMPAIGN_ID"
mkdir -p "$OUT"

set +e
set -o pipefail
/home/qinhaiyan/glm52-goal-runs/with_flexible_gpu.sh -- \
  env SGLANG_ROOT="$SG" KERNEL_HARNESS_PYTHON="$KH/.venv/bin/python" \
  "$BASE/harness/run_flexible_campaign.sh" "$CAMPAIGN_ID" \
  2>&1 | tee "$OUT/wrapper.log"
status=${PIPESTATUS[0]}
set -e
if [[ "$status" == 75 ]]; then
  echo "GPU scheduler busy; preserve the log, continue CPU-only work, and retry later." >&2
fi
exit "$status"
