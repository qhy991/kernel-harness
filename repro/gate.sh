#!/usr/bin/env bash
# Portable op-level A/B runner for the four GLM-5.2 AMD winners.
#
# Each committed winner lives at testbench/tasks/glm52_amd/<op>/candidate.py. That task's
# own run.sh A/Bs the candidate against the reference oracle and persists an auditable
# result.json under runs/glm52/<op>/<run_id>/. This wrapper just loads the environment
# and drives all four ops; verdicts come from run.sh's exit code (see legend below).
#
# Usage (repro/runenv.sh is sourced automatically if the env isn't loaded yet):
#   repro/gate.sh smoke [TASK]     # 1 iteration, fast sanity check (default op: index_score_prefill)
#   repro/gate.sh gate  [TASK]     # authoritative: --repeat 10 --iterations 30 --warmup 3
#                                  #   TASK omitted -> all four ops in sequence
#
# Then audit any persisted run:
#   "$ROCM_TORCH_VENV/bin/python" testbench/bin/audit_result.py runs/glm52/<op>/<run_id>/result.json
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

# Load the env once (idempotent — skip if already sourced).
if [ -z "${KERNEL_HARNESS_PROVIDER:-}" ] || [ -z "${ROCM_TORCH_VENV:-}" ]; then
  # shellcheck source=repro/runenv.sh
  source "$HERE/runenv.sh" || { echo "[gate] could not load runenv.sh" >&2; exit 1; }
fi

OPS=(index_score_prefill moe_total_prefill moe_total_decode dsa_prefill_attn)
mode="${1:-}"; shift || true
task="${1:-}"

run_task () {  # <op> <extra run.sh args...>
  local op="$1"; shift
  local sh="testbench/tasks/glm52_amd/$op/run.sh"
  [ -x "$sh" ] || { echo "[gate] missing runner: $sh" >&2; return 3; }
  echo "===== $op ($mode) ====="
  "$sh" "$@"; local rc=$?
  echo "[gate] $op -> exit=$rc  (0 correct+faster · 1 correct+not-faster · 2 incorrect · 3 infra/contract)"
  return $rc
}

case "$mode" in
  smoke)
    run_task "${task:-index_score_prefill}" --repeat 1 --iterations 1 --warmup 0 ;;
  gate)
    rc=0
    if [ -n "$task" ]; then
      run_task "$task" --repeat 10 --iterations 30 --warmup 3 || rc=$?
    else
      for op in "${OPS[@]}"; do run_task "$op" --repeat 10 --iterations 30 --warmup 3 || rc=$?; done
    fi
    exit $rc ;;
  *)
    echo "usage: repro/gate.sh smoke|gate [TASK]" >&2
    echo "  TASK in: ${OPS[*]}" >&2
    exit 2 ;;
esac
