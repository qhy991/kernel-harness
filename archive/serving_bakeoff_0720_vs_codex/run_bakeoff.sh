#!/usr/bin/env bash
# Run serving-native bake-off cases from registry.yaml.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGISTRY="${HERE}/registry.yaml"
RESULTS="${HERE}/results"
PY="${KERNEL_HARNESS_PYTHON:-${ROOT}/.venv/bin/python}"
RUN_SH="${ROOT}/serving_native/run.sh"
FLEX_GPU="${GLM52_FLEX_GPU:-/home/qinhaiyan/glm52-goal-runs/with_flexible_gpu.sh}"

mkdir -p "${RESULTS}"

ONLY_ID="${1:-}"

mapfile -t CASE_IDS < <(
  "${PY}" - <<'PY' "${REGISTRY}" "${ONLY_ID}"
import sys
from pathlib import Path
import yaml

reg = yaml.safe_load(Path(sys.argv[1]).read_text())
only = sys.argv[2] or ""
for case in reg["cases"]:
    if case.get("status") != "run":
        continue
    if only and case["id"] != only:
        continue
    print(case["id"])
PY
)

if [[ ${#CASE_IDS[@]} -eq 0 ]]; then
  echo "no runnable cases selected" >&2
  exit 1
fi

echo "bakeoff: ${#CASE_IDS[@]} runnable case(s)" >&2

run_one() {
  local id="$1"
  local meta
  meta="$("${PY}" - <<'PY' "${REGISTRY}" "${id}"
import json, sys, yaml
from pathlib import Path
reg = yaml.safe_load(Path(sys.argv[1]).read_text())
cid = sys.argv[2]
for case in reg["cases"]:
    if case["id"] == cid:
        print(json.dumps({
            "workload": case["workload"],
            "candidate": case["candidate"],
            "warmup": reg.get("warmup", 5),
            "repeat": reg.get("repeat", 40),
            "sglang_root": case.get("sglang_root") or "",
            "deepgemm_manifest": case.get("deepgemm_manifest") or "",
            "deepgemm_variant": case.get("deepgemm_variant") or "",
        }))
        break
else:
    raise SystemExit(f"unknown case id: {cid}")
PY
)"
  local workload candidate warmup repeat out_json err_log sglang_root deepgemm_manifest deepgemm_variant
  workload="$("${PY}" -c 'import json,sys; print(json.load(sys.stdin)["workload"])' <<<"${meta}")"
  candidate="$("${PY}" -c 'import json,sys; print(json.load(sys.stdin)["candidate"])' <<<"${meta}")"
  warmup="$("${PY}" -c 'import json,sys; print(json.load(sys.stdin)["warmup"])' <<<"${meta}")"
  repeat="$("${PY}" -c 'import json,sys; print(json.load(sys.stdin)["repeat"])' <<<"${meta}")"
  sglang_root="$("${PY}" -c 'import json,sys; print(json.load(sys.stdin)["sglang_root"])' <<<"${meta}")"
  deepgemm_manifest="$("${PY}" -c 'import json,sys; print(json.load(sys.stdin)["deepgemm_manifest"])' <<<"${meta}")"
  deepgemm_variant="$("${PY}" -c 'import json,sys; print(json.load(sys.stdin)["deepgemm_variant"])' <<<"${meta}")"
  out_json="${RESULTS}/${id}.json"
  err_log="${RESULTS}/${id}.log"

  echo "==== ${id} :: ${workload} ====" >&2
  local cmd=(
    "${RUN_SH}" "${workload}"
    --candidate "${ROOT}/${candidate}"
    --warmup "${warmup}"
    --repeat "${repeat}"
    --output "${out_json}"
  )

  # Per-case overlay env (DeepGEMM forks live in goal worktrees).
  # run.sh re-exports PYTHONPATH from SGLANG_ROOT.
  local -a env_args=()
  if [[ -n "${sglang_root}" ]]; then
    env_args+=("SGLANG_ROOT=${sglang_root}")
  fi
  if [[ -n "${deepgemm_manifest}" ]]; then
    env_args+=("SGLANG_GLM52_DEEPGEMM_MANIFEST=${deepgemm_manifest}")
  fi
  if [[ -n "${deepgemm_variant}" ]]; then
    env_args+=("SGLANG_GLM52_DEEPGEMM_VARIANT=${deepgemm_variant}")
  fi
  if [[ ${#env_args[@]} -gt 0 ]]; then
    cmd=(env "${env_args[@]}" "${cmd[@]}")
  fi

  set +e
  if [[ -x "${FLEX_GPU}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    "${FLEX_GPU}" -- "${cmd[@]}" >"${err_log}" 2>&1
  else
    "${cmd[@]}" >"${err_log}" 2>&1
  fi
  local rc=$?
  set -e

  if [[ ${rc} -ne 0 ]]; then
    echo "FAIL ${id} rc=${rc} (see ${err_log})" >&2
    "${PY}" - <<PY
import json
from pathlib import Path
Path("${out_json}").write_text(json.dumps({
    "schema_version": 1,
    "bakeoff_error": True,
    "case_id": "${id}",
    "exit_code": ${rc},
    "log": Path("${err_log}").read_text(errors="replace")[-8000:],
}, indent=2) + "\n")
PY
  else
    echo "OK ${id} -> ${out_json}" >&2
  fi
}

for id in "${CASE_IDS[@]}"; do
  run_one "${id}"
done

echo "summarizing..." >&2
"${PY}" "${HERE}/summarize_bakeoff.py"
echo "done: ${RESULTS}/bakeoff_summary.csv ${HERE}/BAKEOFF.md" >&2
