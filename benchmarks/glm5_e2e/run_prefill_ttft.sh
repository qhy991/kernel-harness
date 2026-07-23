#!/usr/bin/env bash
# GLM-5.2 prefill TTFT — thin wrapper around `python -m sglang.bench_one_batch`.
# Matches llm_flops's prefill scenario:
#   input_len ∈ {1024, 2048, 4096}, KV cache = 65536, output_len = 1, TP = 8
#
# The heavy lifting is `sglang.bench_one_batch` (which natively supports
# --input-len 1024 2048 4096 as a sweep; each shape gets its own RESULT_JSON
# line in --result-filename). We only:
#   · point PYTHONPATH at benchmarks/glm5_e2e/shim so sitecustomize.py loads
#     the gfx942 compat shim + any $KDA_E2E_OVERRIDES before sglang boots
#   · pin the sglang args to the llm_flops scenario
#   · forward everything else to bench_one_batch
#
# Usage:
#   ./run_prefill_ttft.sh                                              # baseline
#   ./run_prefill_ttft.sh --overrides my_kernels.py                    # your patches
#   ./run_prefill_ttft.sh --input-len 4096                             # single shape
#   ./run_prefill_ttft.sh -- --profile --profile-stage prefill         # extra sglang flags
#     (everything after `--` is spliced verbatim into sglang argv)

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

# Parse our few flags; forward the rest.
INPUT_LEN=(1024 2048 4096)
OVERRIDES=""
OUT_ROOT="${KDA_E2E_OUT:-/tmp/glm5_e2e}"
MODEL_PATH="${KDA_E2E_MODEL:-/mnt/public/qinhaiyan/models/GLM-5.2-FP8}"
TP="${KDA_E2E_TP:-8}"
KV_TOKENS="${KDA_E2E_KV_TOKENS:-65536}"
MEM_FRAC="${KDA_E2E_MEM_FRAC:-0.95}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --overrides)   OVERRIDES="$2"; shift 2;;
    --input-len)   INPUT_LEN=("$2"); shift 2;;
    --model-path)  MODEL_PATH="$2"; shift 2;;
    --tp)          TP="$2"; shift 2;;
    --kv-tokens)   KV_TOKENS="$2"; shift 2;;
    --out-root)    OUT_ROOT="$2"; shift 2;;
    --)            shift; EXTRA_ARGS+=("$@"); break;;
    *)             EXTRA_ARGS+=("$1"); shift;;
  esac
done

STAMP="$(date -u +%Y%m%d_%H%M%SZ)"
RUN_DIR="$OUT_ROOT/prefill_ttft-$STAMP"
mkdir -p "$RUN_DIR"
RESULT_FILE="$RUN_DIR/results.jsonl"

# Load-order plumbing:
#   · $HERE/shim on PYTHONPATH → sitecustomize.py runs on Python start,
#     which imports glm52_gfx942_shim (gfx942 patches) and, if
#     KDA_E2E_OVERRIDES is set, imports the user file and calls register().
#   · $HERE also on PYTHONPATH so `import operator_overrides` works from
#     inside the user's overrides.py.
export PYTHONPATH="$HERE/shim:$HERE:${PYTHONPATH:-}"
if [[ -n "$OVERRIDES" ]]; then
  export KDA_E2E_OVERRIDES="$(realpath "$OVERRIDES")"
fi

# gfx942 env every sglang boot needs; the shim also sets these but exporting
# here guarantees they're in place before Python even parses sitecustomize.
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx942}"
export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
export SGLANG_DSA_FUSE_TOPK="${SGLANG_DSA_FUSE_TOPK:-0}"
export SGLANG_OPT_USE_AITER_SILU_MUL="${SGLANG_OPT_USE_AITER_SILU_MUL:-1}"
export SGLANG_DISABLE_GFX942_BPRESHUFFLE="${SGLANG_DISABLE_GFX942_BPRESHUFFLE:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export HSA_ENABLE_COREDUMP_ON_EXCEPTION="${HSA_ENABLE_COREDUMP_ON_EXCEPTION:-0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

ROCM_TORCH_VENV="${ROCM_TORCH_VENV:-/root/venvs/rocm-torch}"
PYTHON="${ROCM_TORCH_PYTHON:-$ROCM_TORCH_VENV/bin/python}"
ulimit -c 0

# manifest for provenance
python3 - <<PY > "$RUN_DIR/manifest.json"
import json, os, socket, subprocess, time
print(json.dumps({
    "scenario": "prefill_ttft",
    "started_at": "$STAMP",
    "host": socket.gethostname(),
    "model_path": "$MODEL_PATH",
    "tp": $TP,
    "kv_tokens": $KV_TOKENS,
    "input_lens": [${INPUT_LEN[*]// /, }],
    "output_len": 1,
    "overrides": os.environ.get("KDA_E2E_OVERRIDES") or None,
    "backends": {"dsa_prefill":"aiter","dsa_decode":"fa3","dsa_topk":"torch","moe":"triton"},
    "sglang_entry": "python -m sglang.bench_one_batch",
    "env": {k: os.environ[k] for k in sorted(os.environ)
            if k.startswith(("SGLANG_","KDA_","HIP_","PYTORCH_"))},
}, indent=2))
PY

echo "═══ GLM-5.2 prefill TTFT sweep ═══"
echo "  input_len = ${INPUT_LEN[*]},  KV = $KV_TOKENS,  TP = $TP"
echo "  overrides = ${OVERRIDES:-(none — vanilla sglang)}"
echo "  results   = $RESULT_FILE"
echo "═══════════════════════════════════"

exec "$PYTHON" -m sglang.bench_one_batch \
  --model-path "$MODEL_PATH" \
  --tp "$TP" \
  --batch-size 1 \
  --input-len "${INPUT_LEN[@]}" \
  --output-len 1 \
  --trust-remote-code \
  --mem-fraction-static "$MEM_FRAC" \
  --max-total-tokens "$KV_TOKENS" \
  --dsa-topk-backend torch \
  --dsa-prefill-backend aiter \
  --dsa-decode-backend fa3 \
  --moe-runner-backend triton \
  --cuda-graph-max-bs 1 \
  --disable-cuda-graph \
  --run-name "prefill_ttft-$STAMP" \
  --result-filename "$RESULT_FILE" \
  "${EXTRA_ARGS[@]}"
