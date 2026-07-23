#!/usr/bin/env bash
# GLM-5.2 decode throughput — thin wrapper around `python -m sglang.bench_one_batch`.
# Matches the deployment scenario:
#   batch_size ∈ {128, 256}, TP = 8, small prefill (--decode-input-len)
#     so the reported decode number is not TTFT-dominated.
#
# `sglang.bench_one_batch` natively sweeps `--batch-size 128 256` and reports
# median decode latency + throughput per shape in the --result-filename JSONL.
#
# Usage:
#   ./run_decode_throughput.sh                          # baseline (bs = 128, 256)
#   ./run_decode_throughput.sh --overrides my_ov.py
#   ./run_decode_throughput.sh --batch-size 256         # single shape
#   ./run_decode_throughput.sh -- --profile             # extra sglang flags

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

BATCH_SIZE=(128 256)
OVERRIDES=""
OUT_ROOT="${KDA_E2E_OUT:-/tmp/glm5_e2e}"
MODEL_PATH="${KDA_E2E_MODEL:-/mnt/public/qinhaiyan/models/GLM-5.2-FP8}"
TP="${KDA_E2E_TP:-8}"
KV_TOKENS="${KDA_E2E_KV_TOKENS:-65536}"
MEM_FRAC="${KDA_E2E_MEM_FRAC:-0.95}"
INPUT_LEN="${KDA_E2E_DECODE_INPUT_LEN:-128}"
OUTPUT_LEN="${KDA_E2E_DECODE_OUTPUT_LEN:-64}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --overrides)    OVERRIDES="$2"; shift 2;;
    --batch-size)   read -r -a BATCH_SIZE <<< "$2"; shift 2;;
    --model-path)   MODEL_PATH="$2"; shift 2;;
    --tp)           TP="$2"; shift 2;;
    --kv-tokens)    KV_TOKENS="$2"; shift 2;;
    --input-len)    INPUT_LEN="$2"; shift 2;;
    --output-len)   OUTPUT_LEN="$2"; shift 2;;
    --out-root)     OUT_ROOT="$2"; shift 2;;
    --)             shift; EXTRA_ARGS+=("$@"); break;;
    *)              EXTRA_ARGS+=("$1"); shift;;
  esac
done

STAMP="$(date -u +%Y%m%d_%H%M%SZ)"
RUN_DIR="$OUT_ROOT/decode_thpt-$STAMP"
mkdir -p "$RUN_DIR"
RESULT_FILE="$RUN_DIR/results.jsonl"

export PYTHONPATH="$HERE/shim:$HERE:${PYTHONPATH:-}"
if [[ -n "$OVERRIDES" ]]; then
  export KDA_E2E_OVERRIDES="$(realpath "$OVERRIDES")"
fi

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

# For each batch size, we need max_total_tokens ≥ batch × (input+output).
# Take max of user's kv_tokens and what the biggest bs needs.
MAX_BS=$(printf "%s\n" "${BATCH_SIZE[@]}" | sort -n | tail -1)
BS_NEED=$(( MAX_BS * (INPUT_LEN + OUTPUT_LEN) ))
[[ "$KV_TOKENS" -lt "$BS_NEED" ]] && KV_TOKENS="$BS_NEED"

python3 - <<PY > "$RUN_DIR/manifest.json"
import json, os, socket
bs_str = "${BATCH_SIZE[*]}"
print(json.dumps({
    "scenario": "decode_throughput",
    "started_at": "$STAMP",
    "host": socket.gethostname(),
    "model_path": "$MODEL_PATH",
    "tp": $TP,
    "kv_tokens": $KV_TOKENS,
    "batch_sizes": [int(x) for x in bs_str.split()],
    "input_len": $INPUT_LEN,
    "output_len": $OUTPUT_LEN,
    "overrides": os.environ.get("KDA_E2E_OVERRIDES") or None,
    "backends": {"dsa_prefill":"aiter","dsa_decode":"fa3","dsa_topk":"torch","moe":"triton"},
    "sglang_entry": "python -m sglang.bench_one_batch",
    "env": {k: os.environ[k] for k in sorted(os.environ)
            if k.startswith(("SGLANG_","KDA_","HIP_","PYTORCH_"))},
}, indent=2))
PY

echo "═══ GLM-5.2 decode throughput sweep ═══"
echo "  batch_size = ${BATCH_SIZE[*]},  input_len = $INPUT_LEN,  output_len = $OUTPUT_LEN"
echo "  TP = $TP,  KV = $KV_TOKENS  (auto-raised to fit batch × in+out)"
echo "  overrides = ${OVERRIDES:-(none — vanilla sglang)}"
echo "  results   = $RESULT_FILE"
echo "═══════════════════════════════════════"

exec "$PYTHON" -m sglang.bench_one_batch \
  --model-path "$MODEL_PATH" \
  --tp "$TP" \
  --batch-size "${BATCH_SIZE[@]}" \
  --input-len "$INPUT_LEN" \
  --output-len "$OUTPUT_LEN" \
  --trust-remote-code \
  --mem-fraction-static "$MEM_FRAC" \
  --max-total-tokens "$KV_TOKENS" \
  --dsa-topk-backend torch \
  --dsa-prefill-backend aiter \
  --dsa-decode-backend aiter \
  --moe-runner-backend triton \
  --cuda-graph-max-bs "$MAX_BS" \
  --disable-cuda-graph \
  --run-name "decode_thpt-$STAMP" \
  --result-filename "$RESULT_FILE" \
  "${EXTRA_ARGS[@]}"
