#!/usr/bin/env bash
# GLM-5.2 incremental prefill TTFT — uses sglang's native shared-prefix
# scenario (`bench_serving --dataset-name generated-shared-prefix`),
# which is the sglang-official way to measure "N tokens extended on top
# of an already-prefilled prefix".
#
# Layout:
#   · launch sglang server in background (TP=8, GLM-5.2-FP8)
#   · wait for readiness
#   · run bench_serving with generated-shared-prefix dataset:
#     `gsp-system-prompt-len` = shared 64k prefix (identical across prompts,
#         cached after first hit by sglang's automatic prefix caching)
#     `gsp-question-len` = incremental extend length (1024, 2048, or 4096)
#     `num-prompts` = how many requests to send (each shares the prefix)
#   · bench_serving reports per-request mean/median/p90/p99 TTFT — that's
#     the incremental prefill TTFT for this question length
#   · shutdown server, next question_len
#
# Aligned with llm_flops's stated intent: "kvcache=64k 下，增量输入
# 1024/2048/4096 的 token，测试 TTFT".
#
# Usage:
#   ./run_incremental_prefill_ttft.sh                    # baseline, all 3 question lengths
#   ./run_incremental_prefill_ttft.sh --overrides my.py  # with operator swap
#   ./run_incremental_prefill_ttft.sh --question-len 4096  # single shape
#
# Total wall time: model load ~1 min + per shape ~2-3 min = ~10 min for 3 shapes.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

# ── knobs ────────────────────────────────────────────────────────────
QUESTION_LEN=(1024 2048 4096)
OVERRIDES=""
OUT_ROOT="${KDA_E2E_OUT:-/tmp/glm5_e2e}"
MODEL_PATH="${KDA_E2E_MODEL:-/mnt/public/qinhaiyan/models/GLM-5.2-FP8}"
TP="${KDA_E2E_TP:-8}"
KV_TOKENS="${KDA_E2E_KV_TOKENS:-131072}"                # need 64k prefix + up to 4k question + slack
SYSTEM_PROMPT_LEN="${KDA_E2E_PREFIX_LEN:-65536}"        # 64k prefix per llm_flops scenario
OUTPUT_LEN="${KDA_E2E_OUTPUT_LEN:-1}"                   # TTFT scenario → output len = 1
NUM_PROMPTS="${KDA_E2E_NUM_PROMPTS:-8}"                 # each shares the same prefix; 1st fills, rest are incremental
MEM_FRAC="${KDA_E2E_MEM_FRAC:-0.90}"
PORT="${KDA_E2E_PORT:-30011}"
SERVER_READY_TIMEOUT="${KDA_E2E_SERVER_READY_TIMEOUT:-600}"
EXTRA_SERVER_ARGS=()
EXTRA_BENCH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --overrides)      OVERRIDES="$2"; shift 2;;
    --question-len)   read -r -a QUESTION_LEN <<< "$2"; shift 2;;
    --system-prompt-len) SYSTEM_PROMPT_LEN="$2"; shift 2;;
    --num-prompts)    NUM_PROMPTS="$2"; shift 2;;
    --output-len)     OUTPUT_LEN="$2"; shift 2;;
    --model-path)     MODEL_PATH="$2"; shift 2;;
    --tp)             TP="$2"; shift 2;;
    --kv-tokens)      KV_TOKENS="$2"; shift 2;;
    --port)           PORT="$2"; shift 2;;
    --out-root)       OUT_ROOT="$2"; shift 2;;
    --)               shift; EXTRA_BENCH_ARGS+=("$@"); break;;
    *)                EXTRA_BENCH_ARGS+=("$1"); shift;;
  esac
done

STAMP="$(date -u +%Y%m%d_%H%M%SZ)"
RUN_DIR="$OUT_ROOT/incremental_prefill-$STAMP"
mkdir -p "$RUN_DIR"

# Load-order plumbing (same as run_prefill_ttft.sh): PYTHONPATH → sitecustomize
# loads shim + user overrides in the SERVER process (which is where the
# patched operators actually run).
export PYTHONPATH="$HERE/shim:$HERE:${PYTHONPATH:-}"
if [[ -n "$OVERRIDES" ]]; then
  export KDA_E2E_OVERRIDES="$(realpath "$OVERRIDES")"
fi

# gfx942 env inherited by server workers.
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

SERVER_LOG="$RUN_DIR/server.log"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[wrapper] shutting down sglang server pid=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in {1..10}; do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ── manifest ─────────────────────────────────────────────────────────
python3 - <<PY > "$RUN_DIR/manifest.json"
import json, os, socket
q_str = "${QUESTION_LEN[*]}"
print(json.dumps({
    "scenario": "incremental_prefill_ttft",
    "started_at": "$STAMP",
    "host": socket.gethostname(),
    "model_path": "$MODEL_PATH",
    "tp": $TP,
    "kv_tokens": $KV_TOKENS,
    "system_prompt_len": $SYSTEM_PROMPT_LEN,
    "question_lens": [int(x) for x in q_str.split()],
    "output_len": $OUTPUT_LEN,
    "num_prompts_per_shape": $NUM_PROMPTS,
    "overrides": os.environ.get("KDA_E2E_OVERRIDES") or None,
    "backends": {"dsa_prefill":"aiter","dsa_decode":"fa3","dsa_topk":"torch","moe":"triton"},
    "sglang_entry": "python -m sglang.launch_server + python -m sglang.bench_serving --dataset generated-shared-prefix",
    "env": {k: os.environ[k] for k in sorted(os.environ)
            if k.startswith(("SGLANG_","KDA_","HIP_","PYTORCH_"))},
}, indent=2))
PY

echo "══════════════════════════════════════════════════════════════════════"
echo "  GLM-5.2 INCREMENTAL PREFILL TTFT"
echo "  system prompt (cached prefix) = $SYSTEM_PROMPT_LEN tokens"
echo "  question len sweep            = ${QUESTION_LEN[*]}"
echo "  output len                    = $OUTPUT_LEN"
echo "  requests per shape            = $NUM_PROMPTS"
echo "  TP=$TP  KV pool=$KV_TOKENS"
echo "  overrides = ${OVERRIDES:-(none — vanilla sglang)}"
echo "  results   = $RUN_DIR/"
echo "══════════════════════════════════════════════════════════════════════"

# ── launch server (once, reused across question-len sweep) ───────────
echo "[wrapper] launching sglang server on port $PORT (log: $SERVER_LOG)"
"$PYTHON" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --tp "$TP" \
    --trust-remote-code \
    --mem-fraction-static "$MEM_FRAC" \
    --max-total-tokens "$KV_TOKENS" \
    --dsa-topk-backend torch \
    --dsa-prefill-backend aiter \
    --dsa-decode-backend aiter \
    --moe-runner-backend triton \
    --disable-cuda-graph \
    --host 127.0.0.1 --port "$PORT" \
    "${EXTRA_SERVER_ARGS[@]}" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[wrapper] server pid=$SERVER_PID"

# ── wait until server is up (poll /health) ───────────────────────────
echo "[wrapper] waiting for server readiness (timeout ${SERVER_READY_TIMEOUT}s)"
DEADLINE=$(( $(date +%s) + SERVER_READY_TIMEOUT ))
while [[ $(date +%s) -lt $DEADLINE ]]; do
  if curl -s -f "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
    echo "[wrapper] server ready"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[wrapper] server died before ready; see $SERVER_LOG" >&2
    tail -30 "$SERVER_LOG" >&2
    exit 3
  fi
  sleep 5
done
if ! curl -s -f "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
  echo "[wrapper] server not ready after ${SERVER_READY_TIMEOUT}s" >&2
  tail -30 "$SERVER_LOG" >&2
  exit 3
fi

# ── sweep question lengths ───────────────────────────────────────────
for q in "${QUESTION_LEN[@]}"; do
  label="q${q}_prefix${SYSTEM_PROMPT_LEN}_np${NUM_PROMPTS}"
  result_file="$RUN_DIR/${label}.jsonl"
  bench_log="$RUN_DIR/${label}.log"
  echo ""
  echo "─── question_len=$q  (num_prompts=$NUM_PROMPTS) ───"
  "$PYTHON" -m sglang.bench_serving \
      --backend sglang \
      --host 127.0.0.1 --port "$PORT" \
      --dataset-name generated-shared-prefix \
      --gsp-num-groups 1 \
      --gsp-prompts-per-group "$NUM_PROMPTS" \
      --gsp-system-prompt-len "$SYSTEM_PROMPT_LEN" \
      --gsp-question-len "$q" \
      --gsp-output-len "$OUTPUT_LEN" \
      --num-prompts "$NUM_PROMPTS" \
      --output-file "$result_file" \
      "${EXTRA_BENCH_ARGS[@]}" \
      > "$bench_log" 2>&1
  echo "  → $bench_log"
  grep -E "TTFT|Mean|Median|P90|P99|Success" "$bench_log" | head -12 || true
done

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  done — server will be killed on exit trap"
echo "  results in $RUN_DIR/"
echo "══════════════════════════════════════════════════════════════════════"
