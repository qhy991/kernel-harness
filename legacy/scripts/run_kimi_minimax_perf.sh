#!/usr/bin/env bash
# Benchmark Kimi K2.x (DSA ops 1-27) + MiniMax M3-related kernels on B200.
# Usage:
#   source activate_env.sh
#   bash legacy/scripts/run_kimi_minimax_perf.sh
set -uo pipefail

# LEGACY_ROOT holds this stack's own code (run_all.py, benchmarks/, shapes.py);
# REPO_ROOT holds the things it borrows from the repo (the venv, logs/). They were
# the same directory until this stack moved under legacy/, so they must be kept
# apart now or the script silently looks for activate_env.sh inside legacy/.
LEGACY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${LEGACY_ROOT}/.." && pwd)"
source "${REPO_ROOT}/activate_env.sh"
SGLANG_DIR="${SGLANG_DIR:-/home/qinhaiyan/sglang}"
OUT="${REPO_ROOT}/logs/kimi_minimax_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${SGLANG_DIR}/python:${PYTHONPATH:-}"

echo "============================================================"
echo " OUT=$OUT"
echo " PYTHON=$PYTHON"
echo " SGLANG_DIR=$SGLANG_DIR"
echo "============================================================"
"$PYTHON" - <<'PY'
import torch, sgl_kernel, deep_gemm
print(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")
print("sgl_kernel OK, deep_gemm OK")
PY

# ---------- 1) Harness baseline: Kimi DSA ops 1-27 ----------
echo -e "\n##### [1/3] kernel-harness baseline (Kimi DSA ops 1-27) #####"
cd "$LEGACY_ROOT"
"$PYTHON" run_all.py --out "$OUT/baseline" 2>&1 | tee "$OUT/01_baseline.log"
cp -f "$OUT/baseline/results.json" "$OUT/baseline_results.json" 2>/dev/null || true
cp -f "$OUT/baseline/results.csv"  "$OUT/baseline_results.csv"  2>/dev/null || true

# ---------- 2) Official sglang benches for Kimi DSA path ----------
echo -e "\n##### [2/3] official sglang kernel benches (Kimi DSA) #####"
cd "$SGLANG_DIR"
kimi_benches=(
  # Dense FP8 GEMM — ops 1-5, 9-10, 14-16, 19
  "benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm_blackwell.py"
  "benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py"
  # Grouped FP8 MoE — ops 12, 13, 24, 25
  "benchmark/kernels/deepseek/benchmark_deepgemm_fp8_group_gemm.py"
  # Triton MoE fallback
  "benchmark/kernels/fused_moe_triton/benchmark_sglang_fused_moe_triton.py"
  # Router GEMM — ops 11, 23
  "benchmark/kernels/deepseek/benchmark_deepgemm_dsv3_router_gemm_blackwell.py"
  # Indexer / topk / fused-a (may be missing on this checkout)
  "test/registered/jit/benchmark/bench_dsv4_fp4_indexer.py"
  "test/registered/jit/benchmark/bench_topk.py"
  "test/registered/jit/benchmark/bench_dsv3_fused_a_gemm.py"
  "test/registered/jit/benchmark/bench_dsv3_router_gemm.py"
  "benchmark/kernels/deepseek/benchmark_cute_dsl_fp8_paged_mqa_logits.py"
)

: > "$OUT/02_official_kimi.log"
for b in "${kimi_benches[@]}"; do
  if [[ -f "$b" ]]; then
    echo -e "\n>>> $b" | tee -a "$OUT/02_official_kimi.log"
    # Prefer --help first to discover args; most of these self-run with defaults.
    timeout 600 "$PYTHON" "$b" 2>&1 | tee -a "$OUT/02_official_kimi.log" | tail -40
    echo "EXIT:$?" | tee -a "$OUT/02_official_kimi.log"
  else
    echo "  (MISSING: $b)" | tee -a "$OUT/02_official_kimi.log"
  fi
done

# ---------- 3) MiniMax M3-specific benches / tests ----------
echo -e "\n##### [3/3] MiniMax M3 benches + unit tests #####"
cd "$SGLANG_DIR"
: > "$OUT/03_minimax.log"
minimax_benches=(
  "test/registered/jit/benchmark/minimax/bench_minimax_qknorm_rope.py"
  "test/registered/jit/benchmark/minimax/bench_minimax_decode_topk.py"
  "test/registered/jit/benchmark/minimax/bench_minimax_store_kv_index.py"
)
minimax_tests=(
  "test/registered/jit/minimax/test_minimax_qknorm_rope.py"
  "test/registered/jit/minimax/test_minimax_decode_topk.py"
  "test/registered/jit/minimax/test_minimax_store_kv_index.py"
  "test/registered/jit/minimax/test_minimax_decode_topk_page_table.py"
  "python/sglang/srt/layers/attention/minimax_sparse_ops/tests/test_sparse_gqa.py"
  "python/sglang/srt/layers/attention/minimax_sparse_ops/tests/test_flash_with_topk_idx.py"
)

any_minimax=0
for b in "${minimax_benches[@]}"; do
  if [[ -f "$b" ]]; then
    any_minimax=1
    echo -e "\n>>> $b" | tee -a "$OUT/03_minimax.log"
    timeout 600 "$PYTHON" "$b" 2>&1 | tee -a "$OUT/03_minimax.log" | tail -40
  else
    echo "  (MISSING bench: $b)" | tee -a "$OUT/03_minimax.log"
  fi
done
for t in "${minimax_tests[@]}"; do
  if [[ -f "$t" ]]; then
    any_minimax=1
    echo -e "\n>>> pytest $t" | tee -a "$OUT/03_minimax.log"
    timeout 600 "$PYTHON" -m pytest -q "$t" 2>&1 | tee -a "$OUT/03_minimax.log" | tail -30
  else
    echo "  (MISSING test: $t)" | tee -a "$OUT/03_minimax.log"
  fi
done
if [[ "$any_minimax" -eq 0 ]]; then
  echo "NOTE: this sglang checkout has no MiniMax-M3 JIT/sparse_ops sources." | tee -a "$OUT/03_minimax.log"
  echo "      M3 ops 28-43 in docs/kernel_api_mapping.csv are external EntryClass." | tee -a "$OUT/03_minimax.log"
fi

# ---------- Summary ----------
echo -e "\n##### DONE #####"
echo "Results under: $OUT"
ls -la "$OUT"
echo "$OUT" > "$REPO_ROOT/logs/kimi_minimax_latest.path"
