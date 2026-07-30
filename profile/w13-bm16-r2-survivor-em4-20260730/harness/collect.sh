#!/usr/bin/env bash
# One GPU lease: NCU full-set capture of the exact stock and BM16 two-SM W13
# symbols at expected-M 4, plus per-PC stall attribution for the survivor.
set -euo pipefail
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SGLANG=/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/moe-w13-decode/sglang
MANIFEST=/home/qinhaiyan/glm52-hotspot-goal-runs/cache/moe_w13_ptx_r2/deepgemm/w13_variants/manifest.json
PY=/home/qinhaiyan/miniconda3/envs/sglang/bin/python
NCU=/usr/local/cuda/bin/ncu

nvidia-smi --query-gpu=index,uuid,clocks.sm,clocks.mem,pstate \
  --format=csv > "$RUN/analysis/clocks_at_capture.csv"

cd "$SGLANG"
for arm in candidate stock; do
  if [[ "$arm" == candidate ]]; then
    filter='infini_kernel_glm52_moe_w13_decode_em4_bm16_2sm'
    variant=bm16_2sm
  else
    filter='sm100_fp8_fp4_gemm_1d1d_impl'
    variant=bm16_2sm
  fi
  "$NCU" --set full \
    --kernel-name "regex:${filter}" \
    --launch-count 1 \
    --target-processes all \
    --force-overwrite \
    --export "$RUN/reports/full_${arm}" \
    "$PY" third_party/deepgemm_w13/profile_survivor.py \
      --manifest "$MANIFEST" --arm "$arm" --variant "$variant" \
      --expected-m 4 --warmup 3 --profiled-launches 1 \
      --output "$RUN/analysis/launch_${arm}.json" \
    > "$RUN/analysis/collect_full_${arm}.log" 2>&1 || \
    echo "full ${arm} collection exited $?" >> "$RUN/analysis/collect_full_${arm}.log"
done

"$NCU" --set source --section SourceCounters \
  --kernel-name 'regex:infini_kernel_glm52_moe_w13_decode_em4_bm16_2sm' \
  --launch-count 1 --target-processes all --force-overwrite \
  --export "$RUN/reports/source_candidate" \
  "$PY" third_party/deepgemm_w13/profile_survivor.py \
    --manifest "$MANIFEST" --arm candidate --variant bm16_2sm \
    --expected-m 4 --warmup 3 --profiled-launches 1 \
    --output "$RUN/analysis/launch_source_candidate.json" \
  > "$RUN/analysis/collect_source_candidate.log" 2>&1 || \
  echo "source collection exited $?" >> "$RUN/analysis/collect_source_candidate.log"

ls -la "$RUN/reports"
