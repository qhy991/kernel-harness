#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=''

run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
variant_root=/home/qinhaiyan/glm52-v2-goal-runs/cache/24-moe_w13_decode/deepgemm/w13_variants
stock_entry="$variant_root/jit/stock/cache/kernel.sm100_m_grouped_fp8_fp4_gemm_masked_1d1d.fd165cc8140afb4214951eb474041a15"
one_sm_entry="$variant_root/jit/candidate/cache/kernel.sm100_m_grouped_fp8_fp4_gemm_masked_1d1d.51dabe4dc4117fcdce86e78f87f10636"
two_sm_entry="$variant_root/jit/candidate/cache/kernel.sm100_m_grouped_fp8_fp4_gemm_masked_1d1d.8e4280b1570bdb24108477c0207e76bb"
stock_include="$variant_root/artifacts/stock/site/deep_gemm_w13_stock/include"
candidate_include="$variant_root/artifacts/candidate/site/deep_gemm_w13_candidate/include"

compile_ptx() {
  local source_path=$1
  local include_path=$2
  local output_path=$3
  /usr/local/cuda-13.2/bin/nvcc \
    "$source_path" \
    -ptx \
    -o "$output_path" \
    -std=c++20 \
    --diag-suppress=39,161,174,177,186,940 \
    --ptxas-options=--register-usage-level=10 \
    -I"$include_path" \
    --gpu-architecture=sm_100a \
    --compiler-options=-fPIC,-O3,-fconcepts,-Wno-deprecated-declarations,-Wno-abi \
    -O3 \
    --expt-relaxed-constexpr \
    --expt-extended-lambda
}

compile_ptx "$stock_entry/kernel.cu" "$stock_include" "$run_dir/analysis/stock.ptx"
compile_ptx "$one_sm_entry/kernel.cu" "$candidate_include" "$run_dir/analysis/bm32_1sm.ptx"
compile_ptx "$two_sm_entry/kernel.cu" "$candidate_include" "$run_dir/analysis/bm32_2sm.ptx"

/usr/local/cuda-13.2/bin/nvdisasm -g "$stock_entry/kernel.cubin" \
  > "$run_dir/analysis/stock.sass"
/usr/local/cuda-13.2/bin/nvdisasm -g "$one_sm_entry/kernel.cubin" \
  > "$run_dir/analysis/bm32_1sm.sass"
/usr/local/cuda-13.2/bin/nvdisasm -g "$two_sm_entry/kernel.cubin" \
  > "$run_dir/analysis/bm32_2sm.sass"

sha256sum \
  "$stock_entry/kernel.cu" \
  "$stock_entry/kernel.cubin" \
  "$one_sm_entry/kernel.cu" \
  "$one_sm_entry/kernel.cubin" \
  "$two_sm_entry/kernel.cu" \
  "$two_sm_entry/kernel.cubin" \
  "$run_dir/analysis/stock.ptx" \
  "$run_dir/analysis/bm32_1sm.ptx" \
  "$run_dir/analysis/bm32_2sm.ptx" \
  "$run_dir/analysis/stock.sass" \
  "$run_dir/analysis/bm32_1sm.sass" \
  "$run_dir/analysis/bm32_2sm.sass" \
  > "$run_dir/analysis/codegen_sha256.txt"

for artifact in \
  "$run_dir/analysis/stock.ptx" \
  "$run_dir/analysis/bm32_1sm.ptx" \
  "$run_dir/analysis/bm32_2sm.ptx" \
  "$run_dir/analysis/stock.sass" \
  "$run_dir/analysis/bm32_1sm.sass" \
  "$run_dir/analysis/bm32_2sm.sass"; do
  gzip -n -9 -c "$artifact" > "$artifact.gz"
done

sha256sum "$run_dir"/analysis/*.ptx.gz "$run_dir"/analysis/*.sass.gz \
  >> "$run_dir/analysis/codegen_sha256.txt"
