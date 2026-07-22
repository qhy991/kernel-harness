# 复现 E2E Prefill Profile（2026-07-22）

## 环境

```bash
source /root/sglang-exp/env_setup/env.sh   # rocm-torch + aiter + sglang
# 需要：/root/sglang-exp/tasks/dense-fp8-gemm/runs/{profile_entry.py,run_glm52_no_offload.py}
# 可选：GLM5_FP8_FAST_PATH=1 + /root/glm5-flops-amd/fp8_fast_path.py
```

## 关键参数（与本次一致）

- TP=8，`mem-fraction-static 0.95`，`max-total-tokens 65536`
- `--dsa-prefill-backend aiter --dsa-decode-backend aiter --dsa-topk-backend torch`
- `--moe-runner-backend triton`
- `--disable-cuda-graph`
- `--profile --profile-stage prefill --profile-activities CPU GPU --profile-record-shapes`
- `input-len 1024 2048 4096`，`output-len 1`
- **不要**把含 `sitecustomize.py` 的 shim 目录挂进 `PYTHONPATH`（会弄坏 HIP/Triton）
- `ulimit -c 0`；NUMA balancing 若可关更好（本容器只读 sysctl 时关不了）

## 启动示例

```bash
export OUT_DIR=/mnt/public/qinhaiyan/glm52_profile_traces
export RUNS_DIR=/root/sglang-exp/tasks/dense-fp8-gemm/runs
export PYTHONPATH="${RUNS_DIR}:/root/glm5-flops-amd:/root/repos/aiter:/root/repos/sglang/python"
export SGLANG_TORCH_PROFILER_DIR="$OUT_DIR"
export SGLANG_USE_AITER=1
export SGLANG_DSA_USE_AITER_SPARSE_MLA=0
export GLM5_FP8_FAST_PATH=1
export SGLANG_DISABLE_GFX942_BPRESHUFFLE=1

cd "$RUNS_DIR"
python profile_entry.py \
  --model-path /mnt/public/qinhaiyan/models/GLM-5.2-FP8 \
  --tp 8 --batch-size 1 \
  --input-len 1024 2048 4096 --output-len 1 \
  --trust-remote-code --mem-fraction-static 0.95 \
  --dsa-topk-backend torch \
  --dsa-prefill-backend aiter --dsa-decode-backend aiter \
  --moe-runner-backend triton \
  --max-total-tokens 65536 --cuda-graph-max-bs 1 --disable-cuda-graph \
  --result-filename "$OUT_DIR/profile_result.jsonl" \
  --profile --profile-stage prefill \
  --profile-activities CPU GPU --profile-record-shapes \
  --profile-filename-prefix glm52_prefill
```

## 重算份额

```bash
python analyze_chrome_trace.py \
  --trace-dir /mnt/public/qinhaiyan/glm52_profile_traces \
  --glob 'glm52_prefill_20260722_172150_*_prefill.trace.json.gz' \
  --out-csv e2e_prefill_op_share.csv
```

## 已知坑

1. 上次 `fa3` + 强制 FP8 KV 曾 **Memory access fault**；本次用 aiter/aiter + bf16 KV。
2. Crash 后可能 orphan VRAM；容器内 `gpureset` 不可用，需重启 pod。
3. Traces ~30MB，存在 NFS，不入库本 git 目录。
