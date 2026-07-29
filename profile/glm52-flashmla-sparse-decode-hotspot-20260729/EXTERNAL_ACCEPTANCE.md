# External TP8/DP8/EP8 acceptance commands

These commands are retained to make the unavailable checkpoint-backed gate
reproducible. They are **not authorization to deploy this campaign's
candidate**: the candidate failed mandatory local graph and containing-region
gates, so the terminal disposition remains `no-replacement`.

Run from the matching committed SGLang worktree on an eight-B200 node with the
exact `zai-org/GLM-5.2-FP8` checkpoint available locally. Keep stock and the
explicit experiment in separate fresh server processes and preserve all JSONL,
hit-file, server-log, and per-rank profiler artifacts.

## Common environment

```bash
export MODEL_PATH=/absolute/path/to/zai-org/GLM-5.2-FP8
export SGLANG_ROOT=/absolute/path/to/committed/sglang
export KH_ROOT=/absolute/path/to/committed/kernel-harness
export FLASHMLA_ROOT=/absolute/path/to/committed/flashmla
export RESULTS=/absolute/path/to/new/acceptance-results
mkdir -p "$RESULTS"

export CUDA_CACHE_PATH="$RESULTS/cache/cuda"
export TORCH_EXTENSIONS_DIR="$RESULTS/cache/torch_extensions"
export TRITON_CACHE_DIR="$RESULTS/cache/triton"
export XDG_CACHE_HOME="$RESULTS/cache/xdg"
export MAX_JOBS=4
export NVCC_THREADS=2
export CMAKE_BUILD_PARALLEL_LEVEL=4
```

## Stock server

```bash
cd "$SGLANG_ROOT"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
SGLANG_GLM52_OPT=0 \
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 --port 30000 \
  --tp 8 --dp 8 --enable-dp-attention --ep 8 \
  --dsa-decode-backend flashmla_kv \
  --trust-remote-code \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --mem-fraction-static 0.85 \
  --enable-metrics \
  2>&1 | tee "$RESULTS/stock-server.log"
```

## Explicit experimental server

This is the exact API-v1 selector. It remains default-off and must be used only
in an isolated acceptance run.

```bash
cd "$SGLANG_ROOT"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
GLM52_FLASHMLA_SOURCE="$FLASHMLA_ROOT" \
GLM52_FLASHMLA_VARIANT=b3_b5_native_exact \
SGLANG_GLM52_OPT=1 \
SGLANG_GLM52_OPT_PROFILE=hotspot_candidates \
SGLANG_GLM52_OPT_OPS=flashmla_sparse_decode \
SGLANG_GLM52_OPT_M_BUCKETS='dsa_decode_attn:16|32' \
SGLANG_GLM52_HOTSPOT_MODULE="$KH_ROOT/serving_native/candidates/flashmla_sparse_decode_provider.py" \
SGLANG_GLM52_OPT_HIT_FILE="$RESULTS/candidate-hits.json" \
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 --port 30000 \
  --tp 8 --dp 8 --enable-dp-attention --ep 8 \
  --dsa-decode-backend flashmla_kv \
  --trust-remote-code \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --mem-fraction-static 0.85 \
  --enable-metrics \
  2>&1 | tee "$RESULTS/candidate-server.log"
```

Also run separate TP8-only (`--tp 8`, without DP/EP flags) and TP8+DP8
(`--tp 8 --dp 8 --enable-dp-attention`) correctness lanes. The final production
lane is the TP8+DP8+EP8 command above.

## TTFT, TPOT, throughput, and M16/M32 pressure

Run each command once against stock and once against the explicit experiment,
changing only `--output-file`. Global concurrency 128 and 256 target balanced
DP8 local decode M16 and M32; the hit file and per-rank traces must confirm the
actual buckets rather than assuming balance.

```bash
cd "$SGLANG_ROOT"
python -m sglang.benchmark.serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model "$MODEL_PATH" \
  --dataset-name random \
  --num-prompts 4096 \
  --random-input-len 8192 \
  --random-output-len 256 \
  --request-rate inf \
  --max-concurrency 128 \
  --output-details \
  --output-file "$RESULTS/VARIANT-concurrency128.jsonl"

python -m sglang.benchmark.serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model "$MODEL_PATH" \
  --dataset-name random \
  --num-prompts 4096 \
  --random-input-len 8192 \
  --random-output-len 256 \
  --request-rate inf \
  --max-concurrency 256 \
  --output-details \
  --output-file "$RESULTS/VARIANT-concurrency256.jsonl"
```

Require non-regressing p50/p90/p99 TTFT and TPOT, higher request/output-token
throughput, correct M16/M32 hit counts on every rank, and no unsupported-shape
candidate launch.

## Output correctness

Run the repository's checkpoint-backed eight-GPU GLM-5.2 suite first with
stock and then with the explicit experiment environment:

```bash
cd "$SGLANG_ROOT"
python test/registered/8-gpu-models/test_glm52_fp8.py
```

This covers the pinned FP8 model with TP8, TP8+DP8, and TP8+DP8+MTP. Add the
EP8 server lane above and compare deterministic prompts token-for-token or with
the repository accuracy thresholds. A performance result cannot compensate
for any output mismatch.

## Rank-max device latency

Launch each server under Nsys capture-range control:

```bash
nsys profile \
  --trace=cuda,nvtx \
  --cuda-graph-trace=node \
  --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --output="$RESULTS/VARIANT-tp8-dp8-ep8" \
  python -m sglang.launch_server ...the matching server arguments above...
```

Trigger a bounded decode capture while running the same concurrency workload:

```bash
python -m sglang.benchmark.serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model "$MODEL_PATH" \
  --dataset-name random \
  --num-prompts 512 \
  --random-input-len 8192 \
  --random-output-len 128 \
  --max-concurrency 128 \
  --profile \
  --profile-activities CUDA_PROFILER \
  --profile-start-step 10 \
  --profile-steps 20 \
  --output-file "$RESULTS/VARIANT-profile-window.jsonl"
```

Export `cuda_gpu_trace:nvtx-name:base` and compute the maximum complete
main-plus-combine and containing-decode span across all eight worker/rank
processes. Gate on rank-max, not the mean; reject extra graph nodes, a changed
non-target sequence, or a single slow rank.
