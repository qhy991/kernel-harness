# External TP8/DP8/EP8 acceptance commands

These commands are the remaining production acceptance lane for the local
BM16 two-SM survivor. They require one eight-B200 node and the exact
`zai-org/GLM-5.2-FP8` checkpoint. They were not run on the current host.

## Immutable preparation

Use these committed revisions:

- SGLang `5af212d00439a8990a1d64e2b7e32aa207acf2cb`
  (base `83d313104d089bcd2af26b28453ff880f1e6a80b`);
- DeepGEMM `87e0359edbb461181d3bba218442132007b9a738`
  (base `731e7c7a97d269e4b9f482ea18d0e709a948f293`);
- CUTLASS `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`;
- fmt `553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28`;
- DeepGEMM diff SHA256
  `465c8373c0a37970225a0e93267b6c399431b23e22cf35b4511db2308df98092`.

The committed provider intentionally binds the task-local manifest path below.
Use that exact path on the acceptance node.

```bash
export SGLANG_ROOT=/path/to/committed/moe-w13-decode/sglang
export DEEPGEMM_ROOT=/path/to/committed/moe-w13-decode/deepgemm
export W13_CACHE=/home/qinhaiyan/glm52-hotspot-goal-runs/cache/moe_w13_decode
export W13_BUILD="${W13_CACHE}/deepgemm/w13_variants"
export W13_MANIFEST="${W13_BUILD}/manifest.json"
export PYTHON=/path/to/sglang-python

cd "${SGLANG_ROOT}"
CUDA_VISIBLE_DEVICES='' MAX_JOBS=4 \
  TRITON_CACHE_DIR="${W13_CACHE}/triton" \
  TORCH_EXTENSIONS_DIR="${W13_CACHE}/torch_extensions" \
  CUDA_CACHE_PATH="${W13_CACHE}/cuda" \
  XDG_CACHE_HOME="${W13_CACHE}/xdg" \
  "${PYTHON}" third_party/deepgemm_w13/build_variants.py \
  --source "${DEEPGEMM_ROOT}" \
  --output "${W13_BUILD}" \
  --candidate-commit 87e0359edbb461181d3bba218442132007b9a738 \
  --force

sha256sum "${W13_MANIFEST}"

CUDA_VISIBLE_DEVICES='' \
  "${PYTHON}" third_party/deepgemm_w13/build_variants.py \
  --source "${DEEPGEMM_ROOT}" \
  --output "${W13_CACHE}/deepgemm/materialization-audit" \
  --candidate-commit 87e0359edbb461181d3bba218442132007b9a738 \
  --audit-materialization
```

Reject the build unless materialization reports the exact commits/diff above,
schema 3, identical stock/candidate normalized build-plan hashes, hidden C++
visibility, `Bsymbolic`, and no candidate JIT context duplicates.

## Frozen server command

Use the same server command for stock and candidate, changing only the hotspot
environment block. `--tp 8 --dp 8 --enable-dp-attention` plus DeepEP must
resolve production expert parallelism to eight.

Common environment and command:

```bash
cd "${SGLANG_ROOT}"
export PYTHONPATH="${SGLANG_ROOT}/python"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_DEEPGEMM_PDL=1
export DG_JIT_CACHE_DIR="${W13_BUILD}/jit/candidate"
export SGLANG_DG_CACHE_DIR="${W13_BUILD}/jit/candidate"

"${PYTHON}" -m sglang.launch_server \
  --model-path zai-org/GLM-5.2-FP8 \
  --trust-remote-code \
  --tp 8 \
  --dp 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --moe-runner-backend deep_gemm \
  --cuda-graph-bs-decode 16 32 \
  --max-running-requests 256 \
  --mem-fraction-static 0.85 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-metrics \
  --host 0.0.0.0 \
  --port 30000
```

Before a stock restart:

```bash
export SGLANG_GLM52_OPT=0
unset SGLANG_GLM52_OPT_PROFILE
unset SGLANG_GLM52_OPT_OPS
unset SGLANG_GLM52_OPT_M_BUCKETS
unset SGLANG_GLM52_HOTSPOT_MODULE
```

Before a candidate restart:

```bash
export SGLANG_GLM52_OPT=1
export SGLANG_GLM52_OPT_PROFILE=hotspot_candidates
export SGLANG_GLM52_OPT_OPS=moe_w13
export 'SGLANG_GLM52_OPT_M_BUCKETS=moe_gate_proj:16|32'
export SGLANG_GLM52_HOTSPOT_MODULE="${SGLANG_ROOT}/third_party/deepgemm_w13/provider_bm16_2sm.py"
```

At candidate startup, require every worker to report the assigned GPU before
provider initialization, the exact provider commit/build ID, distinct
stock/candidate DSO and JIT identities, PDL=true, `num_sms=148`,
`tc_util=100`, broad precompile disabled, and only expected-M 4/5/8/9 warmed.
Any failure after candidate selection fails the arm; it must not retry stock.

## Decode load

Global concurrency 128 and 256 target local DP8 buckets M16 and M32. A runtime
trace must confirm the actual local bucket and expected-M on every rank.

```bash
cd "${SGLANG_ROOT}"
PYTHONPATH="${SGLANG_ROOT}/python" \
  "${PYTHON}" -m sglang.benchmark.serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model zai-org/GLM-5.2-FP8 \
  --dataset-name random \
  --num-prompts 384 \
  --random-input-len 1024 \
  --random-output-len 256 \
  --random-range-ratio 0 \
  --request-rate inf \
  --max-concurrency 128 \
  --seed 0 \
  --temperature 0 \
  --output-details \
  --output-file /path/to/evidence/m16.jsonl

PYTHONPATH="${SGLANG_ROOT}/python" \
  "${PYTHON}" -m sglang.benchmark.serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model zai-org/GLM-5.2-FP8 \
  --dataset-name random \
  --num-prompts 768 \
  --random-input-len 1024 \
  --random-output-len 256 \
  --random-range-ratio 0 \
  --request-rate inf \
  --max-concurrency 256 \
  --seed 0 \
  --temperature 0 \
  --output-details \
  --output-file /path/to/evidence/m32.jsonl
```

Use full process restarts in the order stock/candidate, candidate/stock,
stock/candidate. Preserve all raw requests and maximum-rank samples. Within
each restart, measure the leaf eager, separately captured leaf graph, complete
`dispatch -> W13 -> activation/quant -> W2 -> combine` eager, and production
graph lanes for each observed expected-M 4/5/8/9.

## Checkpoint correctness and registered smoke

```bash
cd "${SGLANG_ROOT}"
PYTHONPATH="${SGLANG_ROOT}/python" \
  "${PYTHON}" test/registered/8-gpu-models/test_glm52_fp8.py
```

Repeat the registered TP8+DP8 case under the candidate environment and the
same DeepGEMM/DeepEP arguments. Compare checkpoint outputs against stock
before considering performance.

## Acceptance

Promotion to `production-win` requires:

- exact TP8/DP8/EP8 and DeepEP low-latency topology on all eight ranks;
- checkpoint accuracy and request success equal to stock;
- exact W13 ABI, expected-M, provider, symbol, graph, pointer, output, return,
  stream, and no-retry evidence;
- identical dispatch, activation/quant, W2, and combine nodes between arms;
- at least three alternating restart series with all four finite estimators
  at least 1.03 for every enabled leaf and containing-region eager/graph lane;
- improved TTFT, TPOT, throughput, and maximum-rank region latency with no
  enabled-bucket regression.

TP4, synthetic weights, config-only checkpoint data, or a leaf-only result
must remain separately labeled diagnostics.
