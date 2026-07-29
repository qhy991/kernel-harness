# External TP8/DP8/EP8 validation commands

These commands preserve the acceptance contract on one eight-B200 node with
the exact `zai-org/GLM-5.2-FP8` checkpoint. They were not run on the audited
four-GPU/no-weights host. They are diagnostic documentation for a future
candidate: Goal 24's current BM32 candidate already failed a mandatory local
gate and cannot be promoted by an external result.

## Immutable preparation

Use the committed SGLang task revision and materialize stock/candidate from the
tracked inputs. The resulting runtime manifest, not the documentation copy,
must be passed to workers because it contains the external node's absolute DSO
and JIT paths. The required SGLang revision is
`1c671bf3a30360100e7947c87e0c873a387ad0be`.

```bash
export SGLANG_ROOT=/path/to/committed/24-moe-w13-decode/sglang
export W13_CACHE=/path/to/task-local/24-moe_w13_decode
export W13_MANIFEST="${W13_CACHE}/deepgemm/w13_variants/manifest.json"

cd "${SGLANG_ROOT}"
CUDA_VISIBLE_DEVICES='' MAX_JOBS=1 \
  /path/to/sglang-python third_party/deepgemm_w13/build_variants.py \
  --output "${W13_CACHE}/deepgemm/w13_variants" --force
sha256sum "${W13_MANIFEST}"
CUDA_VISIBLE_DEVICES='' \
  /path/to/sglang-python third_party/deepgemm_w13/build_variants.py \
  --upstream /path/to/DeepGEMM-GLM52 \
  --output "${W13_CACHE}/deepgemm/materialization-audit" \
  --audit-materialization
```

Reject the lane unless the source identities remain:

- DeepGEMM `731e7c7a97d269e4b9f482ea18d0e709a948f293`
- CUTLASS `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`
- fmt `553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28`
- patch SHA256
  `997348b6498aa18a7d70a5b1d36249b356b508cdc71e2f514a979818c48490a5`

## Frozen server command

Run the same server command for stock and candidate, changing only the two W13
environment variables. `--tp 8 --dp 8 --enable-dp-attention` plus DeepEP makes
the resolved production expert-parallel size 8; capture the resolved server
args and reject any different topology.

```bash
cd "${SGLANG_ROOT}"
export PYTHONPATH="${SGLANG_ROOT}/python"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_DEEPGEMM_PDL=1
export SGLANG_GLM52_W13_DECODE_VARIANT=
unset SGLANG_GLM52_W13_DECODE_MANIFEST

/path/to/sglang-python -m sglang.launch_server \
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

For the candidate restart, use the identical command after setting:

```bash
export SGLANG_GLM52_W13_DECODE_VARIANT=bm32_2sm
export SGLANG_GLM52_W13_DECODE_MANIFEST="${W13_MANIFEST}"
```

At startup, require every worker to report the assigned GPU before W13
initialization, distinct stock/candidate packages/DSOs/JIT roots, PDL=true,
`num_sms=148`, `tc_util=100`, successful state-independence probes, broad
precompile disabled, and only expected-M 4/5/8/9 warmed. Any mismatch is a
failed candidate arm, not stock fallback.

## Exact decode-load commands

Global concurrency 128 and 256 target local DP8 decode buckets M16 and M32.
The runtime trace must confirm the actual local bucket and expected-M on every
rank; do not infer them from the client setting.

```bash
cd "${SGLANG_ROOT}"
PYTHONPATH="${SGLANG_ROOT}/python" \
  /path/to/sglang-python -m sglang.benchmark.serving \
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
  /path/to/sglang-python -m sglang.benchmark.serving \
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
stock/candidate. Preserve every raw request and maximum-rank latency sample.
Within each restart, validate leaf eager, separately captured leaf graph,
complete `dispatch -> W13 -> activation/quant -> W2 -> combine` eager, and
production graph lanes for all independently observed expected-M 4/5/8/9.

## Checkpoint accuracy and registered server smoke

The unchanged repository acceptance remains:

```bash
cd "${SGLANG_ROOT}"
PYTHONPATH="${SGLANG_ROOT}/python" \
  /path/to/sglang-python \
  test/registered/8-gpu-models/test_glm52_fp8.py
```

That registered lane checks TP8, TP8+DP8 and TP8+DP8+EAGLE variants with GSM8K
baseline accuracy 0.92. For the W13-specific candidate, repeat its TP8+DP8
variant with the same W13 environment and the `deep_gemm`/DeepEP arguments
above, then compare checkpoint outputs against stock before performance is
considered.

## Required evidence

Accept only a run that records:

- physical UUID and maximum-rank timing for all eight ranks;
- resolved TP8/DP8/EP8 and DeepEP low-latency mode;
- exact W13 tensor shapes/strides, device `masked_m`, expected-M and recipes;
- graph IDs/nodes, pointer stability, activation and device-mask mutation,
  output poison restoration, deterministic replay and untouched rows;
- exactly one candidate W13 call with exact `None` return and no retry;
- identical dispatch, activation/quant, W2 and combine nodes in both arms;
- full-server accuracy, request success and losing/unsupported stock fallback;
- raw alternating restart order and all four finite estimators per series.

TP4, NVFP4, synthetic weights or config-only checkpoint data must remain
separately labeled diagnostics and cannot satisfy this lane.
