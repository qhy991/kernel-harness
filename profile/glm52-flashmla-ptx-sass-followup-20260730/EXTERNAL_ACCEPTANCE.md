# External acceptance: not authorized for this candidate

The terminal disposition of this campaign is `no-replacement`. The eager
containing SGLang DSA region lane fails locally at 0.664–0.758, so **this
candidate must not be enabled to seek an external override**. The commands below
are retained only so a future campaign that clears every local lane does not have
to reconstruct them.

Production default stays off. Stock remains the fallback for every bucket. The
registration ceiling for this op remains L1, explicit diagnostic.

## What must change before external acceptance is meaningful

The failing lane is an API-v1 integration property, not a kernel property: a
control whose kernel is source-identical to upstream stock fails it at
0.620–0.696 with a constant +17.4 µs of host Python, and the lane's ceiling is
1.0179 even with a zero-cost guard. Unblocking it needs an integration change,
not a kernel change. In descending measured value:

| Change | Measured saving |
|---|---:|
| Skip the NVTX context manager and its range-name construction entirely when NVTX is disabled | 1.86 µs |
| Cache the resolved `KernelSpec` per `(op, phase, m)` alongside the existing `load_manifest` cache | 2.96 µs |
| Precompute the ABI guard's expected shape and stride tuples per spec instead of per call | part of 4.82 µs |
| Move the fail-closed guard to the C++ side, or select the provider only under graph capture | removes the remainder |

The first three together do not reach the lane's requirement; only the last one
does. None of them may weaken the fail-closed guard's semantics, and all nine
SGLang GLM-5.2 hotspot registry tests must still pass.

## Selector and fallback policy

Selection stays fail-closed and explicit. The bare `hotspot_candidates` profile
must not select this op; `SGLANG_GLM52_OPT_OPS=flashmla_sparse_decode` is
required. The guard admits only `ForwardMode.DECODE` with local M 16 or 32, Q
BF16 contiguous `[M,1,64,576]`, FP8 E4M3FN contiguous paged KV with exactly 2049
pages at M16 and 4097 at M32, int32 `[M]` cache lengths, int32 `[M,1,2048]`
sparse indices, int32 `[148,8]` metadata, int32 `[M+1]` cumulative splits, an
empty int32 `[M,0]` block table, value dimension 512, FP8 KV enabled, softmax
scale exactly 0.0625, and all tensors on one CUDA device. Everything else returns
to stock **before** any candidate launch. After selection, provider errors are
fatal and stock is never executed afterwards.

## Commands, for a future candidate that clears every local lane

Single-operator A/B, one arm per fresh server process:

```bash
export SGLANG_GLM52_OPT=1
export SGLANG_GLM52_OPT_PROFILE=hotspot_candidates
export SGLANG_GLM52_OPT_OPS=flashmla_sparse_decode
export SGLANG_GLM52_OPT_M_BUCKETS='dsa_decode_attn:16|32'
export SGLANG_GLM52_HOTSPOT_MODULE=/abs/path/to/flashmla_sparse_decode_provider.py
export GLM52_FLASHMLA_VARIANT=<variant>
export SGLANG_GLM52_INFINI_KERNEL_NVTX=0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m sglang.launch_server \
  <unchanged GLM-5.2 arguments> \
  --dsa-decode-backend flashmla_kv
```

Run the stock arm with `SGLANG_GLM52_OPT=0` and everything else identical.

Profiler confirmation (turn NVTX back off for authoritative latency):

```bash
export SGLANG_GLM52_INFINI_KERNEL_NVTX=1
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
  --output=glm52-flashmla-external \
  python -m sglang.launch_server <unchanged arguments> --dsa-decode-backend flashmla_kv
```

Expect the NVTX range
`infini_kernel_glm52_flashmla_sparse_decode_fp8_topk2048[M=16]` and a CUDA symbol
beginning `infini_kernel_glm52_flashmla_sparse_decode`. A provider-ready startup
log without a hit counter is not evidence that the candidate ran.

## What must be recorded externally

1. Checkpoint output correctness, stock versus single-operator candidate.
2. Per-rank hit and fallback counts across TP8/DP8/EP8, and rank-max region
   latency.
3. TTFT, TPOT and throughput on a fixed request set with a fixed seed, plus graph
   capture/replay status.
4. Nsys evidence of the actual `infini_kernel` symbol, the target graph node, and
   the complete device critical span.

A whole-model result of 1.00x, a slight regression, or run-to-run noise is a
valid conclusion and must be recorded as such rather than replaced by a citation
of the leaf or graph-lane microbenchmark.
