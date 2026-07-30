# External acceptance: authorized for `p1_consumer_scale`, M16 and M32

This round's terminal disposition is **external-acceptance-candidate**. Every
locally runnable gate passes: bitwise-exact correctness, both mandatory CUDA
Graph lanes at ≥1.03 against installed stock at both buckets, and the round-3
stock-fallback requirement on the eager containing region. The only remaining
gate is the checkpoint-backed TP8/DP8/EP8 lane, which this four-GPU host cannot
run.

This supersedes round-2's `EXTERNAL_ACCEPTANCE.md`, which correctly refused
authorization because a mandatory local lane failed. That lane was an API-v1
integration property; SGLang `d7fe89a71` removed it by making the hotspot
graph-only, and round-3 policy gates the fallback behaviour instead. The kernel
binary is unchanged from round 2.

**Production default stays off until external acceptance passes.** Success here
raises the ceiling to L2 external E2E, not to L3 production-default.

## What is being accepted

| Item | Value |
|---|---|
| Variant | `p1_consumer_scale` |
| FlashMLA commit | `b5af443` |
| SGLang commit | `d7fe89a71` |
| Buckets | local M16 and M32, independently |
| Symbol | `infini_kernel_glm52_flashmla_sparse_decode_p1_consumer_scale_main` |
| Combine | unchanged stock `flash_fwd_mla_combine_kernel` |
| Selection | CUDA graph capture only (`KernelSpec.graph_only`) |

Local graph-lane result to be confirmed or refuted externally: 1.0694–1.0728
(M16 containing), 1.0716–1.0760 (M16 leaf), 1.0636–1.0669 (M32 containing),
1.0628–1.0664 (M32 leaf).

## Selector and fallback policy

Selection stays fail-closed and explicit. The bare `hotspot_candidates` profile
must not select this op; `SGLANG_GLM52_OPT_OPS=flashmla_sparse_decode` is
required. The guard admits only `ForwardMode.DECODE` with local M 16 or 32, Q
BF16 contiguous `[M,1,64,576]`, FP8 E4M3FN contiguous paged KV with exactly 2049
pages at M16 and 4097 at M32, int32 `[M]` cache lengths, int32 `[M,1,2048]`
sparse indices, int32 `[148,8]` metadata, int32 `[M+1]` cumulative splits, an
empty int32 `[M,0]` block table, value dimension 512, FP8 KV enabled, softmax
scale exactly 0.0625, and all tensors on one CUDA device. Everything else
returns to stock **before** any candidate launch. After selection, provider
errors are fatal and stock is never executed afterwards.

Additionally, and specific to this round: outside CUDA graph capture the
dispatch returns before the ABI guard, so eager decode runs stock with no
provider launch. `SGLANG_GLM52_FLASHMLA_GRAPH_ONLY=0` forces eager selection and
is a **diagnostic only** — it must not be set for an acceptance run.

## Commands

Single-operator A/B, one arm per fresh server process:

```bash
export SGLANG_GLM52_OPT=1
export SGLANG_GLM52_OPT_PROFILE=hotspot_candidates
export SGLANG_GLM52_OPT_OPS=flashmla_sparse_decode
export SGLANG_GLM52_OPT_M_BUCKETS='dsa_decode_attn:16|32'
export SGLANG_GLM52_HOTSPOT_MODULE=/abs/path/to/flashmla_sparse_decode_provider.py
export GLM52_FLASHMLA_VARIANT=p1_consumer_scale
export SGLANG_GLM52_INFINI_KERNEL_NVTX=0
# leave SGLANG_GLM52_FLASHMLA_GRAPH_ONLY unset: the default is on

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m sglang.launch_server \
  <unchanged GLM-5.2 arguments> \
  --dsa-decode-backend flashmla_kv
```

Run the stock arm with `SGLANG_GLM52_OPT=0` and everything else identical.

Because selection is graph-only, the acceptance run **must** exercise CUDA graph
decode. A configuration that disables graph capture will legitimately show 1.00x
and is not a refutation of the kernel; it is out of the candidate's scope and
must be recorded as such.

Profiler confirmation (turn NVTX back off for authoritative latency):

```bash
export SGLANG_GLM52_INFINI_KERNEL_NVTX=1
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
  --output=glm52-flashmla-external-r3 \
  python -m sglang.launch_server <unchanged arguments> --dsa-decode-backend flashmla_kv
```

Expect a CUDA symbol beginning `infini_kernel_glm52_flashmla_sparse_decode`
inside the replayed graph. A provider-ready startup log without a hit counter is
not evidence that the candidate ran.

## What must be recorded externally

1. Checkpoint output correctness, stock versus single-operator candidate.
2. Per-rank hit and fallback counts across TP8/DP8/EP8, and rank-max region
   latency. Hits are expected only on graph-replayed decode steps.
3. TTFT, TPOT and throughput on a fixed request set with a fixed seed, plus
   graph capture/replay status.
4. Nsys evidence of the actual `infini_kernel` symbol, the target graph node,
   and the complete device critical span.
5. The eager decode path's behaviour: it must remain on stock, and the
   graph-only early return costs a measured 3–5 µs per call locally. Confirm
   that this does not regress a workload with a low graph-capture ratio.

A whole-model result of 1.00x, a slight regression, or run-to-run noise is a
valid conclusion and must be recorded as such rather than replaced by a citation
of the leaf or graph-lane microbenchmark. The local DSA share of full-server
short-decode GPU kernel time is about 5.1%, so a ~6.5% kernel win bounds the
achievable end-to-end effect at well under 1%.
