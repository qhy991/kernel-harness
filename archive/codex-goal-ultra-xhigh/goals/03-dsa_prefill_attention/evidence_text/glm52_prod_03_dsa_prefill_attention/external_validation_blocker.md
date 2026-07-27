# External production-validation blocker

Date: 2026-07-22

## Blocking facts

1. `/mnt/OS-oKqEXySb/models/GLM-5.2-NVFP4` contains no files. No other cached
   GLM-5.2 checkpoint/config/tokenizer payload was found.
2. The host has four physical NVIDIA B200 GPUs. Production acceptance requires
   one node with eight ranks in the fixed TP8/DP8/EP8 configuration.
3. Downloading/provisioning a multi-hundred-gigabyte model and supplying an
   eight-B200 node are external deployment actions, not evidence that can be
   manufactured by the kernel worktree.

All four local B200s were used through `with_all_gpus_lock.sh` for the separately
named DP4 diagnostic. This supersedes the inherited statement that only one GPU
was authorized. It does not weaken or satisfy the eight-rank gate.

## What remains unvalidated

| Production evidence | Status |
|---|---|
| Real-request backend/callable trace with exact launch flags | blocked by missing checkpoint |
| Live local-token, request-packing, context, and top-k distributions | blocked by missing checkpoint |
| Model projections and live indexer score/top-k | blocked by missing checkpoint |
| Complete DSA region baseline and candidate comparison | blocked; no runnable model, and no leaf candidate won |
| Actual scheduler/stream and graph/overlap behavior | blocked by missing server request |
| Complete SGLang prefill/end-to-end metric | blocked by missing checkpoint |
| TP8/DP8/EP8 correctness and rank-max latency | blocked by four-GPU host |
| Eight-rank graph/overlap and end-to-end gate | blocked by four-GPU host and missing checkpoint |

No result in this session is relabeled to fill those cells.

## Useful local evidence completed

- Source resolution and a runtime backend-class hit counter prove
  `DeepseekSparseAttnBackend.forward_extend -> _forward_trtllm ->
  trtllm_batch_decode_with_kv_cache_mla(backend="trtllm-gen")` for the
  checkpoint-free GLM-shaped fixture.
- The hit trace records the real raw-pool physical ABI: FP8 query
  `[4096,1,64,576]`, FP8 KV `[513,1,64,576]`, physical top-k table
  `[4096,1,2048]`, clipped lengths 2048, maximum context 32768, eager mode,
  and finite BF16 output.
- Three paired backend-class stock controls and an NSYS containing-region
  profile cover fused RoPE/FP8 conversion, raw KV writes, and attention.
- The exact 513-page leaf has three paired controls, three Q32 runs, three Q16
  runs, NSYS for all variants, and full NCU for all variants in one allocation.
- The DP4 rank-local diagnostic has three rank-max stock controls and a bundled
  four-process NSYS report.
- The source overlay is rejected decisively; stock SGLang remains active.

## Required external lane

If this path is revisited:

1. Provision and record an immutable GLM-5.2 checkpoint/config revision.
2. Launch the current stock SGLang tree on eight B200 ranks with TP8/DP8/EP8,
   `SGLANG_GLM52_OPT=0`, and the intended production flags.
3. Issue a real local-M4096 prefill request and capture the resolved backend,
   callable, dtypes/layouts, request/top-k distributions, cache metadata, graph
   mode, streams, and dominant kernels.
4. Record at least three uncontended stock baselines, using maximum latency
   across ranks, for the leaf, complete DSA region, and end-to-end prefill.
5. Only a future candidate that first clears the 3% paired leaf gate should run
   the same correctness, graph/overlap, containing-region, and end-to-end
   comparisons. Unsupported buckets must remain on stock.

The rejected PDL and Q32/Q16 tactics do not merit provisional enablement while
waiting for that lane.
