# External production-validation blocker

Date: 2026-07-23

## Blocking facts

1. The configured model directory
   `/mnt/OS-oKqEXySb/models/GLM-5.2-NVFP4` contains no model files.
2. The Hugging Face cache for
   `nvidia/GLM-5.2-NVFP4@aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa`
   contains only `config.json` and `generation_config.json`; it has no
   tokenizer or weight shards.
3. This host exposes four B200 GPUs. The fixed production acceptance topology
   is one node with TP8/DP8/EP8.
4. The isolated repo-local environment has no `deep_ep` package. Importing a
   binary overlay built in another goal would break this goal's isolated
   provenance and would still not supply real model weights or eight ranks.

The previous sibling-goal attempt to start a full four-GPU dummy GLM server
failed before model load because GPU free memory was unbalanced. It is not
reused as this goal's evidence. This goal instead completed its own locked
four-rank, production-ABI full-indexer plus selected-DSA diagnostic.

## Unpassed gates

| Gate | Status |
|---|---|
| Real checkpoint-backed SGLang decode baseline | blocked by missing weights/tokenizer |
| Real-request scheduler, stream, and graph trace | blocked by missing checkpoint |
| Complete 78-layer SGLang decode metric | blocked by missing checkpoint |
| TP8/DP8/EP8 rank-max correctness and latency | blocked by four-GPU host |
| Eight-rank end-to-end candidate comparison | blocked; also unnecessary for the rejected inner candidate |

No synthetic, single-rank, or TP4 result is relabeled to fill these cells.
The blocker cannot hide a promotable candidate: CuTe-DSL already fails the
necessary repeated score, complete-indexer, selected-DSA, and TP4 diagnostic
gates.

## Useful local evidence completed

- exact backend resolution and runtime callable/ABI trace;
- exact M16/M32 score/top-k workloads;
- graph-replay score and exact top-k-set correctness;
- three paired score series per mode and bucket;
- Nsys and NCU score/top-k profiles;
- complete fused indexer preparation and page-64 K-store region;
- selected TRT-LLM DSA decode region;
- locked four-rank maximum-latency diagnostic.

## Required external lane

If a future candidate first clears the local repeated inner gates:

1. provision the immutable GLM-5.2 NVFP4 checkpoint and tokenizer;
2. install and record the production DeepEP build in the isolated environment;
3. launch the recorded SGLang revision on eight B200 ranks with TP8/DP8/EP8,
   DP attention, DeepEP, and `SGLANG_GLM52_OPT=0`;
4. capture three uncontended stock decode baselines for local M16 and M32;
5. repeat the same correctness, graph, containing-region, and rank-max
   end-to-end measurements with only the qualified bucket enabled.

Until then, stock fallback remains active.
