# GLM-5.2 indexer-score prefill reachability

Status: source-locked and runtime-confirmed by campaign
`20260723T134307Z`; every paired and profiler step completed on one B200
lease.

## Frozen source and deployment contract

- Kernel-Harness base: `bcd0054`
- SGLang base: `f93f8867b`
- model config snapshot:
  `/home/qinhaiyan/.cache/huggingface/hub/models--nvidia--GLM-5.2-NVFP4/snapshots/aec7243e916f585f4d52b97e4530f9a9750b0648/config.json`
- architecture: `GlmMoeDsaForCausalLM`
- B200/SM100, one-node production topology TP8/DP8/EP8
- balanced prefill: global chunk 32768, local DP query count `M=4096`
- hidden 6144, indexer Q LoRA rank 2048, 32 index heads, head dimension
  128, index top-k 2048, page size 64
- 78 layers: 21 `full` indexer producers and 57 index-sharing consumers

The local model cache has only configuration files, not model weights or a
tokenizer. Consequently, the exact rank-local production ABI is runnable here,
but a real eight-rank model-server acceptance run requires an external model
artifact and eight visible B200s. That gate is not weakened or renamed.

## Reached path

For a no-flag SM100 launch:

1. `arg_groups/overrides.py::_dsa_kv_cache_dtype_default` resolves DSA KV
   cache to `fp8_e4m3`.
2. `arg_groups/overrides.py::_dsa_split_backend_resolution` resolves both
   DSA prefill and decode to `trtllm`.
3. `dsa_backend.py::get_topk_transform_method` therefore selects `PAGED`,
   because `RAGGED` is reserved for the FP8 `flashmla_sparse` extend path.
4. `Indexer.forward_cuda` uses fused `wq_b` plus BF16
   `wk_weights_proj`, fused Q RoPE/quantization and fused K
   norm/RoPE/cache-store.
5. Non-CP prefill reaches `Indexer._get_topk_ragged`, which gathers the
   page-64 index K+scale cache via `GetKAndS.execute`, calls
   `deep_gemm.fp8_mqa_logits(..., clean_logits=False)`, applies any
   init/local masks, and calls SGL-Kernel's fused PAGED top-k transform.
6. The selected containing DSA consumer is
   `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` with
   `is_prefill=True` in the production backend.

The method name `_get_topk_ragged` describes its variable-length input
contract; it does not imply the RAGGED top-k transform. The reached default
here uses the PAGED transform.

Under piecewise or breakable CUDA graphs, GLM-5 prefill dispatches the whole
dynamic indexer through `pcg_dsa_indexer_prefill_split` or
`bcg_dsa_indexer_prefill_split`. The split region itself is eager. The
serving-native score and containing-region workloads reproduce that inner
execution. A successful inner result still requires full PCG/BCG and server
acceptance before promotion.

## Branch matrix

| Branch | Condition | Reached action | Evidence |
|---|---|---|---|
| short eager extend | `ForwardMode.EXTEND`, max sequence length `<=2048`, no PCG/BCG split | skip score/top-k logits work; keep K-cache preparation | actual `_should_skip_logits_computation` result emitted in runtime metadata |
| graph split short extend | PCG/BCG split surface | outer graph dispatches the eager split; the split op re-evaluates the condition and takes its K-only path | `Indexer.forward_cuda` plus `pcg_dsa_indexer_prefill_split` source branches |
| main c64 | 8 local requests × extend 512, sequence 8192 | `M=4096`, `K=65536`, 1 GiB logits; full unchunked MQA | workload metadata plus Nsys |
| max-concurrency c256 | 32 local requests × extend 128, sequence 8192 | `M=4096`, `K=262144`, 4 GiB logits; memory-budget chunking | workload metadata plus Nsys |
| mixed contexts | 16 requests, extend sum 4096, contexts 2K–32K | `K=241664`; nonuniform `ks/ke`; memory-budget chunking | workload metadata plus Nsys |
| PP=1 | no concurrent pipeline receive | all 148 B200 SMs available to DeepGEMM | runtime metadata/source |
| PP>1 non-last rank | `logits_with_pp_recv=True` | reserve one SM through `configure_deep_gemm_num_sms` | `_with_real_sm_count` source; not promoted from the PP1 benchmark |

## Exact tensor and metadata ABI

| Value | Contract |
|---|---|
| Q | E4M3 `[M,32,128]`, produced by `fused_q_indexer_rope_first_quant` |
| head gates | FP32 `[M,32,1]`, with Q scale and both softmax/head factors folded in |
| index K cache | fused page-64 byte layout: 128 FP8 bytes plus one FP32 scale per token |
| gathered K | E4M3 `[K,128]`; scale FP32 `[K]` |
| score output | FP32 dense logits `[chunk_rows,K]` |
| ranges | per-query cumulative `ks` and causal `ke`; expanded sequence lengths increase once per extend token |
| top-k output | int32 `[M,2048]`, PAGED request-local cache indices |
| stream | caller's current stream; producer uses its existing alternate stream only for the fused dual-stream preparation |

## Memory guard

Production caches:

`min(free_cuda_bytes*0.2, total_cuda_bytes*(1-mem_fraction_static)*0.2,
total_cuda_bytes*0.3)`.

For B200 total memory 191,490,555,904 bytes and
`mem_fraction_static=0.92`, the static cap is 3,063,848,894 bytes.
The runtime metadata records both the cached budget and resulting chunk rows.
The experiment may reduce a chunk size but never exceed the cached stock
budget, so the OOM guard remains conservative.

## Workload mapping

| Workload | Scope |
|---|---|
| `indexer_score_prefill_m4096` | exact K gather + unchunked score + fused top-k |
| `indexer_score_prefill_m4096_c256` | exact K gather + chunked score/top-k at global request concurrency 256 |
| `indexer_score_prefill_m4096_mixed` | exact variable-context chunked score/top-k |
| `tp4_indexer_score_prefill_m4096_c256_diagnostic` | four-rank replicated local ABI, maximum-rank timing; diagnostic only |
| `indexer_complete_prefill_m4096{,_c256,_mixed}` | packed-UE8M0 `wq_b`, BF16 `wk_weights_proj`, fused Q/K preparation/store, score/top-k |
| `indexer_graph_split_prefill_m4096{,_c256,_mixed}` | exact mutating PCG/BCG split-op surface with its single-stream preparation contract |
| `indexer_dsa_prefill_m4096{,_c256,_mixed}` | complete indexer plus selected FlashInfer TRT-LLM DSA consumer |

The frozen synthetic `testbench/tasks/glm52/index_score_prefill` remains
unchanged and is not used as production evidence.
