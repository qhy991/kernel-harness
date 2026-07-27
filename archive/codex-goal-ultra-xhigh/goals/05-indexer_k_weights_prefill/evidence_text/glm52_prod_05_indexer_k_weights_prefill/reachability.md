# GLM-5.2 indexer K/weights prefill reachability

## Fixed production lane

- Model: `GlmMoeDsaForCausalLM` /
  `nvidia/GLM-5.2-NVFP4@aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa`.
- Hardware/topology gate: one B200 node, TP8/DP8/EP8.
- Balanced recipe: NVFP4 checkpoint, `--tp 8 --dp 8
  --enable-dp-attention --chunked-prefill-size 32768`, which fixes the local
  prefill point at M=4096.
- SGLang source base: `f93f8867b4bc124c9809c9110ec7361ed11b6b4a`.
- Stock policy: `SGLANG_GLM52_OPT=0`; indexer fusion is enabled unless
  `SGLANG_DISABLE_DSA_INDEXER_FUSION=1`.

The balanced recipe is eager for this region. `ServerArgs` disables breakable
prefill CUDA graphs for MLA, MoE A2A, and DP attention. The eager
`Indexer.forward_cuda` branch calls `_fused_q_prepare_and_store` without
overriding its `enable_dual_stream=True` default. The nearby M<=1024 capture
gate controls different K-only/non-fused branches and does not control this
call.

## Fixed-model rank-local ABI: static mapping and reconstruction

`DeepseekAttentionMLA` constructs a CUDA `Indexer` and supplies the model-wide
alternate stream. For a 4096-token sequence, the static branch condition exceeds
`index_topk=2048`, so the K-only shortcut is not eligible. The reconstructed
rank-local method is:

`sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._fused_q_prepare_and_store`

| Value | Production rank-local ABI |
| --- | --- |
| `x` | BF16 `[4096, 6144]`, row-major |
| `q_lora` | BF16 `[4096, 2048]`, row-major |
| `wq_b` | unquantized BF16 `ReplicatedLinear`, BF16 `[4096,2048]` weight, output BF16 `[4096,4096]` |
| `wk_weights_proj.weight` | replicated BF16 `[160, 6144]` |
| fused projection output | BF16 `[4096, 160]`, split as strided key `[4096,128]` plus gate `[4096,32]` |
| Q output | FP8 E4M3 `[4096,32,128]` |
| gate output | FP32 `[4096,32,1]` |
| K destination | page-64 `uint8` cache; each token stores 128 FP8 bytes plus one FP32 scale |
| positions/cache locations | contiguous int64 `[4096]` in the serving-native reproduction |

The pinned config sets default interleaved RoPE with
`max_position_embeddings=1048576` and `rope_theta=8000000`; the index-K norm
type field is absent, so the Indexer constructs its default FP32 `LayerNorm`
with epsilon `1e-6`. The ModelOpt ignore list covers every `self_attn` module.
With `SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN` unset in the fixed launch,
`ModelOptFp4Config` resolves `indexer.wq_b` to
`UnquantizedLinearMethod`. The optional loader conversion list contains only
attention `q_b_proj`, not `indexer.wq_b`. The config dispatch and all three
checkpoint tensor headers are reproduced in `fixed_model_contract_cpu.json`.

The fused checkpoint loader copies `indexer.weights_proj` into rows 128:160.
It copies BF16 `indexer.wk` directly into rows 0:128 or dequantizes its
block-FP8 weight/scale pair to BF16 before copying. The fused parameter is
replicated rather than TP-sharded.

## Stock stream schedule

```text
current stream: wait/start ─ wk_weights_proj BF16 ─ wait(wq_b) ─ Q rope+FP8+gate ─ wait(K) ─ return
alternate stream: wait(current) ─ wq_b BF16 GEMM ─ wait(wk) ─ K LN+RoPE+quant+page64 store ─┘
```

Stage 1 overlaps the large `wq_b` projection with the narrow BF16 projection.
Stage 2 overlaps `fused_q_indexer_rope_first_quant` with
`dpsk_v32_k_indexer_norm_rope_store_p64`. The method's final stream wait makes
both returned Q/gate data and the cache mutation complete under the caller's
current-stream contract.

## LoRA scope

The fixed production recipe does not enable indexer LoRA. Indexer-targeted LoRA
requires disabling fusion because the separate `wk`/`weights_proj` modules do
not exist when folded into `wk_weights_proj`. Static review found a pre-existing
defect in this checkout: `lora_manager.py` imports module-global
`_use_dsa_indexer_fusion`, while fusion is now an `Indexer` instance attribute
and that module global is absent. This goal does not enable LoRA and does not
silently claim LoRA validation; any future LoRA deployment must fix and test
that fail-closed check independently.

## Runtime evidence status

The named serving-native workload exercises the real unbound scheduling method
with fixed deterministic synthetic tensors, SGLang `ReplicatedLinear`, RoPE,
`ForwardContext`, and page-64 cache objects. It is a world-size-1,
production-shaped rank-local reconstruction, not a full model-module request.
A locked TP4/DP1/EP1 dummy-weight attempt acquired all four GPUs but failed
SGLang's free-memory-balance check during distributed initialization, before
health, request, or profiler capture. Its resolved launch log showed
`attention_backend=dsa`, `dsa_prefill_backend=trtllm`,
`dsa_topk_backend=sgl-kernel`, and `kv_cache_dtype=fp8_e4m3`; these are static
server-argument resolution facts from a failed launch, not live backend
execution.

The corrected locked bundle uses TP4/DP4/EP4, DP attention, DeepEP, a global
chunk size of 16384 (4096 per DP rank), four explicitly routed concurrent
4096-token requests, and an Nsys trace gate that requires the local-M4096
wq/wk/Q/K grid and dual-stream mapping on all four devices. One corrected
allocation failed closed before CUDA server launch because a provenance check
compared a logical repo-venv path with its canonical installed-package path;
commit `95060f3` fixes that comparison while preserving the repo-local origin
gate. A fresh corrected request then made 180 locked wrapper attempts, all
returned exit 75 under shared-host contention, and never executed. The blocker
record is `tp4_live/20260722T181018Z-canonical_scheduler_blocker.json`.

Consequently, this report has no live TP4 route, score/top-k, attention, graph,
or performance evidence. The immutable world-size-1 campaign confirms the real
unbound callable and stream contract with exact synthetic tensors, not a live
request. The production TP8/DP8/EP8 request gate requires eight B200s and cannot
be weakened or relabeled on this four-GPU host. No candidate passed the prior
rank-local region gate, and stock fallback remains active.
