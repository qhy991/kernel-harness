# GLM-5.2 indexer score decode reachability

## Fixed production lane

- Model: `nvidia/GLM-5.2-NVFP4` at local config revision
  `aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa`,
  `GlmMoeDsaForCausalLM`.
- Production topology: one B200 node, TP8/DP8/EP8 with DP attention. Local
  decode buckets remain M16 and M32; DP does not divide them.
- Fixed decode context: 8192; sparse/index top-k: 2048.
- Indexer config: 32 heads, head dimension 128, interleaved 64-dimensional
  RoPE, page size 64, UE8M0 per-head/per-token scales.

## Reached call path

For normal CUDA decode, producer layers execute:

```text
DeepseekMLAForwardMixin.forward_absorb
  -> Indexer.forward_cuda
  -> Indexer._fused_q_prepare_and_store
  -> Indexer._get_topk_paged
  -> deepgemm_paged_mqa_logits_split
  -> deep_gemm.fp8_paged_mqa_logits(clean_logits=False)
  -> topk_transform_512_v2
  -> selected DSA decode attention
```

`DSAPagedMQALogitsBackend.resolve("auto")` returns `DEEPGEMM` on CUDA. Normal
decode has `next_n=1`, so it does not enter the target-verify native
multi-token branch and instead uses the split wrapper (`q.unsqueeze(1)`).
Explicit `cutedsl` is a reachable SM100 configuration alternative.

The score ABI is FP8 E4M3 Q `[M,32,128]`, fused uint8 K cache
`[num_pages,64,1,132]`, FP32 head gates `[M,32]`, int32 compact page table
`[M,128]`, int32 sequence lengths `[M,1]`, and precomputed DeepGEMM scheduler
metadata. The fused K page stores 64×128 FP8 bytes followed by 64 FP32 scale
values. The output is a row-major FP32 `[M,8192]` score buffer.

The shipped config has no `index_init_tokens` or `index_local_tokens`, so both
default to zero and the scatter masking step is a no-op. `clean_logits=False`
is intentional: top-k-v2 consumes each row only through its sequence length.
Top-k-v2 jointly selects 2048 scores and maps logical positions through the
compact page-64 table to physical token slots.

## Skip classes and layer frequency

Normal decode never satisfies `_should_skip_logits_computation`; that shortcut
is restricted to `extend_without_speculative` with maximum KV length at most
2048. Model-level `index_topk_freq=4` and `index_skip_topk_offset=3` are a
different skip: 21 producer layers (0, 1, 2, then 6 through 74 every four
layers) run their indexer, while 57 shared layers reuse carried top-k indices
and bypass the indexer entirely. NextN target verification has a separate
native/split decision and is not evidence for normal decode.

## Graph, stream, and SM budget

Decode is captured in the production CUDA graph buckets. Q/K preparation uses
the fused dual-stream path during capture and the outer attention method
overlaps the indexer with attention Q-B projection. Score and top-k are ordered
on the indexer stream; top-k-v2 uses programmatic dependent launch support.

The official topology has PP=1, so all 148 B200 SMs are available. The paged
decode path builds its metadata with `self.sm_count`; the `_with_real_sm_count`
one-SM PP-receive reservation is used by other/nonpaged indexer paths, not
`_get_topk_paged`. This goal does not relabel a PP>1 result as the PP1 contract.

## Evidence boundary

The named serving-native workload reconstructs the exact rank-local score/top-k
ABI on deterministic production-shaped tensors. A runtime result must still
prove module origins, resolved backend, tensor strides, metadata shapes, and
graph replay. Complete indexer, selected DSA attention, and real SGLang request
measurements are separate containing gates; the rank-local workload alone is
not TP8 end-to-end acceptance.
