# DSA prefill reachability

Date: 2026-07-22

## Decision

For the current GLM-5.2/B200 default configuration, DSA prefill resolves to the
FlashInfer wrapper for TRT-LLM generated sparse MLA. A checkpoint-free runtime
fixture reaches the real `DeepseekSparseAttnBackend.forward_extend` method and
records the exact leaf ABI with a scoped hit counter.

A real SGLang request was not possible because the configured model directory
is empty. Therefore live launch overrides, request/index distributions, full
model region, scheduler, and eight-rank behavior remain unconfirmed deployment
facts.

## Source chain

The trace is against the isolated SGLang worktree at
`5a444f66cf5764d2d76003a3a4c4631af152a253`. Its reached backend file has SHA256
`c55b55e166bb10407070ecfb00bbf0650d54dd1d804715560ca59a7920b01adb`,
identical to stock base `f93f8867b4bc124c9809c9110ec7361ed11b6b4a`.

1. `GlmMoeDsaForCausalLM` inherits the DeepSeek MLA model path in
   `python/sglang/srt/models/glm4_moe.py`.
2. B200 automatic DSA KV resolution selects FP8 E4M3, and the split backend
   resolver independently selects `trtllm` for prefill when no explicit
   override is supplied (`srt/arg_groups/overrides.py`).
3. `DeepseekSparseAttnBackend.__init__` stores
   `server_args.dsa_prefill_backend` as `self.dsa_prefill_impl`.
4. `forward_extend` selects that prefill implementation and calls
   `_forward_trtllm(..., is_prefill=True)` when `use_mha=false`.
5. `_forward_trtllm` fuses RoPE and BF16-to-FP8 query/key conversion, writes
   current K to the raw paged cache, consumes fused physical top-k indices, and
   invokes
   `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` with
   `backend="trtllm-gen"`.

Source resolution proves the no-override default. It does not prove that an
external deployment passed no override; that still requires the missing real
request.

## Backend-class runtime proof

The named `dsa_backend_prefill_m4096_ctx32768_fixture` uses SGLang's test-only
model-runner seam to build the real backend with GLM dimensions and a raw FP8
pool. A temporary wrapper around the FlashInfer leaf is installed for exactly
one call and restored in `finally` before performance timing.

`profile/dsa-backend-prefill-m4096-fixture-20260722/results/
backend_hit_trace_zeroed_v2.json` records:

| Item | Observed value |
|---|---|
| Python entry | `DeepseekSparseAttnBackend.forward_extend` |
| selected implementation | prefill `trtllm`, decode `trtllm`, `use_mha=false`, fused top-k |
| forward mode | extend/prefill |
| entry Q-nope | BF16 `[4096,64,512]` |
| entry Q-rope | BF16 `[4096,64,64]` |
| entry K-nope | BF16 `[4096,1,512]` |
| entry K-rope | BF16 `[4096,1,64]` |
| leaf hit count | exactly 1 |
| leaf backend | `trtllm-gen` |
| leaf query | FP8 E4M3 `[4096,1,64,576]` |
| raw paged KV | FP8 E4M3 `[513,1,64,576]` |
| physical sparse table | int32 `[4096,1,2048]` |
| clipped sparse lengths | int32 `[4096]`, min=max=2048 |
| maximum context | 32768 |
| workspace | 384 MiB, zeroed once before first use |
| execution | B200 CC10.0, eager, stream 0, not capturing |
| output | finite BF16 `[4096,1,64,512]` |

The Python stack captured at the leaf contains both `_forward_trtllm` and
`forward_extend` from the isolated SGLang source.

## Physical page correction

`DSATokenToKVPool(size=32768,page_size=64)` allocates `size + page_size`
physical slots. Page zero is reserved; usable tokens map to slots 64–32831.
Viewing the pool for TRTLLM therefore yields 513 pages, not 512.

The inherited direct workload used 512 compact pages and token indices starting
at zero. It remains useful as a compact replay, but it is not labeled the exact
backend pool. The separately named
`dsa_trtllm_prefill_m4096_ctx32768_rawpool` reproduces 513 pages, zeroes the
dummy page, and offsets physical indices by 64.

## Graph, stream, and topology

The backend hit is eager on the current stream and records
`is_current_stream_capturing=false`. This is consistent with current MLA/DP
prefill graph policy, but it is not a live server trace of an explicitly
overridden deployment.

The rank-local workload encodes attention TP1 inside intended DP8/TP8/EP8. Its
world-size-1 measurement isolates one rank. A separate world-size-4 workload
runs the same leaf on all four host B200s and uses maximum-rank timing; it is
explicitly DP4 diagnostic evidence only.

## What the fixture covers and excludes

Covered:

- real backend class and metadata construction;
- extend-length expansion and top-k clipping;
- fused RoPE/BF16-to-FP8 conversion;
- raw 576-byte FP8 paged-cache write;
- fused physical top-k path;
- exact TRTLLM-gen sparse MLA leaf and output ABI.

Excluded:

- model Q/K/V and absorbed-query projections;
- live indexer score and top-k selection;
- scheduler/request packing and radix-cache reuse;
- collectives and eight-rank topology;
- actual server graph/stream overlap and end-to-end latency.

The fixture's deterministic scattered top-k is causal and unique, but is not
presented as live indexer traffic.

## Frozen-task representativeness

The frozen `testbench/tasks/glm52/dsa_prefill_attn` task is mathematically
related but operationally non-representative. It uses flat BF16 Q/KV and
`sgl_kernel.flash_mla.flash_mla_sparse_fwd`, while the reached path uses FP8
paged KV, a 384 MiB workspace, physical sparse indices, and FlashInfer's
TRTLLM-gen kernel. No synthetic latency or speedup is used in this production
decision.
