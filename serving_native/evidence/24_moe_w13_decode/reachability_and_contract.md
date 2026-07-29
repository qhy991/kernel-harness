# W13 decode reachability and dispatch contract

## Current production path

The task-worktree source and runtime traces establish this rank-local path:

```text
DeepEP low-latency dispatch
  -> fused packed-FP8 W13 grouped masked GEMM
  -> stock SwiGLU + packed UE8M0 quantization
  -> stock W2 grouped masked GEMM
  -> DeepEP low-latency combine
```

The optimized ownership boundary is only the first grouped GEMM.
Activation/quantization, W2, dispatch and combine remain stock.
This integration is committed locally at SGLang revision
`1c671bf3a30360100e7947c87e0c873a387ad0be`.

### Source reachability

1. `token_dispatcher/deepep.py:667-723` computes `expected_m`, calls the
   low-latency dispatch core, retains device-resident `masked_m`, waits through
   the production event/hook contract, and packages both fields into
   `DeepEPLLDispatchOutput`.
2. `moe_runner/deep_gemm.py:397-477` consumes that output, selects the packed
   masked W13 path, allocates the BF16 output and calls
   `grouped_gemm_nt_f8f8bf16_masked`.
3. `moe_runner/deep_gemm.py:507-596` immediately follows W13 with stock
   SwiGLU/packed quant and the stock W2 grouped GEMM.
4. `moe_runner/deep_gemm.py:937-1074` is the stock activation/quantization
   implementation retained by the containing-region comparison.
5. `deep_gemm_wrapper/entrypoint.py:66-149` first offers only the exact private
   W13 route. The generic `try_dispatch_moe_masked` path is explicitly skipped
   for exact W13 tensors and remains guarded by recipe/overlap compatibility.
6. `deep_gemm_wrapper/entrypoint.py:311-362` invokes the bounded initializer
   only after the worker's GPU assignment and stock DeepGEMM configuration.

The old generic `glm52_opt` path is not accepted as W13 evidence. It can return
`out`, enter generic hooks/precompile, and does not carry the private
production-decode marker. Exact W13 calls bypass it.

## Exact tensor and call contract

The private candidate accepts only:

- E=32, slab M=1024, K=6144, N=4096;
- FP8 E4M3 activation `[32,1024,6144]`, stride
  `[6291456,6144,1]`;
- packed `int32` activation scale `[32,1024,12]`, stride
  `[12288,1,1024]`;
- FP8 E4M3 weight `[32,4096,6144]`, stride
  `[25165824,6144,1]`;
- packed `int32` weight scale `[32,4096,12]`, stride
  `[49152,1,4096]`;
- BF16 output `[32,1024,4096]`, stride `[4194304,4096,1]`;
- device `int32` mask `[32]`;
- independently named expected-M 4, 5, 8 or 9;
- no recipes, no overlap arguments, `max_block_n=256`, sm_100, the assigned
  device, and a private exact-DECODE marker for token bucket 16 or 32.

`w13_decode.py:704-751` rechecks every condition at launch, performs exactly
one candidate call and requires its return to be exactly `None`. Candidate
errors propagate. There is no catch-and-stock retry, scale conversion,
allocation, D2H read, lock, file write, environment lookup, NVTX range or
statistics call on the selected path.

Unsupported metadata is selected to stock before launch. An explicitly
requested invalid variant, manifest, artifact or startup state fails closed
instead of generating a stock-labeled candidate result.

## Production graph marker

`DecodeCudaGraphRunner` calls `model.forward` directly, so a marker installed
only in `ModelRunner._forward_raw` is not production-reachable during graph
capture. The task instead adds the W13-private marker at the real forward
calls:

- `decode_cuda_graph_runner.py:964-977` for CUDA graph capture;
- `eager_runner.py:251-265` for eager production execution.

`w13_context.py:27-75` installs a token-reset `ContextVar` only for exact
DECODE M16/M32 and always restores it in `finally`. The runtime trace covers
all four expected-M values, both eager and graph markers, proves reset after
every forward, and proves that a marker-free exact tensor call selects stock:

- [`production_trace_bm32_1sm.json`](production_trace_bm32_1sm.json)
- [`production_trace_bm32_2sm.json`](production_trace_bm32_2sm.json)

Both traces show four selected calls, four candidate low-level calls, zero
stock calls during selection, exact `None` returns, a non-default stream, and
candidate-error propagation without retry.

## Startup, cache and runtime-state contract

Importing `w13_decode.py` performs no CUDA query, DSO load or cache mutation.
When and only when an opt-in variant is requested,
`update_deep_gemm_config()` calls
`initialize_w13_decode_after_assignment()` after the worker owns its GPU.

Startup order is deterministic:

1. set/read back installed DeepGEMM at PDL=true, `num_sms=148`,
   `tc_util=100`;
2. bind and import exact same-source stock using its private JIT root;
3. disable broad masked-GEMM precompile and warm only expected-M 4/5/8/9;
4. import SGLang `compile_utils`;
5. bind/import/warm the candidate using its distinct JIT root;
6. set/read back both side modules independently;
7. mutate, read back and restore each of PDL/SM-count/TC-util in both
   directions while proving the other module is unchanged;
8. poison-probe both compiler paths to prove cache ownership is frozen;
9. restore the caller's cache environment.

The runtime traces record package, DSO, cache and JIT artifact hashes. Stock
and candidate use distinct packages, DSOs and cache roots. The manifest SHA256
is `afa7063a860cd045138e51abcb5b8b44c226db13c08e212ae30da077f5655621`.

## Correctness and graph observation

[`correctness.json`](correctness.json) validates both genuine one-SM and
two-SM variants across expected-M 4/5/8/9 plus zero, minimum, maximum, skewed,
31/32/33 and 127/128/129 boundaries, random, deterministic-ramp,
extreme-finite, changed-input and poisoned cases. Both variants have zero
anomaly mismatches, zero failing elements, and maximum absolute/relative error
of exactly zero. Packed-scale hashes are unchanged, output ownership is
distinct, calls use a non-default stream, and masked regions outside the
variant's legal store tiles remain poisoned.

The graph-safe harness never calls `.tolist()` or performs a device-to-host
mask read in capture. CPU-known mask metadata is fixed before capture and the
full output is pre-poisoned. Each replay:

- mutates both activation data and the device `masked_m`;
- restores output poison;
- checks graph and tensor pointer stability;
- validates fresh output and exact deterministic replay;
- validates untouched masked regions;
- requires separate stock/candidate graphs and identical non-W13 region
  nodes after substituting only W13.

The containing-region graph result embeds all raw ordered pairs, graph
identities, capture/replay observations and per-phase candidate/reference call
counts:
[`results/region_m16_em4_graph_bm32_2sm.json`](results/region_m16_em4_graph_bm32_2sm.json).

## Default and rollback

The production default is stock. `SGLANG_GLM52_W13_DECODE_VARIANT` is empty by
default, and the private initializer reports `default_off`. The measured
candidate is not eligible for deployment because a mandatory local series
failed; leaving the variable unset is the complete rollback.
