# GLM-5.2 DSA decode value-path report

Disposition: **no replacement**.  The source experiment is numerically correct
and reduces combine bookkeeping, but neither production CUDA-graph bucket meets
the 3% paired-p50 gate.  No candidate dispatch was added; stock
`flashmla_kv` remains active for M16, M32, and every fallback case.

## Reachability and fixed ABI

The user-supplied production B300 trace contains
`flash_fwd_splitkv_mla*` followed by the FlashMLA combine kernel, which selects
the `flashmla_kv` path rather than TRT-LLM.  In the current SGLang source, the
plan's `DSAAttentionBackend` entry label corresponds to the literal class
`DeepseekSparseAttnBackend`; its `_forward_flashmla_kv` method imports and calls
`sgl_kernel.flash_mla.flash_mla_with_kvcache`.  The explicit deployment launch
must include:

```text
--attention-backend dsa
--kv-cache-dtype fp8_e4m3
--dsa-prefill-backend flashmla_sparse
--dsa-decode-backend flashmla_kv
--tp 8 --dp 8 --enable-dp-attention
```

The prefill flag is part of the ABI: if either DSA phase remains TRT-LLM, SGLang
selects the incompatible raw 576-byte cache.  With no backend flags, current
SM100 FP8 resolution selects TRT-LLM; the retained
`../dsa-trtllm-stock-v0612/` trace is negative evidence, not a target result.

The added workloads are separately named `dsa_flashmla_kv_decode_m16` and
`dsa_flashmla_kv_decode_m32`.  The extracted TRT-LLM workload block and runner
block are byte-identical to parent commit `245ff19^` (SHA-256
`ad396ee98906637c389edfba46bfafa3cee31847f21edf91e41ad311af42a83a` and
`4f7474be37a6b0edb86f84ec5c4bb8ff1034b4d09d94606ba04c60c932229c21`,
respectively); TRT-LLM was neither relabelled nor optimized.

The local traces in
`../dsa-flashmla-kv-stock/analysis/nsys_driver_m16.json` and
`../dsa-flashmla-kv-stock/analysis/nsys_driver_m32.json` directly invoke that
production symbol and freeze its exact M16/M32 ABI.  They do not themselves
claim a full exact-shape SGLang backend-method hit; that reachability comes from
the supplied production trace, current code mapping, and the separate SGLang
backend-hit regression described below.  The direct calls are:

| Tensor/value | M16 | M32 |
|---|---:|---:|
| query | BF16 `[16,1,64,576]` | BF16 `[32,1,64,576]` |
| scaled FP8 KV cache | `[2048,64,1,656]` | `[4096,64,1,656]` |
| sparse physical indices | int32 `[16,1,2048]` | int32 `[32,1,2048]` |
| selected cache lengths | 16 values of 2048 | 32 values of 2048 |
| scheduler metadata | int32 `[148,8]` | int32 `[148,8]` |
| cumulative split counts | `[0,8,...,128]` | `[0,4,...,128]` |
| output | BF16 `[16,1,64,512]` | BF16 `[32,1,64,512]` |

The block table is an empty int32 `[M,0]` tensor, page size is 64, and the score
scale is `1/sqrt(192+64) = 0.0625`.  Each 656-byte cache row contains 512 FP8
latent bytes, four FP32 block scales (16 bytes), and 64 BF16 RoPE values
(128 bytes).  Input creation, quantization, and metadata construction are outside
the timed call, as in production graph setup.

The stock package is `sglang-kernel 0.4.4`.  Its extension SHA-256 is
`d8d97150bd86381c73406603cb7d6b682767535e0526053f04e3acefadb13316`;
the SGLang base is `f93f8867b4bc124c9809c9110ec7361ed11b6b4a`, FlashMLA is pinned at
`05e26647fe840b8baedae486c2d86d5ce4efeb7c`, and its CUTLASS submodule is
`147f5673d0c1c3dcf66f78d677fd647e4a020219`.

Nsight Systems confirms this SM100 kernel pair:

- main: `sm100::decode::head64::flash_fwd_splitkv_mla_fp8_sparse_kernel`
  (grid `[1,148,1]`, block 384, 168 registers/thread, 232,656 bytes shared);
- stock combine:
  `flash_fwd_mla_combine_kernel<bf16,512,8,160,256>` (block 256,
  grid `[M,1,8]`), launched with PDL overlap.

## Reference noise floor

The identity candidate calls the same stock reference.  Each row is a separate
same-session alternating A/B run; latency is microseconds and speedup is the
median of paired ratios.

| Bucket/run | Ref p50 | Identity p50 | paired p50 | paired p10–p90 |
|---|---:|---:|---:|---:|
| M16/1 | 32.688 | 32.544 | 1.0029× | 0.9757–1.0441× |
| M16/2 | 32.576 | 32.528 | 1.0005× | 0.9546–1.0568× |
| M16/3 | 32.608 | 32.624 | 0.9956× | 0.9503–1.0618× |
| M32/1 | 56.112 | 56.736 | 0.9829× | 0.8406–1.1914× |
| M32/2 | 46.368 | 46.448 | 1.0041× | 0.9289–1.0670× |
| M32/3 | 46.480 | 46.864 | 0.9900× | 0.9356–1.0570× |

The M32 run-1 session is an outlier affecting both arms.  This broad lower-tail
noise is why the decision uses repeated paired p50 and the fixed 3% threshold,
not a favorable minimum or one short probe.  Raw JSON files are under
`../dsa-flashmla-kv-stock/analysis/baseline_m*_run*.json`.

## Profiler diagnosis

The split-KV main kernel is the dominant value-path work.  Full NCU reports show:

| Metric | M16 main | M32 main |
|---|---:|---:|
| NCU duration | 22.272 µs | 30.048 µs |
| DRAM bytes read | 22.881 MB | 45.688 MB |
| DRAM read rate | 1.027 TB/s | 1.521 TB/s |
| SM throughput | 14.63% | 20.03% |
| eligible warps/cycle | 0.216 | 0.293 |
| long-scoreboard / issue-active | 5.235 | 4.272 |
| barrier / issue-active | 3.683 | 2.044 |

It launches only one resource-limited wave (148 CTAs on 148 SMs) and has no
local-memory spill loads or stores.  Sparse KV reads, long-scoreboard latency,
and cross-warp barriers bind the fused score/softmax/value work; output stores
are not the primary limit.

The stock combine pass reads approximately 16.8 MB of FP32 partial output in
both buckets.  It has low wave count (0.173 waves/SM at M16, 0.346 at M32), 48
registers/thread, and is long-scoreboard dominated.  The template selected
`MAX_SPLITS=160` from 148 device-wide scheduler parts even though the observed
requests have only 8 splits at M16 and 4 at M32.

## Source experiment

Hypothesis: specialize combine using a safe request-capacity bound.  Every
scheduler split advances by at least one 64-token block, so a request with a
2,048-entry top-k tensor cannot exceed 32 splits.  The patch adds a
`max_num_splits` launch field, computes
`min(num_sm_parts, ceil(topk/64) + ceil(extra_topk/64))` without a device read or
synchronization, and preserves the old `num_sm_parts` bound for dense decode.
Its safety precondition is the existing API contract that dynamic
`topk_length <= topk` and `extra_topk_length <= extra_topk`; out-of-capacity
lengths were already invalid inputs.

The exact patch is
`source/0001-experiment-bound-sparse-decode-combine-splits.patch`; build pins,
commands, commits, import isolation, and artifact hashes are in `BUILD.md`.
The candidate is confirmed as
`flash_fwd_mla_combine_kernel<bf16,512,8,32,256>`.  It removes no KV/value
traffic and adds no adapter kernel.

NCU and SASS show that the intended bookkeeping reduction occurred:

| Combine metric, M16 | Stock MAX160 | Candidate MAX32 |
|---|---:|---:|
| dynamic/shared bytes per block | 6,144 | 2,048 |
| registers/thread | 48 | 48 |
| shared-store instructions (NCU) | 5,120 | 1,024 |
| static SASS instructions | 359 | 279 |
| DRAM bytes read | 16.818 MB | 16.817 MB |
| long-scoreboard / issue-active | 14.75 | 18.08 |
| NCU replay duration | 10.656 µs | 10.528 µs |

Thus shared bookkeeping drops sharply, but the fixed partial-output stream
remains and memory latency becomes an even larger fraction.  The NCU duration
improves only 1.2%.  The candidate artifact is a full rebuild with nvcc 13.2,
so main-kernel differences can also include compiler variation; the selected
combine template and its NCU/SASS deltas are the causal source evidence.

Steady nsys medians provide the whole device-chain delta:

| Bucket/path | main | combine | overlap | chain | combine tail |
|---|---:|---:|---:|---:|---:|
| M16 stock | 17.600 | 12.704 | 4.288 | 26.080 | 8.576 |
| M16 candidate | 17.376 | 11.936 | 4.192 | 25.152 | 7.744 |
| M32 stock | 24.960 | 9.696 | 4.032 | 30.624 | 5.632 |
| M32 candidate | 24.704 | 9.728 | 4.128 | 30.272 | 5.600 |

All values are microseconds.  The M16 device chain improves 3.69%, but M32 only
1.16%; host/graph fixed costs further dilute both.

## Paired eager result

The isolated `sgl_kernel_goal01` namespace lets the installed reference and
candidate alternate in one process.  Every run performs correctness first and
contains 100 A/B pairs after ten warmups.

| Bucket/run | Ref p50 µs | Candidate p50 µs | paired p50 | paired p10–p90 |
|---|---:|---:|---:|---:|
| M16/1 | 44.160 | 42.560 | 1.0446× | 0.9761–1.1104× |
| M16/2 | 44.768 | 42.896 | 1.0393× | 0.9797–1.1001× |
| M16/3 | 44.416 | 42.592 | 1.0417× | 0.9748–1.1489× |
| M32/1 | 56.432 | 54.336 | 1.0363× | 0.9811–1.1062× |
| M32/2 | 49.168 | 47.424 | 1.0300× | 0.9596–1.1158× |
| M32/3 | 49.104 | 47.632 | 1.0271× | 0.9654–1.1124× |

M16 clears 3% in eager p50 three times.  M32 does not: run 3 is below the gate.
Neither eager observation is sufficient because serving decode uses CUDA graph
replay.

## CUDA-graph result and correctness

`graph_validate.py` captures separate real `torch.cuda.CUDAGraph` objects for
stock and candidate, alternates replays, mutates the captured query storage,
and compares outputs after replay.  All six runs have bit-identical BF16 output
(`max_abs_diff = 0`); the mutation changes output by `8.77e-5` at M16 and
`9.92e-5` at M32, proving replay is not returning a stale capture-time tensor.

| Bucket/run | Ref p50 µs | Candidate p50 µs | paired p50 | paired p10–p90 |
|---|---:|---:|---:|---:|
| M16/1 | 30.592 | 30.672 | 0.9954× | 0.9470–1.0508× |
| M16/2 | 30.688 | 30.736 | 1.0000× | 0.9491–1.0579× |
| M16/3 | 30.400 | 30.304 | 0.9957× | 0.9444–1.0605× |
| M32/1 | 35.920 | 36.064 | 1.0009× | 0.9550–1.0415× |
| M32/2 | 35.776 | 35.872 | 0.9982× | 0.9625–1.0542× |
| M32/3 | 36.256 | 36.256 | 1.0004× | 0.9714–1.0291× |

No graph run approaches the 1.03 promotion threshold.  Raw outputs are
`analysis/graph_m{16,32}_run{1,2,3}.json`.

The custom operator also passed SGLang's model-free
`TestDSAAttentionBackendCorrectness.test_sparse_fp8_cuda_graph_decode_case`.
`analysis/sglang_region_candidate.json` records three confirmed custom-op hits
through cache population, projections, DSA metadata lifecycle, backend call,
value output, and output projection.  That fixture uses M1/prefix128/top-k128,
synthesizes indices, and does not create a real CUDA graph; it complements but
does not replace the exact-shape graph test above.

## Bucket policy, containing region, and end-to-end result

| Bucket | Eager gate | Graph gate | Policy |
|---|---|---|---|
| M16 | passes repeated p50 | fails all repeats | stock |
| M32 | one repeat below 3% | fails all repeats | stock |

No static oracle or production integration was added.  The local SGLang change
only permits a build-time alternate operator namespace; default compilation and
runtime dispatch remain unchanged.

The complete GLM-5.2 indexer-score/top-k-to-attention region was not runnable:
the model-free fixture synthesizes top-k indices.  The required TP8/DP8/EP8
end-to-end gate is externally blocked because the host has four B200 GPUs and
the scheduler authorized only physical GPU 1 (logical GPU 0).  The four-GPU
diagnostic was also not run because using GPUs outside that allocation would
violate the session constraint.  These facts are recorded in
`analysis/validation_blockers.json`; the eight-rank gate was not weakened,
relabelled, or inferred from a smaller lane.

Because the candidate already fails the production graph gate, the absent
distributed end-to-end run cannot change this session's no-replacement
decision.  Stock fallback remains the only active policy.

## Attempt ledger and rollback

1. A first draft used context 8192 as FlashMLA's selected length and
   `1/sqrt(576)`.  It was rejected before profiler/candidate work; retained
   files are isolated under the stock profile's rejected-preflight directory.
2. The no-flag SM100 path was traced and found to be TRT-LLM, not the requested
   backend.  Those traces are retained but excluded from all performance claims.
3. The scheduler-bounded combine experiment was built and fully measured.  It
   is correct and locally reduces combine work, but graph replay is neutral, so
   it is rejected for deployment.

Rollback is immediate: do not load `artifacts/flashmla_goal01_ops.so` and retain
the installed `sgl_kernel` namespace.  That is the state left by this session.

## Repository validation

- `.venv/bin/python testbench/bin/check_env.py`: passed on one visible B200,
  SM100, PyTorch `2.11.0+cu130`.
- `.venv/bin/python serving_native/selftest.py`: 41 fixed workloads, passed.
- `python3 testbench/bin/selftest.py`: 24 tasks, zero problems.
- Both new `serving_native/run.sh --describe` contracts: passed.
- Candidate/profile script `py_compile` and both repository `git diff --check`
  lanes: passed.
- Exact custom extension build: passed; stock and custom namespaces co-loaded.
- SGLang stock regression method: one test passed; custom-op wrapper regression:
  one test passed with three candidate hits.
- `python3 testbench/bin/verify_harness.py`: exit 0.  Its normal pointer audit
  reported only the advisory absence of historical `runs/index.jsonl`; it
  audited no persisted frozen-task results in this isolated worktree.
- Knowledge entry lint, generated index check, and distillation check: passed
  after appending the no-win recipe.
