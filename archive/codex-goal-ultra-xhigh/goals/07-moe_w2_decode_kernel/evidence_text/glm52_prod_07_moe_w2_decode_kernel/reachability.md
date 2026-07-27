# Production W2 reachability

Date: 2026-07-22

## Current SGLang call chain

GLM-5.2 inherits the DeepSeek-v2 MoE implementation in
`python/sglang/srt/models/glm4_moe.py`. With DeepEP enabled, the model selects
`DeepEPMoE`, enters the low-latency dispatcher, and resolves the FP8 runner to
`DeepGemmMoeRunner`. The containing sequence is:

1. DeepEP low-latency dispatch;
2. fused W13 masked grouped GEMM;
3. fused SwiGLU plus production FP8/packed-UE8M0 quantization;
4. W2 masked grouped GEMM; and
5. DeepEP combine.

The exact W2 call is
`python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:551` into
`python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py:35`, which reaches
`deep_gemm.fp8_m_grouped_gemm_nt_masked`. No mathematically similar frozen
E8/f32-scale task is used as production evidence.

## Fixed rank-local ABI

| Field | M16 bucket | M32 bucket |
|---|---:|---:|
| decode tokens per rank | 16 | 32 |
| local experts | 32 | 32 |
| expert slab | 1024 | 1024 |
| plan `expected_m` | 4 | 8 |
| current-source-derived `expected_m` | 5 | 9 |
| activation | FP8 E4M3 `[32,1024,2048]` | same |
| activation scale | packed int32 UE8M0 `[32,1024,4]`, stride `[4096,1,1024]` | same |
| W2 weight | FP8 E4M3 `[32,6144,2048]` | same |
| weight scale | packed int32 UE8M0 `[32,6144,4]`, stride `[24576,1,6144]` | same |
| output | BF16 `[32,1024,6144]` | same |
| valid assignments | 128 | 256 |

The deterministic leaf masks are:

- M16: `[2,3,6,3,1,5,3,7,5,6,8,5,4,3,2,1,2,4,7,5,5,1,3,9,5,5,4,2,2,4,4,2]`
- M32: `[4,5,9,7,5,9,11,9,8,11,14,6,9,5,8,5,6,9,13,7,7,5,8,14,11,7,8,7,7,9,8,5]`

They sum to `M * topk` and are passed as actual device `masked_m` arguments.
They are production-ABI workload data, not live EP8 `packed_recv_count`.
Capturing the live masks remains an unchanged eight-rank gate.

DeepEP computes `expected_m` at `token_dispatcher/deepep.py:675`. Current
source adds `self.num_experts` rather than `self.num_experts - 1` before
division, so exactly divisible EP8 traffic gets one extra unit: 5/9 rather
than plan 4/8. The workloads remain separately named. The B200 dispatch asks
for rounded packed UE8M0 scales at `deepep.py:731`. Slab1024 requires
`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024`; the repository default
128 is not silently substituted.

## Runtime-confirmed leaf contract

The locked single-B200 campaign confirmed the exact wrapper, ABI, and selected
config on NVIDIA B200 UUID
`GPU-30b619de-87f2-1862-0d07-a595da8fe417`:

- `recipe_a=None`, `recipe_b=None`, and `overlap_args=None`;
- production PDL enabled and restored after each experiment;
- 148 active SMs;
- legacy default stream `cudaStream_t == 0`;
- eager mode with `is_current_stream_capturing=false`;
- stock alignment restored to 128 after every call; and
- stock and candidate both return `None`, matching the output-buffer contract.

The selected stock config is BM128/BN128/BK128, cluster 1x2, load-M64,
store-M16, eight stages, and 213804 configured shared-memory bytes. The
experimental config is BM16/BN128/BK128, cluster 1x2, load-M8, store-M16,
twelve stages, and 230188 configured bytes. Both report 1,536 logical tile
tasks, 11 logical scheduler waves, and a 56-task final wave. Because the kernel
is persistent, NCU observes a 148-block grid and one launch wave per SM.

The plan/current-source hints do not affect this selection: all 4/5/8/9
workloads choose the same config at a given alignment. Config JSON, exact
selection logs, and generated PTX/SASS/cubin identity are jointly captured by
the per-run `config_*.json`, log files, and
[`jit_inventory.json`](../../profile/moe-w2-alignment16/analysis/jit_inventory.json).
No individual result JSON is claimed to embed every generated artifact.

## Recipe, overlap, stream, and return contracts

Ordinary GLM-5.2 block FP8 reaches W2 with null recipes. On B200, routed W2
combine/down-GEMM two-stream overlap is disabled by
`single_batch_overlap.py:28`; the current leaf therefore has no overlap tuple
to feed combine and uses DeepGEMM's process-global SM allocation. Shared-expert
communication may reserve SMs in another enclosing region and is not evidence
that routed W2 does.

The wrapper's unsupported-path contract still matters. On a topology/package
that supports overlap, it scopes `deep_gemm.get_num_sms()` to
`overlap_args.num_sms`, forwards `enable_overlap`, `max_block_n`, and `signal`,
and returns DeepGEMM's `(block_m, threshold)` tuple to combine. Recipes and
overlap both bypass the GLM52 replacement. The four CPU regression tests in the
isolated SGLang worktree cover bypass, argument identity, SM scope, return
identity, replacement success, and fail-closed fallback. That SGLang commit is
test-only; production code is unchanged.

The pinned post1 grouped-masked Python signature lacks the overlap keywords.
This is latent for the current non-overlap B200 W2 call but prevents this host
from certifying a future overlap-enabled branch.

## CUDA Graph boundary

The microbenchmark records the current stream and eager capture state. The
separate alignment-16 graph suite captured the exact leaf with fixed production
pointers. All four workloads observed capture during launch, replayed 30 times
deterministically, matched eager output exactly, and preserved return semantics.
The authority is
[`leaf_validation_summary.json`](leaf_validation_summary.json), whose scope is
explicitly single-GPU leaf graph and edge correctness, not TP8 acceptance.

`glm52_opt.dispatch` needs real forward M because the tensor's physical M is
the 1024-row slab. Normal eager forward installs that context in
`model_runner.py:3153`, but decode graph capture calls `model.forward` directly
in `decode_cuda_graph_runner.py:928` and bypasses the setter. A future
GLM52-dispatch integration must repair and test that context. This does not
affect the isolated DeepGEMM alignment experiment, which receives `expected_m`
directly.

## Package and cache identity

The repository declares `sgl-deep-gemm==0.1.4.post1`, while the shared harness
venv contains distribution `0.1.4`. Measurements prepend an isolated post1
overlay under the isolated SGLang worktree; the venv is not modified. The
peeled upstream tag resolves to
`edcf77b276965de8f03cdc47c23f01b08bf7c7ab`. The pinned hashes are:

- Python package: `b33e89deacdce241f01f5070d321918f5e5480e3e6d3af569678d4192db4f2a7`;
- extension: `cd8beab174071777c972c5948af7706ae2cfb5d2adcdbb7e6fbea253ce3a81bf`; and
- SM100 1D1D device source: `9c1e70677ede6ba09ab98e629482da7874182f8227907382efe0a81658da5a37`.

Wheel, source tag, CUTLASS/fmt submodules, install command, artifact path, and
import resolution are in
[`stock_deep_gemm_provenance.json`](stock_deep_gemm_provenance.json).

## Topology boundary

The leaf ABI represents EP8 rank-local geometry, but a single-GPU invocation
proves only the exact leaf ABI. Four-rank DeepEP evidence remains
TP4/DP4/EP4 diagnostic evidence. Only an eight-rank dispatch -> W13 ->
SwiGLU+quant -> W2 -> combine run can satisfy the production region gate.
Because that gate and eight-rank SGLang end-to-end validation are unavailable,
the locally favorable alignment remains disabled and stock stays active.
