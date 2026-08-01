# GLM-5.2 B300 fusion audit

Date: 2026-08-01

## Decision

Eight phase-specific tasks were added without replacing the existing 28 tasks:

| task pair | why it belongs in Kernel Harness | production denominator |
|---|---|---|
| `indexer_q_rope_quant_{decode,prefill}` | repeated DSA Q transform whose side-output scale must remain observable | `fused_q_indexer_rope_first_quant` |
| `indexer_k_norm_rope_store_{decode,prefill}` | repeated K transform with a stateful paged-cache write | `fused_k_indexer_norm_rope_store` |
| `moe_swiglu_quant_{decode,prefill}` | largest non-GEMM compute kernel in the short/decode-heavy trace | `silu_and_mul_masked_post_quant` |
| `router_gemm_topk_{decode,prefill}` | B300 still launches router GEMM and top-k as adjacent kernels | FP32 `linear` then `moe_fused_gate` |

The first three production paths are already fused. They are tasks because there is
measurable residual headroom and a useful exact ABI to optimize, not because the
harness should time a slower decomposition. Router projection plus routing is the
new, still-open fusion boundary.

## Evidence read on B300-M2

Primary captures and reviews:

- `/mnt/b300-shared/home/qinhaiyan/wwxq/bench_results/nsys_op_single/`
- `/mnt/b300-shared/home/qinhaiyan/wwxq/bench_results/nsys_infini_hotspot_20260729T152819Z/`
- `/mnt/b300-shared/home/qinhaiyan/wwxq/bench_results/nsys_s32768_decode_winners_20260730T182858Z/`
- `/mnt/b300-shared/home/qinhaiyan/wwxq/bench_results/nsys_s32768_new_ops_no_opt0_20260801T011304Z/`
- `/mnt/b300-shared/home/qinhaiyan/wwxq/SGLang-DGMK/glm52_opt/default_path_ops_and_shares.md`
- `/mnt/b300-shared/home/qinhaiyan/wwxq/SGLang-DGMK/glm52_opt/e2e_gpu_kern_categories.md`

In the short-prompt, decode-heavy `nsys_op_single/opt0_k2026` capture:

| kernel | GPU time | calls | mean |
|---|---:|---:|---:|
| `silu_mul_quant_varlen_kernel` | 6.7% / 2956.1 ms | 78,000 | 37.899 us |
| fused add RMSNorm | 1.5% / 655.0 ms | 160,056 | 4.092 us |
| `_router_triton_kernel` | 0.5% / 238.5 ms | 76,950 | 3.099 us |
| `router_gemm_kernel` | 0.4% / 196.4 ms | 75,600 | 2.598 us |
| `act_and_mul_kernel` | 0.4% / 185.1 ms | 80,028 | 2.314 us |
| `concat_mla_absorb_q_kernel` | 0.4% / 170.8 ms | 80,028 | 2.135 us |
| `_quantize_k_cache_fast_kernel` | 0.2% / 100.0 ms | 80,028 | 1.250 us |
| fused K indexer norm/RoPE/store | 0.1% / 50.6 ms | 21,546 | 2.348 us |
| fused Q indexer RoPE/quant | 0.1% / 42.1 ms | 21,189 | 1.986 us |

The S=32768 decode trace changes the mix: FlashMLA is 22.0%, DeepEP/NCCL are
prominent, `act_and_mul` is 1.3%, `concat_mla_absorb_q` is 1.2%, fused RoPE is
0.4%, cache quant is 0.2%, router top-k is 0.2%, and the fused Q indexer is 0.1%.
This is why task-level speedup is evidence about a boundary, not an end-to-end
claim.

The full default-path category rollup also assigns 52.5% of GPU kernel time to
NCCL plus DeepEP in its short/decode-heavy capture. Those kernels cannot be modeled
honestly as a single-GPU elementwise fusion task.

## Production ABI preserved

- Both DSA task pairs use an existing context length of S=65536. Incremental
  prefill appends positions `[S,S+M)`; decode uses one position S for every batch
  row. They do not emulate prefix-zero full prefill.
- The K input is the non-contiguous `kw[:, :128]` view of a 160-wide projection.
  Its page-size-64 cache uses 132 bytes per token: 128 FP8 bytes followed by four
  FP32 scale bytes. Correctness dequantizes and checks only assigned cache rows;
  untouched historical rows are not rewritten.
- The Q result includes both FP8 Q and the head-gate tensor after q-scale folding.
- Masked MoE checks only valid expert rows and returns the production transposed
  packed-UE8M0 scale layout. Padding is poisoned and cannot become a hidden oracle.
- Router correctness gates top-k IDs exactly and weights numerically. The parameter
  remains BF16 while the stock path uses its cached FP32 copy.

These choices close the largest gap between a local microbenchmark and the cached
serving path. They do not remove the need to replay the containing SGLang region in
eager and CUDA-graph modes and then measure end-to-end TTFT/ITL.

## Initial B200 execution

All eight stock candidates passed initial and unseen-seed post-timing correctness
on `verda-b200x4` with `calc_diff=0` in representative decode M=16 and prefill
M=1024 probes. Those low-repeat runs establish API and numerical compatibility;
they are not performance verdicts because candidate and reference are identical.

All 20 canonical shapes across the eight tasks subsequently passed the same
correctness and unseen-seed checks in full-shape low-repeat sweeps.

One local variant, `router_jit_gemm`, uses SGLang's JIT router GEMM for decode M<=16
and retains the exact production FP32 path as fallback for larger shapes. Increasing
the inner CUPTI sample from the two-iteration diagnostic to the default 30 iterations
removed the earlier launch-scale variance. Its full eligible decode run is:

`runs/glm52/router_gemm_topk_decode/20260801T053943Z-4007d5/result.json`

- M=16: `calc_diff=0`, exact top-k IDs, 10.096 us versus 27.424 us;
  median 2.721x, conservative p10 2.712x, timing spread 1.010x.
- M=32: exact FP32 fallback, 71.344 us versus 71.615 us; neutral with conservative
  speedup 0.998x and timing spread 1.016x.
- Final status: `COMPLETE_WIN`; 1/2 shapes won, none regressed, and post-timing
  correctness passed on a different seed.

This is a valid harness gate but remains provisional evidence because it was measured
before the new task and candidate were committed, and the shared worktree contains
unrelated user changes. It still needs a clean rerun and validation in the containing
MoE/serving region before production promotion.

## Deferred task candidates

| candidate boundary | evidence | why deferred |
|---|---|---|
| MLA Q concat + RoPE + KV-cache quant/store | concat 0.4-1.2%, RoPE 0.4-0.5%, cache quant 0.2% | crosses attention-module and paged-cache ownership; first define one end-to-end cache mutation/consumer ABI rather than three synthetic outputs |
| FlashMLA split-K + combine | 22.0% in S=32768 decode | already a specialized attention region; treat as a FlashMLA backend task with real block tables/cache, not generic operator fusion |
| DeepEP dispatch/GEMM/combine | large share in both captures | distributed EP synchronization and rank-max latency require a serving-native multi-rank task |
| NCCL collectives + consumers | 31.3% in the default-path rollup | topology/collective problem; invalid as a single-GPU kernel task |

The next additive harness work should start with the MLA cache-mutating region and
the distributed DeepEP containing region. Leaf-only Q/O/indexer wins have previously
shown 10-14% local gains while representing roughly 1% of total GPU kernel time, so
they should not be prioritized without a new fusion mechanism or serving evidence.
