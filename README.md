# kernel-harness

Reproducible per-operator microbenchmarks for the hot kernels that SGLang uses when
serving **DeepSeek V3.2 / Kimi K2.x (K2, K2.5, K2.6, K2.7) / MiniMax-M3** style
Multi-head Latent Attention (MLA) + Deep Sparse Attention (DSA) MoE models.

The harness covers the 27 DSA-path ops (+ 16 MiniMax-M3-specific ops) that make up
one transformer layer's prefill and decode pass. Every op has:
- A precise pointer into the actual sglang source that dispatches the kernel
  (see [`docs/kernel_api_mapping.csv`](docs/kernel_api_mapping.csv), 15 columns × 43 rows).
- A shape config in [`shapes.py`](shapes.py) that mirrors a canonical config
  (`hidden=6144, num_heads=64, q_lora_rank=2048, kv_lora_rank=512, qk_nope_head_dim=192,
  qk_rope_head_dim=64, v_head_dim=256, num_index_heads=32, index_head_dim=128,
  n_routed_experts=256, moe_intermediate_size=2048, TP=8+PP=2 for prefill,
  DP32×EP32 with M_local=16 for decode`).
- A **baseline** benchmark in `benchmarks/` that runs the shape through pure
  `torch`, `flash_attn`, or `torch._scaled_mm` — no `sgl_kernel`/`deep_gemm`
  required. Useful as a lower-bound reference on any GPU node.
- A **pointer** into sglang's official pytest / benchmark scripts
  (see [`scripts/run_official_tests.sh`](scripts/run_official_tests.sh) and
  [`scripts/run_official_benchmarks.sh`](scripts/run_official_benchmarks.sh)) that
  invoke the real production kernels — use these on a machine where
  `sgl-kernel` / `deep_gemm` are installed to get numbers you can actually deploy against.

## What this harness is (and is not)

**Is**: a self-contained shape catalogue and a lower-bound-baseline runner.
Everything runs with just `torch + flash_attn + flashinfer`. Every op has a shape
you can eyeball against your model config.

**Is not**: a replacement for sglang's own pytest / benchmark suites. Those live
under `sgl-kernel/tests/`, `test/registered/`, and `benchmark/kernels/` in the
sglang repo, and require `sgl-kernel` + `deep_gemm` to be installed. This
harness gives you a **map** to those scripts (via the two `run_official_*.sh`
wrappers) and a **framework-free reference number** for every op — so when the
real kernel is 3× the baseline you know it's healthy, and when it's 0.8× the
baseline you know something's wrong with the dispatch.

## Layout

```
kernel-harness/
├── README.md                                # this file
├── requirements.txt                         # torch + flash_attn + flashinfer
├── shapes.py                                # single source of every op's (M, K, N, ...) config
├── run_all.py                               # runs every bench, writes JSON+CSV to logs/
├── benchmarks/
│   ├── _util.py                             # bench()/gemm_tflops()/gemm_gbs() helpers
│   ├── linear_gemm.py                       # ops 1-11 prefill; 14-16, 19-21, 23 decode
│   ├── grouped_gemm_moe.py                  # ops 12, 13, 24, 25
│   ├── bmm_absorb.py                        # ops 17, 18 (actual sglang path)
│   ├── indexer_score.py                     # ops 8, 22 (DSA sparse indexer score)
│   └── attention.py                         # ops 26, 27 (Flash Decoding sparse MQA + FA causal MHA)
├── scripts/
│   ├── run_official_tests.sh                # pytest wrapper — invokes sglang's own unit tests
│   └── run_official_benchmarks.sh           # python wrapper — invokes sglang's own bench scripts
├── docs/
│   ├── kernel_api_mapping.csv               # op_id → attr → class → file:line → kernel → tests
│   └── benchmark_results_h800_baseline.csv  # my measured H800 baseline (for sanity check)
└── logs/
    ├── results_h800_baseline.json           # raw JSON from a prior H800 run
    └── results_h800_extra.json              # extra sparse-topk + fp8 proxy runs
```

## Quick start — baseline (no sgl-kernel needed)

```bash
git clone git@github.com:qhy991/kernel-harness.git
cd kernel-harness

# Any Python 3.10+ venv with torch that matches your CUDA driver.
# The env that shipped this harness's baseline: torch 2.5.1+cu124 + flashinfer 0.6.12 + flash_attn 2.7.4
python -m pip install -r requirements.txt

# Run everything and dump results to ./logs/{results.json, results.csv}
python run_all.py

# Or run just one family:
python run_all.py --only linear
python run_all.py --only grouped
python run_all.py --only bmm
python run_all.py --only indexer
python run_all.py --only attention
```

Sample line from `run_all.py` output (H800 SM 9.0, torch 2.5.1+cu124):

```
   1 | Q_a (fused w/ KV_a)              | prefill  | F.linear bf16+fp8               | [16384,6144]x[6144,2624]                    |     759.22 us | TFLOPS_bf16=695.82 GB/s=420.9 fp8_TFLOPS=1200.7
  17 | q_nope absorb BMM                | decode   | torch.bmm bf16 (ACTUAL path)    | [16,64,192]x[64,192,512]                    |      18.48 us | TFLOPS_bf16=10.89
```

## Quick start — full sglang path (real production kernels)

Install `sgl-kernel` and `deep_gemm` alongside your sglang checkout:

```bash
# Following sglang's own install guide:
pip install sgl-kernel deep_gemm

# Then invoke sglang's own tests directly (from the harness scripts):
git clone https://github.com/sgl-project/sglang /path/to/sglang
export SGLANG_DIR=/path/to/sglang
bash scripts/run_official_tests.sh all           # or: gemm | moe | attention | dsa | jit
bash scripts/run_official_benchmarks.sh          # runs the real benchmark_deepgemm_* scripts
```

The wrapper scripts run these actual pytest / benchmark files:

- `sgl-kernel/tests/test_fp8_blockwise_gemm.py`       — block-fp8 GEMM (ops 1–5, 9–10, 14–16, 19)
- `sgl-kernel/tests/test_dsv3_fused_a_gemm.py`        — fused Q_a+KV_a fast path (ops 1, 14)
- `sgl-kernel/tests/test_bmm_fp8.py`                  — bmm_fp8 (ops 17, 18)
- `sgl-kernel/tests/test_flash_attention.py`          — FA3 (op 27)
- `sgl-kernel/tests/test_flash_attn_sparse.py`        — flash_mla_sparse_fwd (op 26)
- `sgl-kernel/tests/test_flashmla.py`                 — flashmla_kv path (op 26)
- `sgl-kernel/tests/test_fp8_blockwise_moe.py`        — grouped fp8 MoE (ops 12, 13, 24, 25)
- `test/registered/jit/test_dsv3_router_gemm.py`      — router GEMM (ops 11, 23)
- `test/registered/jit/test_dsv32_indexer_fusion.py`  — fused Q/K indexer (ops 6, 7, 20, 21)
- `test/registered/kernels/test_dsa_indexer.py`       — end-to-end DSA indexer
- `test/registered/kernels/test_deepgemm_paged_mqa_logits.py` — paged Index_Score (op 22)
- `test/registered/mla/test_flashmla.py`              — MLA + MTP e2e
- `test/registered/moe/test_moe_runners_1gpu.py`      — MoeRunner smoke
- `test/registered/moe/test_moe_ep.py`                — EP MoE end-to-end
- MiniMax-M3 kernels (CUDA subset): `test/registered/jit/minimax/*.py`

Official benchmarks live at:
- `benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py`
- `benchmark/kernels/deepseek/benchmark_deepgemm_fp8_group_gemm.py`
- `benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm_blackwell.py`
- `benchmark/kernels/deepseek/benchmark_deepgemm_dsv3_router_gemm_blackwell.py`
- `benchmark/kernels/deepseek/benchmark_cute_dsl_fp8_paged_mqa_logits.py`
- `benchmark/kernels/fused_moe_triton/benchmark_sglang_fused_moe_triton.py`
- `benchmark/kernels/deepep/tuning_deepep.py`
- `test/registered/jit/benchmark/bench_dsv3_fused_a_gemm.py`
- `test/registered/jit/benchmark/bench_dsv3_router_gemm.py`
- `test/registered/jit/benchmark/bench_topk.py`
- `test/registered/jit/benchmark/bench_dsv4_fp4_indexer.py`
- `test/registered/jit/benchmark/minimax/bench_minimax_*.py`

## Op catalogue (one row per op)

Full mapping (15 columns × 43 rows) is in
[`docs/kernel_api_mapping.csv`](docs/kernel_api_mapping.csv). Columns:

| # | Column | Meaning |
|---|---|---|
| 1 | `op_id` | 1–43 |
| 2 | `model_family` | `DSA (DeepSeek V3.2 / Kimi K2.x)` or `MiniMax M3 (external EntryClass)` |
| 3 | `operator` | short human name |
| 4 | `phase` | `prefill` / `decode` |
| 5 | `shape_or_role` | canonical shape |
| 6 | `attr_and_class` | Python attribute + Linear/Module class |
| 7 | `definition_or_instantiation` | class definition + instantiation `file:line` |
| 8 | `callsite` | forward-time call `file:line` |
| 9 | `ultimate_kernel_or_api` | actual kernel dispatched (deep_gemm/CUTLASS/Triton/flashinfer/...) |
| 10 | `kernel_impl_location` | kernel implementation `file:line` |
| 11 | `dispatch_switch` | `--flag` / `SGLANG_*` env var that picks the backend |
| 12 | `unit_test` | pytest command that exercises the kernel |
| 13 | `integration_test` | end-to-end test command |
| 14 | `benchmark` | official benchmark script |
| 15 | `notes` | gotchas (fused paths, cuda-graph behavior, shape assumptions) |

By family:
- **DSA path (op 1–27)** — DeepSeek V3.2 / Kimi K2.x. Kimi K2.7 routes here
  through `_KimiK2ConfigAlias` (`python/sglang/srt/utils/hf_transformers/common.py:136`)
  → `DeepseekV3ForCausalLM` in `python/sglang/srt/models/deepseek_v2.py`.
- **MiniMax M3 path (op 28–43)** — per-head GQA + a single-head sparse indexer
  (NOT MLA). The top-level `MiniMaxM3SparseForCausalLM` class is expected via
  `SGLANG_EXTERNAL_MODEL_PACKAGE` (`python/sglang/srt/environ.py:797`); all the
  kernels/pool/hybrid-cache/JIT-kernels are in-tree. Some M3 fused kernels
  (`swiglu_oai_mxfp8_quant`, `qk_gemma_rmsnorm_rope` ROCm variant) only fire on
  AMD gfx95; CUDA subset (`minimax_qknorm_rope`, `minimax_decode_topk`,
  `minimax_store_kv_index`) can be tested on any Hopper/Blackwell.

## Baseline numbers I saw on an H800 (SM 9.0)

Full CSV in [`docs/benchmark_results_h800_baseline.csv`](docs/benchmark_results_h800_baseline.csv). Highlights:

| op | shape | backend | latency | perf |
|---|---|---|---|---|
| Q_a fused prefill | [16384,6144]×[6144,2624] | `F.linear bf16` | 759 us | 696 TFLOPS |
| Q_a fused prefill | same | `torch._scaled_mm fp8` | 440 us | **1201 TFLOPS** (proxy for `deep_gemm.fp8_gemm_nt`) |
| Dense GateUp prefill | [16384,6144]×[6144,6144] | `F.linear bf16` | 1666 us | 743 TFLOPS |
| Dense GateUp prefill | same | `torch._scaled_mm fp8` | 954 us | **1296 TFLOPS** |
| O_proj prefill | [16384,2048]×[2048,6144] | `F.linear bf16` | 552 us | 746 TFLOPS |
| MoE GateUp GroupGEMM prefill | 256×[512,6144]×[6144,512] | `torch.bmm bf16` | 1266 us | 651 TFLOPS (proxy) |
| q_nope absorb BMM decode | [16,64,192]×[64,192,512] | `torch.bmm bf16` (**actual path**) | 18.5 us | 11 TFLOPS |
| v absorb BMM decode | [16,64,512]×[64,512,256] | `torch.bmm bf16` (**actual path**) | 18.8 us | 14 TFLOPS |
| Q_a fused decode | [16,6144]×[6144,2624] | `F.linear bf16` | 28 us | 1.16 TB/s (35% HBM) |
| O_proj decode | [16,16384]×[16384,6144] | `F.linear bf16` | 92 us | 2.20 TB/s (66% HBM) |
| FlashAttn causal MHA prefill | Q:[1,8,16384,256] causal | `flash_attn 2.7.4 fp16` | 3678 us | 299 TFLOPS |
| Flash Decoding MLA sparse-topk decode | Q:[1,64,16,512] topk=2048/T_kv=8k | `SDPA + gather` (proxy) | 457 us | 9 TFLOPS |
| Index_Score prefill (naive) | Q:[16384,32,128] × K:[16384,128] | `torch einsum + topk` | **56 ms** | 41 TFLOPS (naive; `deep_gemm.fp8_mqa_logits` is 10–50× faster) |

Rules of thumb the baseline numbers should give you:
- **H800 bf16 peak**: ~989 TFLOPS. My prefill Linear GEMMs hit 50–75% of peak.
- **H800 fp8 peak**: ~1979 TFLOPS. My `torch._scaled_mm` runs hit 55–65% of peak;
  `deep_gemm` typically reaches 70–80% (10–15% higher than these proxies).
- **H800 HBM peak**: ~3.35 TB/s. My decode GEMMs hit 30–70% of BW.
- **Index_Score naive is meaningless** for the real kernel — the naive fp16
  einsum baseline is a *worst-case reference*, not a target. The real fp8 mqa
  kernel is 10–50× faster.
- **FlashAttn 2 vs FA3**: on H800, FA3 typically gives 1.5–2× over FA2 for
  head_dim=256 causal; my baseline is FA2, so add ~1.5× to estimate the FA3
  number.

## Adapting to a different config

Every shape lives in `shapes.py` — dataclasses grouped by op family. To match a
different MLA/DSA config:
1. Update the constants at the top of `shapes.py` (or edit each `LinearShape`
   entry directly).
2. Re-run `python run_all.py`.
3. The pretty-printed lines and the per-run CSV/JSON pick up the new shapes
   automatically.

For MiniMax M3 (per-head GQA + single-head sparse indexer, NOT MLA), the shapes
are structurally different — you'd add a new file `benchmarks/m3_sparse.py`
following the same pattern. Signatures to model against:
- `python/sglang/srt/layers/attention/minimax_sparse_ops/minimax_sparse.py:30-62`
  (`minimax_sparse_prefill` and `minimax_sparse_decode`).
- `python/sglang/srt/layers/attention/minimax_sparse_ops/{prefill,decode}/*.py`
  (Triton entry points).
- `python/sglang/jit_kernel/minimax_qknorm_rope.py:152` (CUDA fused QK-norm+RoPE).

## Sample output

`logs/results_h800_baseline.json` contains the actual run I captured on an H800
node with driver 12.4 / torch 2.5.1+cu124 / no `sgl-kernel`. Use it to sanity-
check your own run's shape parsing / sample counts.

## License

Same as the sglang repo you point to — Apache 2.0. The harness itself is BSD-3
so you can drop it into internal repos without friction.
