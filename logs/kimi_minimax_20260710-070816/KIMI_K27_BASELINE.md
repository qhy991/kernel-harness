# Kimi-K2.7 kernel baselines on B200 — test method & sglang API mapping

> **Status: COMPLETE (microbenchmark scope).** 52 operators, all `已测`; sglang
> API mapped per op; non-config axes swept as ranges; findings recorded in
> `kimi_k27.csv` 备注. Three scripts + this doc capture the method. Remaining
> items are **resource-blocked or out of single-op scope** and do NOT block
> completion: (1) real-inference shape capture — needs a Kimi-K2.7 checkpoint
> (none on this host); (2) true decode kernel time below the ~50µs launch floor —
> needs a CUDA-graph/event timing harness; (3) correctness — random data, timing
> only; (4) DeepEP comm / MTP-NextN / KV-store — multi-GPU or spec-decode scope.
> A follow-up round is warranted only if a checkpoint becomes available (→ shape
> capture + correctness) or production-accurate decode latency is required.


Backfills every `未测` ("B200 待补测") row in `kimi_k27.csv` with a measured
single-op latency/TFLOPS baseline, using the **same** kernel APIs sglang
dispatches at runtime on B200 (sm100). All 31 rows are now `已测`.

- GPU: NVIDIA B200 (sm100) · torch 2.11.0+cu130
- Model: Kimi-K2.7 — MLA + MoE, hidden=7168, 61 layers (1 dense + 60 MoE),
  384 routed experts, 1 shared expert, **no DSA indexer**. TP=8 prefill,
  DP32×EP32 decode (M_local=16). Config: `kimi_model_config.py::KIMI_K2`.

## How to run

```bash
cd /home/qinhaiyan/kernel-harness
source activate_env.sh            # puts source sglang (/home/qinhaiyan/sglang/python) on PYTHONPATH
                                  # + the glm52 venv (sgl_kernel, deep_gemm, flashinfer)
"$PYTHON" scripts/bench_kimi_untested.py logs/<run>/kimi_untested.json
```

- First run JIT-compiles the DeepGEMM FP8 shapes (~1 min/shape); the compile
  cache is on disk so re-runs are seconds.
- `scripts/bench_kimi_untested.py` reuses the validated helpers in
  `scripts/bench_kimi_real_kernels.py` (`bench_fp8_gemm`, `bench_grouped_fp8`,
  `bench_bmm`, `bench`) and adds the masked-grouped / router / fused-a /
  MLA-attention paths.
- Timing: `bench()` = 10 warmup + 50 (attention 3+10) timed iters,
  `torch.cuda.synchronize()` around a `perf_counter` loop. Shapes are taken
  verbatim from the CSV for GEMM/BMM/router/grouped ops.

## sglang kernel API per operator (as dispatched on B200)

| Operator (phase) | sglang entry | Kernel API benchmarked | us | TFLOPS |
|---|---|---|---|---|
| Q_a+KV_a fused (prefill) | `fused_qkv_a_proj_with_mqa` | `deep_gemm` w8a8_block_fp8 (`w8a8_block_fp8_matmul_deepgemm`) | 436.8 | 1170 |
| Q_a+KV_a fused (decode) | `fused_qkv_a_proj_with_mqa` | deep_gemm fp8 (default); `sgl_kernel.dsv3_fused_a_gemm` bf16 alt = **54.2us** | 56.6 | 8.8 |
| Q_b (prefill) | `q_b_proj` | deep_gemm w8a8_block_fp8 | 77.8 | 993 |
| Q_b (decode) | `q_b_proj` | deep_gemm w8a8_block_fp8 | 52.5 | 11.5 |
| KV_b (prefill) | `kv_b_proj` | deep_gemm w8a8_block_fp8 | 73.4 | 468 |
| KV_b absorb BMM (decode) | `w_kc` (from kv_b) | `torch.bmm` bf16 → cuBLAS | 49.6 | 0.34 |
| V absorb BMM (decode) | `w_vc` (from kv_b) | `torch.bmm` bf16 → cuBLAS | 50.1 | 0.34 |
| O_proj (prefill) | `o_proj` | deep_gemm w8a8_block_fp8 | 264.2 | 910 |
| O_proj (decode) | `o_proj` | deep_gemm w8a8_block_fp8 | 57.6 | 32.6 |
| MLA attention (prefill) | `dsa_backend`/flashinfer | `flashinfer.single_prefill_with_kv_cache` (ragged, causal) | 2256 (4k) / 8426 (8k) | 152 / 163 |
| MLA attention (decode) | `forward_mla` | `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` | 71.3 (4k) / 146.4 (8k) | 128 / 125 |
| MoE Router (prefill) | `mlp.gate` | `F.linear`/`torch.mm` cuBLAS bf16 (M>16 path) | 173.4 | 520 |
| MoE Router (decode) | `mlp.gate` | `sgl_kernel.dsv3_router_gemm` (CUDA JIT, out fp32) | 57.8 | 1.5 |
| MoE GateUp grouped (decode) | `experts.w13_weight` | `deep_gemm.fp8_m_grouped_gemm_nt_masked` | 73.5 | 51.1 |
| MoE Down grouped (prefill) | `experts.w2_weight` | `deep_gemm.m_grouped_fp8_gemm_nt_contiguous` | 2441 | 296 |
| MoE Down grouped (decode) | `experts.w2_weight` | `deep_gemm.fp8_m_grouped_gemm_nt_masked` | 61.0 | 30.8 |

## Method notes & assumptions

- **FP8 W8A8 linears** — activation quant via `sglang_per_token_group_quant_fp8`
  (column-major, UE8M0, TMA-aligned) + weight `requant_weight_ue8m0`, mirroring
  `benchmark_deepgemm_fp8_gemm_blackwell.py`. This is the B200 default DeepGEMM
  `fp8_gemm_nt` path.
- **MoE decode masked grouped GEMM** — per-token (act) + per-block (weight)
  UE8M0 casts, then `deep_gemm.transform_sf_into_required_layout(recipe=(1,128,128))`
  to get the mn-major TMA-aligned scale layout the masked kernel requires
  (matches `moe_runner/deep_gemm.py::_run_masked_gemm`). Grouped shapes E and N
  are the CSV's "approx" DP32×EP32 decode assumption (E=8).
- **MoE Router prefill** uses cuBLAS `F.linear`, not `dsv3_router_gemm`:
  `deepseek_v2.py` only takes the `dsv3_router_gemm` fast path when M≤16 and
  hidden==7168 and experts∈{256,384}; prefill M=16384 falls through to F.linear.
- **MLA attention** shapes were **under-specified in the CSV** (`Q[16,64,320]`
  doesn't map to the Kimi-K2.7 MLA config), so real Kimi-K2.7 MLA dims are used:
  64 heads, qk_nope=128, qk_rope=64, kv_lora=512, v=128; absorbed decode
  head_dim_qk = 512+64 = 576, out v = 512, MQA, paged KV (page=64), bf16.
  Baselined at **both seq=4096 and seq=8192** (batch=16 decode).
- **MLA decode kernel choice**: `sgl_kernel.flash_mla.flash_mla_with_kvcache`
  dense decode raises *"only supported on SM90a"* — it is Hopper-only. On B200
  the sglang production path is FlashInfer TRTLLM-gen MLA
  (`trtllm_mla_backend.py`), which is what is benchmarked here.
- **Already-`已测` rows** (embedding, RMSNorm, dense FFN, shared expert, MoE
  GateUp prefill, LM head) come from `bench_kimi_real_kernels.py` /
  `bench_kimi_layer_coverage.py` and were left unchanged.

Raw results: `kimi_untested.json` · full stdout: `06_untested.log`

---

## Round 2 — per-layer element-wise / routing / norm kernels

`scripts/bench_kimi_missing_ops.py` covers the small-but-every-layer kernels the
GEMM-centric CSV skipped (21 rows appended → 52 total). Same env / timing.

| Operator (phase) | sglang entry | Kernel API | us |
|---|---|---|---|
| MoE routing gate (pre/dec) | `mlp.gate`→topk | **`sgl_kernel.kimi_k2_moe_fused_gate`** (Kimi-specific, sigmoid+biased top-8) | 79.1 / 52.2 |
| SwiGLU act — dense (pre/dec) | `mlp.act_fn` | `sgl_kernel.silu_and_mul` | 146.6 / 49.9 |
| SwiGLU act — MoE grouped (pre) | `experts.act_fn` | `silu_and_mul` (m_sum=384×512) | 445.7 |
| SwiGLU act — MoE masked (dec) | `experts.act_fn` | `silu_and_mul` | 50.9 |
| Act FP8 quant — hidden (pre/dec) | pre-GEMM quant | `sglang_per_token_group_quant_fp8` (UE8M0) | 158.0 / 49.0 |
| Fused add+RMSNorm (pre/dec) | `input_/post_attention_layernorm` | `sgl_kernel.fused_add_rmsnorm` | 336.2 / 49.8 |
| q_a_layernorm (pre/dec) | `q_a_layernorm` | `sgl_kernel.rmsnorm` [M,1536] | 62.0 / 49.2 |
| kv_a_layernorm (pre/dec) | `kv_a_layernorm` | `sgl_kernel.rmsnorm` [M,512] | 53.9 / 48.6 |
| RoPE q+k (pre/dec) | `rotary_emb` | `jit_kernel.rope.apply_rope_with_cos_sin_cache_inplace` | 144.2 / 48.8 |
| MoE combine / reduce (pre/dec) | `experts` (topk weighted sum) | `sgl_kernel.moe_sum` [M,8,7168] | 1233.4 / 50.9 |
| KV_b / V absorb BMM **H=64** (dec) | `w_kc`/`w_vc` | `torch.bmm` (CSV rows used H=8) | 50.3 / 50.3 |

### Important caveats

- **Decode latencies for these ops floor at ~49µs** — that is the per-call
  Python launch + `perf_counter` overhead of this microbench at M=16, **not**
  kernel compute. The tiny decode element-wise ops are launch-bound here; only
  their *relative* order and the **prefill** numbers are meaningful signal.
  (The same floor applies to the round-1 decode GEMMs at ~50–57µs.)
- **Act FP8 quant is measured separately and is *excluded* from the round-1
  GEMM latencies** (`prep_fp8_gemm` quantizes outside the timed loop). Real
  per-op decode cost ≈ act-quant + GEMM. Prefill hidden quant alone is ~158µs.
- **Absorb BMM head count**: round-1 rows (H=8, from the CSV) undersize the op;
  DP-attention decode runs all 64 heads → the H=64 rows are the realistic size
  (compute ~8× larger; still launch-bound in wall-clock at M=16).
- **MoE combine (moe_sum) prefill is expensive** (~1.2ms) — top-8 × hidden=7168
  reduction over 16384 tokens; a real optimization target.

### Still not covered (by design / scope)
- **DeepEP dispatch/combine** (EP all-to-all) — multi-GPU comm, not a single-GPU op.
- **MTP / NextN draft layer** — only relevant with speculative decoding enabled.
- **KV-cache store** (`set_kv_buffer`) — often fused into RoPE
  (`fused_qk_rope_reshape_and_cache`); not benched standalone.
- **Batch/seq sweeps** — still single point (decode M=16, prefill M=16384).

Raw results: `kimi_missing_ops.json` · script: `scripts/bench_kimi_missing_ops.py`

---

## Round 3 — sweeps over the non-config axes

Everything that is **not** a fixed model-config dimension is now swept over a
range instead of a single point (`scripts/bench_kimi_sweeps.py`, 164 points).
Long-form data: **`kimi_k27_sweeps.csv`** (one row per op×axis×value) and
`kimi_sweeps.json`. The single-point values in `kimi_k27.csv` are the reference
operating point; each swept row's 备注 now carries its measured range.

Grids: prefill M `[512,1k,2k,4k,8k,16k,32k]` · decode B `[1,2,4,8,16,32,64,128,256]`
· tokens/expert `[16..1024]` + realistic routed dist · masked_m `[1..128]` +
skewed vector · attn prefill seq `[1k..16k]` · attn decode KV `[1k..32k]`.

### A. Prefill token count M (µs)
| op | 512 | 2k | 8k | 16k | 32k |
|---|---|---|---|---|---|
| Q_a fused (fp8) | 58 | 72 | 248 | 436 | 878 |
| Dense GateUp (fp8) | 62 | 149 | 435 | 890 | 1781 |
| O_proj (fp8) | 55 | 65 | 156 | 265 | 534 |
| MoE Router (cuBLAS) | 56 | 60 | 84 | 173 | 351 |
| Fused add+RMSNorm | 52 | 61 | 169 | 337 | 623 |
| SwiGLU dense | 51 | 55 | 74 | 148 | 245 |
| Act FP8 quant | 51 | 57 | 80 | 159 | 269 |
| **MoE combine (moe_sum)** | 64 | 169 | 619 | 1236 | **2779** |

FP8 GEMMs are launch-bound (~55µs) up to M≈2k, then scale ~linearly. `moe_sum`
is the steepest per-token cost and the clearest prefill optimization target.

### B. Decode batch size B (µs) — **flat / launch-bound across the whole range**
| op | 1 | 16 | 64 | 256 |
|---|---|---|---|---|
| Q_a fused (fp8) | 57 | 57 | 57 | 57 |
| Q_b (fp8) | 53 | 53 | 53 | 55 |
| O_proj (fp8) | 57 | 57 | 59 | 62 |
| Router cuBLAS | 53 | 53 | 54 | 56 |
| Router dsv3_router_gemm | 50 | **58** | (M≤16 only) | — |
| absorb BMM H=64 | 50 | 50 | 51 | 54 |

Decode GEMMs barely move B=1→256 → they are memory/launch-bound at Kimi decode
dims, **not** compute-bound. Router crossover: `dsv3_router_gemm` grows 50→58µs
over B=1→16 and by B=16 is slower than cuBLAS (53µs) — this is exactly why
sglang gates it to M≤16.

### C. Tokens per expert — contiguous grouped, prefill (µs)
`16:588 · 64:587 · 128:587 · 256:661 · 341:876 · 512:1072 · 1024:1807`;
realistic routed dist `M=4096→588, M=16384→875` (matches the uniform equivalent).
**Flat below 128** because deep_gemm pads each expert to the 128-row alignment —
sub-128 per-expert counts pay for 128 rows regardless (a real EP-decode penalty).

### D. masked_m — masked grouped, decode (µs)
`1..128 → 73µs flat`; skewed `[1..128]` vector → 73µs; `ep12` (12 experts) → 87µs.
Same 128-alignment padding floor: masked_m below 128 is free-riding on padded work;
cost tracks the **number of expert groups**, not the valid-token count.

### E. Attention prefill seqlen (µs) — clean O(seq²) causal
`1024:343 · 2048:787 · 4096:2254 · 8192:8449 · 16384:31560` (~4× per 2× seq).

### F. Attention decode (µs)
- KV len @B=16: `1k:65 · 2k:72 · 4k:71 · 8k:146 · 16k:173 · 32k:334` (~linear in KV).
- Batch @seq=4096: `1:60 · 8:71 · 32:145 · 64:174 · 128:333 · 256:615` (flat ≤8, then linear).

### Caveats unchanged
- The ~49–62µs floor on all decode curves is microbench launch/timing overhead at
  small work, not kernel time — read curve *shape* and the large-M/large-seq end.
- Data is random; routing distribution is `topk(randn)` (min288/max400/mean341 over
  384 experts) — **less skewed than production MoE**, so real per-expert imbalance
  (and thus the padding penalty in C/D) is likely worse.

Raw results: `kimi_sweeps.json` · `kimi_k27_sweeps.csv` · script: `scripts/bench_kimi_sweeps.py`

---

## Round 4 — recovered kernels (2nd coverage audit)

A second audit of the real forward path found critical-path kernels the
GEMM/elementwise rounds missed. `scripts/bench_kimi_recovered.py` benches them
(added to `kimi_k27_all.csv` as `测量类型=recovered`).

| Kernel | sglang API | prefill | decode |
|---|---|---|---|
| KV-cache store (MLA) | `set_mla_kv_buffer_triton` (JIT CUDA / Triton) | 51.7µs | 51.3µs |
| MoE scale TMA-align | `tma_align_input_scale` | 100.9µs | 52.2µs |
| Sampler softmax | `torch.softmax` [B,163840] | — | 154µs(B16) / 269µs(B256) |
| Sampler greedy | `torch.argmax` | — | 107µs / 155µs |
| **Sampler top_k_top_p** | `flashinfer.top_k_top_p_sampling_from_probs` | — | **246µs / 907µs** |

**Key finding:** token **sampling is a heavy, per-step, previously-invisible
cost** — softmax + top_k_top_p ≈ 400µs @B16 and **~1.2ms @B256** over
vocab=163840, on the same order as a whole layer's compute, and it runs once
every decode step. KV-store and TMA-align are per-layer but launch-bound (~50µs).

### Identified but NOT benched (honest audit closure — `测量类型=identified_gap`)
- **MoE token permute** (`ep_scatter`/`ep_gather`/`deepep_post_reorder`,
  `ep_moe/kernels.py:838,988`) — sorts/scatters tokens for contiguous grouped
  GEMM every MoE layer; needs the DeepEP dispatch context to feed realistic
  inputs, so not micro-benched here.
- **TP/DP/EP collectives** (all_reduce / all_gather+reduce_scatter / all-to-all)
  — multi-GPU, not a single-GPU op; often a large decode share.
- **MTP/NextN draft layer** — a full extra attention+MoE path, only with
  speculative decoding on.
- **Attention-side KV FP8 quant** (`scaled_fp8_quant`, trtllm path) — same family
  as the already-benched act-quant.

Raw results: `kimi_recovered.json` · script: `scripts/bench_kimi_recovered.py`

---

## Round 5 — quant kernels (both models)

Prior rounds benched only the block-FP8 *activation* quant standalone. This round
recovers the other runtime quant kernels, for **both** harness models.

### Kimi-K2.7 (added to `kimi_k27_all.csv`, `recovered`)
| Quant kernel | API | prefill | decode |
|---|---|---|---|
| KV-cache FP8 quant | `scaled_fp8_quant` (per-tensor, trtllm FP8 KV) | 53.8µs | 49.1µs |
| MoE masked 8bit — fused silu+quant | `sglang_per_token_group_quant_8bit` | — | 49.4µs |
| MoE masked 8bit — quant only | `sglang_per_token_group_quant_8bit` | — | 50.0µs |

### MiniMax-M3 / DSA indexer quant (NEW file `minimax_m3_all.csv`)
The DSA sparse-attention indexer runs quant every layer; **absent from the Kimi
model entirely** (no indexer) and done *outside* the timed loop in the harness's
own indexer bench, so previously invisible.
| Quant kernel | API | prefill | decode |
|---|---|---|---|
| Indexer Q FP8 act-quant | `dsa.act_quant` | 83.7µs | 49.5µs |
| Indexer K FP8 act-quant | `dsa.act_quant` | 50.3µs | 49.0µs |
| Indexer Hadamard rotate | `rotate_activation` (hadamard) | 57.7µs | 49.2µs |
| **Index K-cache FP8 quant** | `quantize_k_cache_separate` (tiled) | **140.0µs** | 49.3µs |

**Finding:** the DSA K-cache FP8 quant (140µs prefill) is the heaviest single
quant op — it runs every layer in the MiniMax-M3/DSA model and was completely
uncovered. All indexer-quant work belongs to the DSA model only; it does not
apply to Kimi-K2.7.

Scripts: `bench_kimi_quant.py`, `bench_minimax_indexer_quant.py` ·
Raw: `kimi_quant.json`, `minimax_indexer_quant.json`

---

## Round 6 — consolidation + MiniMax-M3 gap closure

**One unified file:** `all_models_kernel_inventory.csv` (274 rows, 15-col superset)
merges all three models by a `模型` column:
- **Kimi-K2.7** — 247 rows (reference/sweep/fusion/recovered/gap microbenches).
- **MiniMax-M3** — 19 rows: 15 from the authoritative `docs/minimax_m3_operator_backend_inventory.csv`
  (real GQA model, `minimax_sparse_ops`) + 4 measured `mega_moe`-core rows.
- **DeepSeek-V3.2/DSA** — 8 rows (**relabeled**; see below).

**★缺口3 closed (measured):** `mega_moe` can't be microbenched whole (needs built
weights + deep_gemm symm buffers), so its **grouped-GEMM core** is measured at real
M3 shapes (E=128, hidden=6144, inter=3072) as a block-FP8 proxy for MXFP8:
GateUp/Down prefill = **5213 / 2873µs** (949 / 861 TFLOPS), decode masked = 265 / 160µs.
This is the dominant B200 M3 MoE cost, previously a 备注 with no number.

**Model mislabel corrected:** the earlier `minimax_m3_all.csv` actually measured the
**DeepSeek-V3.2 / DSA (MLA)** indexer-quant path, *not* MiniMax-M3 (which is GQA +
`minimax_sparse_ops`, per the inventory). Renamed → `deepseek_v32_dsa_indexer_quant.csv`
with `模型=DeepSeek-V3.2/DSA`. The inventory CSV is the MiniMax-M3 source of truth.

Consolidated file: `all_models_kernel_inventory.csv`

---

## Round 7 — MiniMax-M3 coverage parity

Ran the same pipeline (reference + sweeps) on the M3 ops that ARE runnable in
this checkout (`scripts/bench_minimax_m3.py`, 119 pts). MiniMax-M3 in the unified
file went **19 → 134 rows** (15 inventory + 21 reference + 98 sweep).

M3 config: hidden=6144, GQA 64q/4kv/128, 128 experts top-4, moe_inter=3072,
sigmoid routing, SwiGLU-OAI. Prefill M=16384, decode M=16.

| op (prefill ref) | M3 | vs Kimi | note |
|---|---|---|---|
| Main QKV proj (fp8) | 1518µs | Kimi Q_a 437µs | M3 fuses all QKV (N=9216) vs Kimi split MLA proj |
| GQA attention @4k (dense) | 2057µs | Kimi MLA 2256µs | comparable; M3 real path is *sparse* (topk) |
| MoE Combine (moe_sum) | 1153µs | Kimi 1233µs | ~same |
| MoE GateUp grouped | 5414µs | Kimi 897µs | **not comparable** — see caveat |

**Sweep behavior mirrors Kimi:** prefill scales ~linearly above M≈2k; decode is
launch-bound flat (~50–60µs) across B=1→256; grouped GEMM scales with
tokens/expert (651µs→9746µs over 16→1024); masked flat (~264µs) below the 128
alignment; GQA attention is O(seq²) (319µs→25.7ms over 1k→16k).

### Caveats (M3 rows are proxies)
- **FP8 = deep_gemm block-fp8** proxy for the MXFP8 checkpoint.
- **SwiGLU = standard silu_and_mul** proxy for SwiGLU-OAI (clamped, α=1.702).
- **Attention = dense GQA** upper bound; M3 runs it **sparse** (topk block).
- **MoE GateUp not comparable to Kimi's row**: M3 used full un-sharded N=6144
  (2·3072); Kimi's reference used TP-sharded N=256 (2048/8). Same kernel, ~24×
  compute — a TP-assumption mismatch, not a model difference.
- M3-specific kernels (sparse indexer, fused_gemma_qknorm_rope, store_kv_index,
  minimax_decode_topk, gqa_share_sparse, SwiGLU-OAI/MXFP8, whole mega_moe) live
  in the `amd_add_m3` worktree — **not in this checkout** — so they remain
  inventory-only (`docs/minimax_m3_operator_backend_inventory.csv`).

Script: `scripts/bench_minimax_m3.py` · Raw: `minimax_m3_bench.json`
