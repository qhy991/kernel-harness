# Kimi-K2.7 kernel-fusion opportunities (B200 / sm100)

Grounded in the actual sglang forward path (`models/deepseek_v2.py`,
`models/deepseek_common/attention_forward_methods/forward_mla.py`) and ranked by
the measured baselines in `kimi_k27_all.csv`.

## Why fusion matters here — the measured motivation
The sweeps show **every decode op floors at ~49–57µs regardless of batch size**
(B=1→256 flat) — that floor is kernel-launch + memory round-trip, not compute.
A Kimi decode step issues ~25–30 kernels/layer × 60 layers. So the dominant
decode lever is **reducing kernel count + memory traffic (fusion)**, not
per-kernel speedup. At prefill the win is **fewer large-activation round-trips**
(moe_sum alone is 1.2ms @16k, the steepest per-token op).

## Already fused in sglang (do NOT re-propose)
| Fusion | Kernel |
|---|---|
| Q_a + KV_a projection | `fused_qkv_a_proj_with_mqa` (single GEMM) |
| Residual add + RMSNorm | `fused_add_rmsnorm` |
| Router gate + sigmoid + biased top-k | `kimi_k2_moe_fused_gate` (Kimi-specific) |
| SwiGLU + FP8 quant of down-input | `silu_and_mul_contig_post_quant` / `_masked_post_quant_fwd` |
| Min-latency fused-a (M≤16) | `dsv3_fused_a_gemm` |
| Shared-expert ∥ dispatch/combine, q_a∥kv_a norm | alt-stream *overlap* (not a fused kernel) |

## Opportunities (not fused on the CUDA/B200 path)

### Tier 1 — high value, clean, a CUDA gap already exists

**1. RMSNorm → FP8 quant fusion.**
`fused_add_rmsnorm` emits bf16, then a *separate* `per_token_group_quant_fp8`
quantizes it before each FP8 GEMM (Q_a, MoE gate, dense gate_up). AMD already
has `fused_rms_fp8_group_quant`; **the NVIDIA path does norm then quant as two
kernels** (`layernorm.py` has no CUDA rms+quant; `forward_mla.py:231-232`).
Fusing norm→quantized-fp8-output removes one kernel **and one full hidden-state
memory round-trip per FP8 GEMM**. Measured: act-quant = 158µs @M16k prefill,
~49µs decode, and it runs before *every* FP8 GEMM. **Biggest, safest win.**

**2. MoE combine epilogue: `moe_sum` + routed_scaling + shared-expert add.**
Top-k weighted reduce (`moe_sum`) is one kernel; then `x.add_(final_hidden,
alpha=routed_scaling_factor)` and the shared-output add are separate elementwise
passes (`deepseek_v2.py:1548-1560`). Fold `routed_scaling` and the shared add
into the `moe_sum` epilogue. Measured: moe_sum prefill 1.2ms @16k (steepest
per-token op) + 2 extra hidden-state passes → 1 fused pass.

### Tier 2 — medium, structurally feasible

**3. q_a_layernorm + kv_a_layernorm → one fused dual-RMSNorm kernel.**
On CUDA these are two `rmsnorm` launches (`forward_mla.py:231-232`), only
*stream-overlapped* in capture mode. AMD fuses them (`fused_qk_rmsnorm_bf16`).
A single kernel normalizing two different-width tensors (1536 + 512) halves the
launches every layer. Combine with #1 → a single **q/kv-norm + fp8-quant** kernel
(exactly what AMD's `fused_rms_fp8_group_quant` does; no CUDA equivalent wired).

**4. RoPE + KV-cache store fusion.**
`fused_qk_rope_reshape_and_cache` exists but is **not wired into the CUDA
`forward_mla` path** (rope at `:418`, then a separate `set_kv_buffer`). Fusing
rope with writing k_pe/k_nope into the paged cache removes a kernel + a KV write.
(The trtllm *decode* path already fuses rope+quant via `_fuse_rope_for_trtllm_mla`;
this targets prefill / non-trtllm.)

**5. K-path norm + RoPE (`fused_qk_norm_rope`).**
`fused_qk_norm_rope` / `dsv4_fused_q_norm_rope` are available but unused on the
Kimi CUDA MLA path. The **K** side (kv_a_layernorm on k_nope → rope on k_pe) is
adjacent and fusable; the **Q** side is blocked by q_b_proj GEMM sitting between
norm and rope, so only partial.

### Tier 3 — advanced / framework-level

**6. Absorb-BMM pair (w_kc, w_vc) → one batched/grouped kernel, or into the
attention epilogue.** Two `torch.bmm` calls bracket the attention core
(`forward_mla.py:372-397`). Fuse the pair (or push v-absorb into the
flashinfer/trtllm output epilogue). Each ~50µs decode (launch-bound) → merge
saves launches. `bmm_fp8` path already exists as the quantized variant.

**7. Gate GEMM + top-k co-launch.** `mlp.gate` GEMM then `kimi_k2_moe_fused_gate`
are back-to-back tiny decode kernels (58 + 52µs); a gemm-with-fused-topk-epilogue
would remove one launch. Low compute, decode-launch-bound only.

## Ranked shortlist
| # | Fusion | Where | Measured motivation | Effort |
|---|---|---|---|---|
| 1 | RMSNorm + FP8 quant | forward_mla.py:231, layernorm.py | quant 158µs@16k, every FP8 GEMM | Med (new CUDA kernel; AMD ref exists) |
| 2 | moe_sum + scaling + shared-add | deepseek_v2.py:1548-1560 | moe_sum 1.2ms@16k + 2 passes | Med (epilogue) |
| 3 | dual q_a/kv_a RMSNorm (+#1) | forward_mla.py:231-232 | 2 launches/layer → 1 | Low–Med |
| 4 | RoPE + KV-cache store | forward_mla.py:418 | 1 kernel + KV write/layer | Med (wire existing kernel) |
| 5 | K-norm + RoPE | forward_mla.py | 1 launch/layer (K side) | Med (partial) |
| 6 | absorb-BMM pair | forward_mla.py:372-397 | 2×50µs decode launches | High (correctness-sensitive) |

## Caveats
- Decode motivation is launch/memory-traffic, largely hidden by CUDA graphs
  sglang already uses — fusion still cuts graph-replay time + bandwidth, but the
  wall-clock win at decode is smaller than the ~50µs microbench floor suggests.
- #1/#3 have AMD reference implementations (`fused_rms_fp8_group_quant`,
  `fused_qk_rmsnorm_bf16`) to port; NVIDIA versions must be written/validated.
- All numbers are single-op microbench (random data); confirm with an end-to-end
  profile before committing kernel work — a fusion that saves a 50µs launch-bound
  op may vanish under CUDA-graph capture.
