# MI300X GLM-5.2 — optimization degree re-tested against the AITER baseline

The earlier campaign measured our tuned kernels against a torch/triton reference. Per the
follow-up ask, we re-baseline against **AITER** — the sglang-ROCm production kernel path
(qhy's `SGLang-DGMK:decode-fusion-r1` routes GLM-5.2 DSA/FP8 through exactly these) — and
re-measure how much our kernels still improve.

Reproduce: `.venv/bin/python rewardbench/amd/aiter_baseline.py` (needs the aiter repo on
`PYTHONPATH`, `AITER_TRITON_ONLY=1`). Data: `amd_glm5_aiter_baseline.csv`.

## AITER operators used as baseline
| op | aiter operator | correct vs our oracle |
|---|---|---|
| o_proj / index_k (prefill GEMM) | `aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale` | yes (calc_diff ≤ 5e-6) |
| dsa_attn (decode sparse MLA) | `aiter.ops.triton.attention.pa_decode_sparse` | yes (calc_diff ~4e-6) |

Both consume our frozen `glm52_ops` inputs directly — aiter's a8w8 blockscale layout is
identical to ours (`x_scale [M,K//128]`, `w_scale [ceil(N/128),K//128]`), and
`pa_decode_sparse` takes `q[N,H,D]` + a flat shared KV pool + `kv_indices/indptr` (we map
our topk `indices` to it, mask the value dim to the first 512, disable the attn-sink).

## Result — our tuned kernels vs the AITER baseline (HIP-event cold-L2 median)
| op | M | aiter µs | our tuned µs | **tuned / aiter** | aiter / our-old-ref |
|---|---|---|---|---|---|
| o_proj | 1024 | 946 | 428 | **2.21×** | 1.87× |
| o_proj | 2048 | 1913 | 904 | **2.12×** | 1.67× |
| o_proj | 4096 | 3162 | 1326 | **2.38×** | 1.93× |
| index_k | 1024 | 568 | 289 | **1.97×** | 1.49× |
| index_k | 2048 | 540 | 287 | **1.88×** | 1.56× |
| index_k | 4096 | 540 | 286 | **1.88×** | 1.56× |
| dsa_attn | 16 | 274 | 123 | **2.23×** | 0.96× |
| dsa_attn | 32 | 433 | 128 | **3.38×** | 0.89× |

**Geomean: our kernels are 2.22× faster than the AITER baseline.** AITER's GEMM is a
genuinely harder baseline than our old reference (1.5–1.9× faster than it); its sparse-MLA
decode is ~on par with our chunked reference at these tiny decode batches (0.9×).

## Why we still win, and the honest caveat
- **GEMM win = native fp8 MFMA + AMD MFMA-knob tuning.** Our kernel does `tl.dot` on
  native `e4m3fnuz` (2.6 PF matrix core) with `matrix_instr_nonkdim`/`waves_per_eu`/`kpack`
  tuned per shape; aiter's runnable triton blockscale kernel is a generic-config path.
- **dsa_attn win = fused tk-split flash-decode + fused combine**, which fills the CU grid
  the tiny decode batch otherwise starves.
- **CAVEAT — this is aiter's Triton path, not its ASM peak.** aiter's CK/ASM GEMM
  (`gemm_a8w8_blockscale_bpreshuffle_asm`, ~2.64× over CK) needs a C++ JIT build into
  aiter's package dir, which is **not writable on this node**, and no prebuilt gfx942 code
  objects ship — so only the Triton path runs here. Against aiter's ASM peak the GEMM gap
  would narrow substantially (aiter ASM could approach or beat our kernel at large M). Our
  result is "beats aiter's runnable Triton kernels by ~2.2×", not "beats aiter's ASM peak".

## Another optimization round?
Not pursued: our kernels already exceed every **runnable** aiter kernel by ~1.9–3.4×, and
the earlier 12h campaign already found the Triton config ceiling (two generations agree).
The remaining headroom is structural — aiter's ASM GEMM path (needs a writable aiter build)
for o_proj, and cross-request batching for dsa_attn — neither reachable by more Triton
autotuning. A hybrid dispatch to aiter buys nothing here since ours wins on every runnable
shape. If aiter's ASM path is built on this node later, re-run `aiter_baseline.py` to
compare against that peak and decide if an ASM-targeting round is warranted.
