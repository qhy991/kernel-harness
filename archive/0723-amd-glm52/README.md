# 0723-amd-glm52 — MI300X GLM-5.2 optimized operators

Our 8 tuned operators for GLM-5.2 on **MI300X (gfx942)**, one file per (operator, shape).
Baseline for the speedups here is **aiter's production kernels** (the sglang-ROCm path:
`gemm_a8w8_blockscale` for the GEMMs, `pa_decode_sparse` for the sparse-MLA decode) — the
Triton path that runs on this node. Each operator passed the opbench correctness gate
(`calc_diff < 5e-6` vs a bf16 dequant / full-attention oracle).

## The 8 operators

| file | operator | shape | lat (µs) | aiter (µs) | **vs aiter** | vs ref | roofline |
|---|---|---|---|---|---|---|---|
| `o_proj_prefill_m1024.py` | Attention O Proj | M=1024 | 428 | 946 | **2.21×** | 3.64× | 16% (compute) |
| `o_proj_prefill_m2048.py` | Attention O Proj | M=2048 | 904 | 1913 | **2.12×** | 3.50× | 17% (compute) |
| `o_proj_prefill_m4096.py` | Attention O Proj | M=4096 | 1326 | 3162 | **2.38×** | 4.47× | 23% (compute) |
| `index_k_prefill_m1024.py` | DSA Index K Proj | M=1024 | 289 | 568 | **1.97×** | 2.93× | 29% (memory) |
| `index_k_prefill_m2048.py` | DSA Index K Proj | M=2048 | 287 | 540 | **1.88×** | 2.93× | 29% (memory) |
| `index_k_prefill_m4096.py` | DSA Index K Proj | M=4096 | 286 | 540 | **1.88×** | 2.95× | 29% (memory) |
| `dsa_attn_decode_bs16.py` | MLA Decode Attn | BS=16 | 123 | 274 | **2.23×** | 2.09× | 6% (memory) |
| `dsa_attn_decode_bs32.py` | MLA Decode Attn | BS=32 | 128 | 433 | **3.38×** | 2.97× | 12% (memory) |

**Geomean 2.22× over the aiter baseline.** (`index_k` prefill projects all S=65536 KV
tokens, so its cost is independent of the prefill chunk M — the three rows are the same
`[65536,6144]×[6144,128]` GEMM; only the layer-time fraction changes with M.)

## What's the optimization

- **GEMM (o_proj / index_k)** — `variant=fp8_dot`: native fp8 `e4m3fnuz` `tl.dot` on the
  2.6 PF matrix core (vs bf16-upcast at 1.3 PF), tuned per shape with the AMD MFMA knobs
  `waves_per_eu` / `matrix_instr_nonkdim` / `kpack`.
- **sparse MLA decode (dsa_attn)** — `variant=flash_split`: tk-split flash-DECODING to fill
  the CU grid at tiny decode batch, plus a fused combine kernel; f32-accumulated QK
  (bf16-rounded logits fail the 5e-6 gate).

## Layout / usage

- `_amd_kernels.py` — shared Triton kernels + `gemm_factory` / `dsa_factory` (byte-identical
  to `rewardbench/amd/tuned/`, the campaign winners).
- Each operator file exposes `run(inputs)`, plus `TARGET` (op/phase/shape) and `META`
  (measured latency, speedups, roofline, tuned config). `inputs` are the frozen
  `testbench/harness/glm52_ops.py` tensors for that (op, phase, shape).
- `manifest.json` — machine-readable index of all 8.

Provenance/replay only — not part of the live harness contract. Reproduce the head-to-head
vs aiter with `rewardbench/amd/aiter_baseline.py`. Measured on MI300X gfx942, HIP-event
cold-L2 median.
