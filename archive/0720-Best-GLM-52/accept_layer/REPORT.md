# accept_layer — archive best candidates layer swap

Advisory acceptance via `testbench/bin/accept_layer.py` (not the primary gate).
Timing: CUPTI cold-L2 device-kernel median, warmup=3, iterations=10, idle GPU 0.
Phase: **decode**. Layer = sum of the 12 glm52 ops in `ALL_OPS`.

## Headline

| Scenario | M | Backend layer | Swapped layer | Layer speedup | Time saved |
|---|---:|---:|---:|---:|---:|
| All archived decode swaps (9 ops) | 16 | 432.6 µs | 309.4 µs | **1.40×** | **−28.5%** |
| All archived decode swaps (9 ops) | 32 | 447.2 µs | 358.9 µs | **1.25×** | **−19.7%** |
| **Winners only** (5 ops) | 16 | 406.7 µs | 311.7 µs | **1.31×** | **−23.4%** |
| **Winners only** (5 ops) | 32 | 436.4 µs | 342.1 µs | **1.28×** | **−21.6%** |

**Recommended expected gain (stable winners only):** about **1.28–1.31×** end-to-end decode layer, i.e. **~22–23%** lower device-kernel layer time.

Winners-only swaps:
- `q_b` ← `best/q_b_decode` (DeepGEMM fork fused)
- `o_proj` ← `best/o_proj_decode_hbm35`
- `index_q_upproj` ← `best/index_q_upproj_decode_hbm15`
- `moe_gate` ← `best/moe_gate_proj_decode_hbm40`
- `moe_up` ← `best/moe_up_proj_decode_hbm40`

Unswapped (no better archive / kept reference): `fused_qkv_a`, `index_k`, `absorbed_W_UK`, and in winners-only also `moe_down`, `dsa_attn`, `index_score`, `absorbed_W_UV`.

## Per-op contribution (winners-only, illustrative)

### M=16
| op | backend µs | candidate µs | speedup | save µs |
|---|---:|---:|---:|---:|
| o_proj | 53.9 | 33.7 | 1.60× | 20.3 |
| moe_up | 47.6 | 31.0 | 1.54× | 16.7 |
| moe_gate | 47.6 | 30.9 | 1.54× | 16.7 |
| q_b | 35.6 | 11.9 | 2.99× | 23.7 |
| index_q_upproj | 25.1 | 7.4 | 3.41× | 17.8 |

### M=32
| op | backend µs | candidate µs | speedup | save µs |
|---|---:|---:|---:|---:|
| o_proj | 54.7 | 34.7 | 1.58× | 20.0 |
| moe_up | 47.5 | 31.2 | 1.52× | 16.3 |
| moe_gate | 47.3 | 31.0 | 1.53× | 16.3 |
| q_b | 36.0 | 12.0 | 2.99× | 23.9 |
| index_q_upproj | 26.3 | 8.5 | 3.10× | 17.8 |

## Caveats
- Advisory only; primary gate remains per-task `run.sh`.
- Absolute layer totals vary slightly across runs (GPU boost / noise); ratios are the signal.
- `moe_down` archive candidate was **unstable** in the full swap (M16 ~1.33×, M32 ~0.76× regression) — excluded from winners-only until re-validated.
- Stock-like archives (`dsa_attn`, `index_score`, `absorbed_W_UV`) contribute ~0.
- Prefill layer not measured in this report (decode focus).

## Reproduce

```bash
BEST=archive/0720-Best-GLM-52/best
CUDA_VISIBLE_DEVICES=0 .venv/bin/python testbench/bin/accept_layer.py --M 32 \
  --swap q_b=$BEST/q_b_decode/candidate \
  --swap o_proj=$BEST/o_proj_decode_hbm35/candidate \
  --swap index_q_upproj=$BEST/index_q_upproj_decode_hbm15/candidate \
  --swap moe_gate=$BEST/moe_gate_proj_decode_hbm40/candidate \
  --swap moe_up=$BEST/moe_up_proj_decode_hbm40/candidate
```
