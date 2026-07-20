# Leaderboard — Best GLM-5.2 decode results (2026-07-20)

Sorted by **achieved HBM%** on the binding shape (usually the worse of M16/M32).

| Rank | Op | Best HBM% (M16 / M32) | Latency µs | vs stock/ref | Mechanism class |
|---:|---|---|---|---|---|
| 1 | moe_up_proj_decode | **41.33 / 41.74** | 30.86 / 30.95 | ~1.53× | candidate pack + PDL |
| 2 | moe_gate_proj_decode | **41.21 / 41.64** | 30.94 / 31.03 | ~1.51× | candidate pack + PDL |
| 3 | moe_down_proj_decode | **40.93 / 41.56** | 31.31 / 31.39 | ~1.50× | candidate pack |
| 4 | o_proj_decode | **38.27 / 37.54** | 33.04 / 33.84 | vs f32-scale ref | candidate pack |
| 5 | q_b_decode | **36.30 / 36.52** | 11.75 / 11.87 | ~2.9× vs stock | **DeepGEMM fork fused** |
| 6 | index_q_upproj_decode | **14.59 / 12.90** | 7.33 / 8.44 | ~3.1× | TMA split-K Triton（目标 15% NO-GO） |

## DeepGEMM source improvement (q_b only)

| Stage | HBM% | Note |
|---|---|---|
| Stock | ~29% | separate UE8M0 pack |
| Fork A4 sms-clamp | ~29.3% | >25% |
| Fork A7 fused pack | **~36.4%** | **current best** |

## Physical no-gos (do not re-target same bar)

| Op | Failed target | Binding reason |
|---|---|---|
| absorbed_W_UV_decode | >86% | span floor ≫ ceiling |
| dsa_attn_decode | >40% | M16 gather floor ≈ ceiling |
| index_score_decode | ≥82% | NCU kernel already > limit |
| o_proj @40% | >40% | packed path plateau ~38% |
| index_q @15% | >15% | 2D tiled W-read floor <15% |
