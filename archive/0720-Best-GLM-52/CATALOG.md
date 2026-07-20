# Catalog — all agent best candidates

| Directory under `best/` | Kind | Has `candidate/` |
|---|---|:---:|
| `q_b_decode` | TARGET_MET | yes |
| `moe_up_proj_decode_hbm40` | TARGET_MET | yes |
| `moe_gate_proj_decode_hbm40` | TARGET_MET | yes |
| `moe_down_proj_decode_hbm40` | TARGET_MET | yes |
| `o_proj_decode_hbm35` | TARGET_MET | yes |
| `o_proj_decode` | WIN | yes |
| `index_q_upproj_decode_hbm15` | WIN_MISS_TARGET | yes |
| `index_k_prefill_bw70` | WIN_MISS_TARGET | yes |
| `moe_down_proj_prefill_mfu65` | PARTIAL_WIN | yes |
| `o_proj_prefill` | PARTIAL_WIN | yes |
| `o_proj_decode_hbm40_extreme` | NO_GO | yes |
| `moe_gate_proj_prefill_mfu` | NO_GO | yes |
| `index_score_decode_hbm82` | NO_GO | yes |
| `dsa_attn_decode_hbm40` | NO_GO | yes |
| `absorbed_W_UV_decode_hbm86` | NO_GO | yes |

机器可读：`manifest.json`。
