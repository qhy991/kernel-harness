# Catalog — Codex goal ultra-xhigh archive

Source runner: `/home/qinhaiyan/glm52-goal-runs` (2026-07-22..23).
Model pin: `gpt-5.6-sol` + `model_reasoning_effort=ultra` (see runner README).

This archive keeps **custom candidates**, **REPORT/FINAL text**, and **evidence text**.
It does **not** copy multi-GB nsys/ncu traces or nested `sglang/` trees.

| ID | Goal | Disposition | Status | Candidates | Reports | Evidence files | Harness tip |
|---:|---|---|---|---:|---:|---:|---|
| 01 | `01-dsa_decode_value_path` | no-replacement | analyzed | 0 | 1 | 0 | `5c359d1` |
| 02 | `02-dsa_decode_attention` | no-replacement | analyzed | 7 | 2 | 0 | `d91dc16` |
| 03 | `03-dsa_prefill_attention` | local-reject + blocked | analyzed | 2 | 3 | 41 | `640f7e4` |
| 04 | `04-dsa_decode_score_path` | no-replacement | analyzed | 0 | 1 | 0 | `9d2f5db` |
| 05 | `05-indexer_k_weights_prefill` | no-replacement (inner-gate) | analyzed | 4 | 1 | 163 | `1e38cef` |
| 06 | `06-moe_w2_decode_pack_launch` | local no-replacement + blocked | analyzed | 3 | 0 | 0 | `8e6d8ca` |
| 07 | `07-moe_w2_decode_kernel` | leaf-win + no-replacement | analyzed | 0 | 0 | 62 | `8bca9f9` |
| 08 | `08-moe_w2_prefill` | leaf-win + no-replacement | analyzed | 5 | 4 | 2 | `10190d8` |
| 09 | `09-moe_w13_prefill` | pending analysis | pending | 5 | 1 | 1 | `3d363b7` |
| 10 | `10-attn_o_decode_baseline` | pending analysis | pending | 0 | 1 | 0 | `229f4dd` |
| 11 | `11-attn_o_decode_packed_port` | thin / little local artifact | sparse | 0 | 0 | 0 | `bcd0054` |
| 12 | `12-attn_o_decode_source_tuning` | thin / little local artifact | sparse | 0 | 0 | 0 | `bcd0054` |
| 13 | `13-attn_o_prefill` | pending analysis | pending | 0 | 1 | 10 | `0d2d8d2` |
| 14 | `14-attn_q_b_decode_packed` | pending analysis | pending | 2 | 1 | 7 | `93fc189` |
| 15 | `15-indexer_wq_b_decode` | pending analysis | pending | 4 | 1 | 0 | `c8c9cb9` |
| 16 | `16-indexer_score_decode` | pending analysis | pending | 2 | 1 | 206 | `768eecf` |
| 17 | `17-indexer_score_prefill` | pending analysis | pending | 6 | 0 | 83 | `bc45ddc` |
| 18 | `18-moe_w13_decode_scale_path` | pending analysis | pending | 4 | 1 | 1 | `a980a91` |
| 19 | `19-moe_w13_decode_kernel` | pending analysis | pending | 4 | 0 | 1 | `b941f5f` |
| 20 | `20-moe_w13_prefill_graph` | thin / little local artifact | sparse | 0 | 0 | 0 | `bcd0054` |
| 21 | `21-attn_q_b_decode_source_fork` | thin / little local artifact | sparse | 0 | 0 | 0 | `bcd0054` |
| 22 | `22-dsa_flashmla_kv_production` | no-replacement | analyzed | 1 | 2 | 0 | `660f88e` |
| 23 | `23-dp_allgather_production` | no-replacement + blocked | analyzed | 2 | 1 | 0 | `fdc227a` |
| 24 | `24-tp_allreduce_reachability` | in-progress | in-progress | 0 | 0 | 0 | `bea4a8c` |
| 25 | `25-deepep_dispatch_combine` | no-replacement | analyzed | 1 | 0 | 0 | `ab78c4c` |

## Highlights

- **leaf-win (not production-swapped):** goal-07 (DeepGEMM BM16), goal-08 (`moe_w2_contig_psum.py`).
- **Analyzed no-replacement set:** 01–06, 22, 23, 25 (see `EXPERIMENT_RESULTS.md`).
- **Sparse stubs** (worktree present, little archived artifact): 11, 12, 20, 21; 24 in-progress.

## Per-goal candidate names

- `02-dsa_decode_attention`: `dsa_direct_trtllm.py`, `dsa_flashmla_kv_bank_layout.py`, `dsa_flashmla_kv_sched_override.py`, `dsa_pdl_disabled.py`, `dsa_preallocated_out.py`, `dsa_split9/`, `dsa_tensor_bmm1_scale.py`
- `03-dsa_prefill_attention`: `dsa_prefill_pdl_off.py`, `dsa_prefill_swaps_tactic/`
- `05-indexer_k_weights_prefill`: `indexer_single_stream.py`, `indexer_wk_cutedsl_tgv.py`, `indexer_wk_flashinfer.py`, `indexer_wk_torch_mm.py`
- `06-moe_w2_decode_pack_launch`: `moe_w2_direct_launch.py`, `moe_w2_registry_integration.py`, `moe_w2_reuse_output_floor.py`
- `08-moe_w2_prefill`: `_moe_w2_contig_psum.py`, `moe_w2_contig_psum.py`, `moe_w2_contig_psum_expected_none.py`, `moe_w2_contig_psum_mnk.py`, `moe_w2_contig_psum_zero_padding.py`
- `09-moe_w13_prefill`: `_moe_w13_contig_psum.py`, `moe_w13_contig_psum.py`, `moe_w13_contig_psum_expected_none.py`, `moe_w13_contig_psum_mnk.py`, `moe_w13_contig_psum_zero_padding.py`
- `14-attn_q_b_decode_packed`: `q_b_packed_warp.py`, `q_b_packed_warp_static_nk.py`
- `15-indexer_wq_b_decode`: `indexer_wq_b_packed.py`, `indexer_wq_b_region_sglang.py`, `indexer_wq_b_sglang_dispatch.py`, `indexer_wq_b_sms_dispatch.py`
- `16-indexer_score_decode`: `indexer_region_cutedsl.py`, `indexer_score_cutedsl.py`
- `17-indexer_score_prefill`: `indexer_score_balanced_chunks.py`, `indexer_score_balanced_mixed_bucket.py`, `indexer_score_gather_b128_w4.py`, `indexer_score_gather_b128_w8.py`, `indexer_score_gather_b64_w4.py`, `indexer_score_gather_tuned.py`
- `18-moe_w13_decode_scale_path`: `moe_w13_direct_default_args.py`, `moe_w13_direct_launch.py`, `moe_w13_registry_integration.py`, `moe_w13_reuse_output_floor.py`
- `19-moe_w13_decode_kernel`: `moe_w13_deepgemm_overlay.py`, `moe_w13_deepgemm_overlay_ep4_region.py`, `moe_w13_deepgemm_overlay_region.py`, `moe_w13_deepgemm_policy.py`
- `22-dsa_flashmla_kv_production`: `flashmla_goal22_overlay.py`
- `23-dp_allgather_production`: `allgather_grouped_broadcast.py`, `allgather_pynccl_native.py`
- `25-deepep_dispatch_combine`: `deepep_config_env.py`
