# GLM-5.2 production optimization goals

This directory contains goal files that are ready to run from Codex Goal mode.
Each goal targets the operator that current SGLang actually executes and uses the
production-facing suite in `serving_native/` as the primary microbenchmark.  The
frozen synthetic tasks remain useful for candidate development, but they are not
production baselines.

Start one goal per chat and give concurrent goals separate Git worktrees.  For
example:

```text
/goal Execute /home/qinhaiyan/Kernel-Harness/goal_plans/glm52_production/14_attn_q_b_decode_packed/plan.md completely. Treat that file and /home/qinhaiyan/Kernel-Harness/goal_plans/glm52_production/COMMON_RULES.md as the objective, constraints, and definition of done.
```

The detailed file is intentionally separate from the `/goal` command so the goal
objective stays short.  Do not run two goals that can edit the same checkout at
the same time.

## Goal index

| # | Converted logical focus | Goal file | Production target |
|---:|---|---|---|
| 01 | absorbed W-UV decode | `01_dsa_decode_value_path/plan.md` | Fused DSA decode value path in `flashmla_kv` |
| 02 | DSA attention decode | `02_dsa_decode_attention/plan.md` | End-to-end `flashmla_kv` decode kernels |
| 03 | DSA attention prefill | `03_dsa_prefill_attention/plan.md` | Runtime-selected DSA prefill backend |
| 04 | absorbed W-UK decode | `04_dsa_decode_score_path/plan.md` | Fused `flashmla_kv` score/QK path |
| 05 | indexer K prefill | `05_indexer_k_weights_prefill/plan.md` | Fused BF16 `wk_weights_proj` and K-cache preparation |
| 06 | MoE down decode overhead | `06_moe_w2_decode_pack_launch/plan.md` | W2 decode pack and launch overhead inside the DeepEP region |
| 07 | MoE down decode kernel | `07_moe_w2_decode_kernel/plan.md` | Production packed W2 grouped GEMM |
| 08 | MoE down prefill | `08_moe_w2_prefill/plan.md` | Production W2 prefill after normal DeepEP dispatch |
| 09 | MoE gate prefill | `09_moe_w13_prefill/plan.md` | Fused W13 prefill grouped GEMM |
| 10 | O projection decode | `10_attn_o_decode_baseline/plan.md` | O-projection production baseline and safe replacement |
| 11 | O projection decode packed port | `11_attn_o_decode_packed_port/plan.md` | Native packed O-projection candidate port |
| 12 | O projection decode source tuning | `12_attn_o_decode_source_tuning/plan.md` | DeepGEMM source tuning for small-M O projection |
| 13 | O projection prefill | `13_attn_o_prefill/plan.md` | Production packed O-projection prefill |
| 14 | attention Q-B decode packed path | `14_attn_q_b_decode_packed/plan.md` | Attention Q-B native packed decode GEMM |
| 15 | indexer Q up-projection decode | `15_indexer_wq_b_decode/plan.md` | Indexer `wq_b`, distinct from attention Q-B |
| 16 | indexer score decode | `16_indexer_score_decode/plan.md` | Runtime-selected paged MQA score path |
| 17 | indexer score prefill | `17_indexer_score_prefill/plan.md` | Runtime-selected dense/chunked MQA score path |
| 18 | MoE gate decode | `18_moe_w13_decode_scale_path/plan.md` | Fused W13 decode scale and launch path |
| 19 | MoE up decode | `19_moe_w13_decode_kernel/plan.md` | Fused W13 decode grouped-GEMM source tuning |
| 20 | MoE up prefill | `20_moe_w13_prefill_graph/plan.md` | W13 prefill with graph and full-region validation |
| 21 | attention Q-B source fork | `21_attn_q_b_decode_source_fork/plan.md` | Isolated DeepGEMM source optimization for Q-B |
| 22 | production FlashMLA decode | `22_dsa_flashmla_kv_production/plan.md` | Unified `flashmla_kv` split-KV, combine, and scheduler optimization |
| 23 | production DP AllGather | `23_dp_allgather_production/plan.md` | SGLang AllGather backend, launch, and overlap at fixed decode/prefill shapes |
| 24 | TP AllReduce reachability | `24_tp_allreduce_reachability/plan.md` | Runtime-selected SGLang AllReduce implementation and four-rank diagnostics |
| 25 | DeepEP communication | `25_deepep_dispatch_combine/plan.md` | Low-latency decode and normal prefill dispatch/combine |

## Shared production and communication coverage

All goals inherit `COMMON_RULES.md`.  In particular:

- Decode tests keep local `M=16` and `M=32`; DP8 does not divide them.
- Eight-rank AllGather and DeepEP tasks are the production topology.
- Four-rank AllReduce, AllGather, and DeepEP tasks are a diagnostic TP4/DP4/EP4
  lane and cannot prove an eight-rank deployment win.
- A winning bucket may be enabled alone while a losing bucket keeps the stock
  SGLang implementation.
- Source changes to SGLang and relevant open-source kernel libraries are allowed
  when isolated, versioned, and validated through the production ABI.

Because the observed serving bottlenecks are AllGather and DeepEP, measure these
before starting a compute-kernel goal and again during its full-region acceptance:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 serving_native/run.sh dp_allgather_decode_m16 --candidate serving_native/candidates/allgather_torch.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 serving_native/run.sh deepep_ll_dispatch_decode_m16 --candidate serving_native/candidates/reference.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 serving_native/run.sh deepep_ll_combine_decode_m16 --candidate serving_native/candidates/reference.py
```

The independent four-GPU diagnostic lane is available for backend/config search:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 serving_native/run.sh tp4_allreduce_decode_m16 --candidate serving_native/candidates/allreduce_torch.py
CUDA_VISIBLE_DEVICES=0,1,2,3 serving_native/run.sh tp4_allgather_decode_m16 --candidate serving_native/candidates/allgather_torch.py
CUDA_VISIBLE_DEVICES=0,1,2,3 serving_native/run.sh ep4_deepep_ll_dispatch_decode_m16 --candidate serving_native/candidates/reference.py
```

Repeat all decode commands at M32.  Four-GPU results select diagnostic candidates;
they do not replace the required eight-rank production validation.
