# NOTE — baseline vs E2E (2026-07-22)

本 CSV（`amd-glm-object.csv`）记录的是 **单卡微基准层占比 >5%** 的 campaign 对象
（`o_proj` / `index_k` / decode `dsa_attn`），baseline 列为 aiter **Triton/plain** 路径。

**不要**把它直接当成 TP=8 生产 e2e 的瓶颈排序：

| E2E (TP=8 short prefill) | 本 CSV campaign |
|--------------------------|-----------------|
| AllReduce ~35% | 不在范围 (TP1) |
| MoE fused ~16% | 非主攻 |
| dense `mla_dec_stage1` ~14% | 主攻是 sparse decode MLA |
| FP8 GEMM 合计 ~11%（已 ASM） | o_proj/index_k 按更软 baseline 宣称 2× |

完整分析与 API 地图：`e2e_profile_20260722/`。
