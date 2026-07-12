# Kimi 分层算子覆盖说明

## 层类型与 shape 是否相同？

| 层类型 | 层号 (Kimi K2) | FFN 算子 | shape 特点 |
|--------|----------------|----------|------------|
| **Dense FFN** | layer 0 (`first_k_dense_replace=1`) | `DeepseekV2MLP` GateUp/Down | `intermediate_size=18432` → TP8: GateUp `[M,7168]×[7168,4608]` |
| **MoE 层** | layer 1–60 | Router + GroupGEMM + Shared | `moe_inter=2048`, **384 experts** |
| **栈算子** | 每层 / 首尾 | RMSNorm / Embed / LMHead | 与 hidden 绑定，不随层变 |

**Attention/MLA/Indexer**：同一模型内各 MoE 层 shape 相同；仅 `index_topk_freq>1` 时部分层 **跳过 Index_Score**（复用上一层 topk）。

---

## 新增实测（`kimi_layer_coverage.json`）

### Kimi K2 官方配置（hidden=7168, TP=8）

| 层类型 | 算子 | Prefill µs | Decode µs | shape |
|--------|------|----------:|----------:|-------|
| Dense FFN | GateUp | 1045 | 57 | `[16384,7168]×[7168,4608]` |
| Dense FFN | Down | 451 | 53 | `[16384,2304]×[2304,7168]` |
| MoE Shared | GateUp | 148 | 56 | `[M,7168]×[7168,512]` |
| MoE Shared | Down | 240 | 51 | `[M,256]×[256,7168]` |
| MoE Routed | GateUp GroupGEMM | 898 | — | `384×[512,7168]×[7168,256]` |
| 全层 | RMSNorm | 168 | 49 | `[M,7168]` |
| 输入 | Embedding | 242 | 55 | vocab=163840 |
| 输出 | LM Head | 58362 | 893 | `[M,7168]×[7168,163840]` |

### 对比：MoE 层已测的 Shared Expert（原 harness op9/10）

原 harness 用 `hidden=6144`、GateUp N=6144（对应 **shared 更大中间维** 的部署假设），与官方 Kimi shared（N=512）不同：

| 场景 | Shared GateUp shape | Prefill µs |
|------|---------------------|----------:|
| harness op9 | `[16384,6144]×[6144,6144]` | 996 |
| **Kimi 官方** | `[16384,7168]×[7168,512]` | **148** |

Dense FFN 比 MoE Shared **大得多**（GateUp N=4608 vs 512），是 layer 0 的主要算力开销。

---

## 完整覆盖清单

1. **MoE+DSA 热路径 25 项** → `kimi_real_kernel.json`（一层代表所有 MoE 层）
2. **Dense FFN + 栈算子 15 项** → `kimi_layer_coverage.json`（Kimi K2 官方维）
3. **合并表** → `kimi_full_layer_coverage.xlsx`

## sglang API 速查

| 算子 | 入口 |
|------|------|
| Dense FFN | `deepseek_v2.py` → `DeepseekV2MLP`（`layer_id < first_k_dense_replace`） |
| MoE FFN | `DeepseekV2MoE` → `mlp.experts` / `shared_experts` |
| RMSNorm | `input_layernorm` / `post_attention_layernorm` |
| Embed / LMHead | `embed_tokens` / `lm_head` |
