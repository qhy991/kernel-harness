# Kimi K2.x 实际算子规模与 B200 实测

配置来源：`shapes.py`（Kimi K2.x / DeepSeek-V3.2 DSA 同族）

- hidden=6144, heads=64, q_lora=2048, kv_lora=512
- qk_nope=192, qk_rope=64, v=256 → head_dim=256
- index_heads=32, index_dim=128, topk=2048
- experts=256, moe_intermediate=2048
- Prefill：M=16384（TP=8 分片后的本地 token）
- Decode：M_local=16（DP32×EP32）

说明：xlsx 是 GLM-5.2 / DeepSeek-V4-Pro，**不是** Kimi 清单。下面只列 Kimi 一层里会打到的算子。

---

## Prefill（M=16384）

| # | 算子 | 规模 | latency | TFLOPS | 后端 |
|--:|------|------|--------:|-------:|------|
| 1 | Q_a+KV_a fused | `[16384,6144]×[6144,2624]` | 444.30 µs | 1218 | DeepGEMM FP8 |
| 2 | Q_b | `[16384,2048]×[2048,2048]` (TP 8 heads) | 151.80 µs | 905 | DeepGEMM FP8 |
| 4 | KV_b | `[16384,512]×[512,3584]` | 149.03 µs | 403 | DeepGEMM FP8 |
| 27 | FlashAttn MHA | Q/K/V `[1,8,16384,256]` causal | 7172 µs | 153 | flash_attn 2.8.3（flashinfer 7952µs/138） |
| 5 | O_proj | `[16384,2048]×[2048,6144]` | 354.61 µs | 1163 | DeepGEMM FP8 |
| 6 | Index_Q | `[16384,6144]×[6144,4096]` | 693.66 µs | 1189 | DeepGEMM FP8 |
| 7 | Index_K | `[16384,6144]×[6144,128]` | 66.54 µs | 387 | DeepGEMM FP8 |
| 8 | Index_Score | Q`[16384,32,128]` × K`[16384,128]` → topk=2048 | 2668.81 µs | 824 | `fp8_mqa_logits` |
| 9 | Shared GateUp | `[16384,6144]×[6144,6144]` | 996.20 µs | 1242 | DeepGEMM FP8 |
| 10 | Shared Down | `[16384,1536]×[1536,6144]` | 327.85 µs | 943 | DeepGEMM FP8 |
| 11 | MoE Router | `[16384,6144]×[6144,256]` | 72.48 µs | 711 | DeepGEMM FP8 |
| 12 | MoE GateUp | 256 experts × `[512,6144]×[6144,512]` | 902.02 µs | 914 | grouped FP8 |
| 13 | MoE Down | 256 experts × `[512,256]×[256,6144]` | 1546.81 µs | 267 | grouped FP8 |

Prefill 不单独测：op3 KV_a（并进 op1）。

---

## Decode（M=16）

| # | 算子 | 规模 | latency | TFLOPS | 后端 |
|--:|------|------|--------:|-------:|------|
| 14 | Q_a+KV_a fused | `[16,6144]×[6144,2624]` | 55.54 µs | 9.5 | DeepGEMM FP8 |
| 15 | Q_b | `[16,2048]×[2048,16384]` (64 heads) | 53.70 µs | 20.0 | DeepGEMM FP8 |
| 17 | q_nope absorb | `[16,64,192]×[64,192,512]` | 50.23 µs | 4.0 | torch.bmm |
| 18 | v absorb | `[16,64,512]×[64,512,256]` | 50.77 µs | 5.3 | torch.bmm |
| 22 | Index_Score | Q`[16,32,128]` × K`[2048,128]` topk=2048 | 125.21 µs | 2.1 | `fp8_mqa_logits`* |
| 26 | Sparse MLA | Q`[16,64,512]` × topk=2048 / pool=8192 | 157.57 µs | — | `flash_mla_sparse_fwd` |
| 19 | O_proj | `[16,16384]×[16384,6144]` | 67.89 µs | 47.5 | DeepGEMM FP8 |
| 20 | Index_Q | `[16,6144]×[6144,4096]` | 55.57 µs | 14.5 | DeepGEMM FP8 |
| 21 | Index_K | `[16,6144]×[6144,128]` | 54.88 µs | 0.5 | DeepGEMM FP8 |
| 23 | MoE Router | `[16,6144]×[6144,256]` | 55.22 µs | 0.9 | DeepGEMM FP8 |
| 24 | MoE GateUp | 8 × `[16,6144]×[6144,4096]` | 147.48 µs | 43.7 | grouped FP8† |
| 25 | MoE Down | 8 × `[16,2048]×[2048,6144]` | 140.92 µs | 22.9 | grouped FP8† |

Decode 不测：op16 KV_a（并进 op14）；op4 KV_b（decode 已吸收进 17/18）。

\* 生产路径是 `fp8_paged_mqa_logits`；本次用 ragged API 同规模近似。  
† 生产 decode 倾向 `fp8_m_grouped_gemm_nt_masked`；本次用 contiguous grouped 同规模近似。  
Decode 小 M 有 ~50µs launch 噪声，TFLOPS 仅供参考。

---

## 覆盖结论

- Kimi 一层热路径：**25 个独立规模全部已测**。
- op27：B200 上用 `flashinfer.single_prefill_with_kv_cache`（7952 µs / 138 TFLOPS）；Dao `flash-attn` 正在后台源码编译（sm80/90/100/120 全架构，约 15–30 分钟）。
- 不在范围内：MiniMax ops 28–43；xlsx 里的 GLM / DSV4 算子。
