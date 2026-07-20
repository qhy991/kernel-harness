# PR#5 (best-hechenxi-0720) vs 当前 best/ — Decode CUPTI 对打

协议：CUPTI cold-L2，`run.sh --candidate`，B200，M∈{16,32}。
判据：**绝对 candidate 中位 µs**（跨 GPU 的 reference/speedup 会漂，不作主判据）。
两边均数值正确；DSA 两边 exit=1 仅因未达 >40% HBM 硬目标，但可比较绝对延时。

| op | ours µs M16/M32 | PR5 µs M16/M32 | ours/PR5 | 胜方 | ours 机制 | PR5 机制 |
|----|----------------:|---------------:|---------:|------|-----------|----------|
| `fused_qkv_a_decode` | 17.7/18.3 | 8.9/10.6 | 1.99×/1.72× | **PR5** | DeepGEMM fork fused | Triton no-TMA split-K |
| `q_b_decode` | 11.7/11.8 | 13.7/14.7 | 0.86×/0.81× | **ours** | DeepGEMM fork fused | UE8M0 weight-only prepack |
| `moe_gate_proj_decode` | 30.8/30.7 | 30.7/30.5 | 1.00×/1.01× | 平手 | pack+PDL | pack+PDL weightonly |
| `moe_up_proj_decode` | 31.0/31.1 | 30.6/30.7 | 1.01×/1.01× | 平手 | pack+PDL | pack+PDL weightonly |
| `index_q_upproj_decode` | 7.1/8.2 | 6.4/7.5 | 1.12×/1.09× | **PR5** | Triton split-K | Triton no-TMA |
| `dsa_attn_decode` | 44.8/46.4 | 27.3/37.1 | 1.64×/1.25× | **PR5** | stock FlashMLA | flashinfer trtllm-gen |

## 结论

| 胜方 | 算子 |
|------|------|
| **换 PR5** | `fused_qkv_a_decode`（~2×）、`index_q_upproj_decode`（~1.1×）、`dsa_attn_decode`（~1.4–1.6×，flashinfer trtllm） |
| **留 ours** | `q_b_decode`（DeepGEMM fork fused，~1.2× 快于 PR5） |
| **平手** | `moe_gate_proj_decode` / `moe_up_proj_decode`（差 <2%） |

### 覆盖差异

- **PR5 独有：** `index_weights_proj_decode`（fuse wk+weights，README ≥1.65×；无独立 harness task，未 CUPTI 对打）
- **ours 独有：** `o_proj`、`moe_down`、`index_k`、以及全部 prefill winners

### 建议晋升到 DECODE_SWAPS

1. `fused_qkv_a_proj` ← `best-hechenxi-0720/fused_qkv_a_decode` — **已替换**
2. `index_q_upproj` ← `best-hechenxi-0720/index_q_upproj_decode` — **已替换**
3. `dsa_decode_attn` ← `best-hechenxi-0720/dsa_decode_attn` — **已替换**
4. `q_b_proj` 继续 `best/q_b_decode` — **保持**

Decode 层（CUDA Graph drop-in）：**1.75× / 1.55×**（M16/M32）。

原始日志：`results/pr5_vs_ours/`
