# GLM-5 算子性能测试 Reward 指标设计

配套脚本：`bench_GLM5_ops_prefill.py` / `bench_GLM5_ops_decode.py`（共享 `glm5_ops_common.py`）。
13 个算子清单与 `llm_flops/bench_glm5_prefill.py`、`bench_glm5_decode.py` 一致；**后端 kernel 与 dtype 已按
GLM-5.2 在 B200 上的真实运行时逐个核对对齐**（对 `算子baseline确认.xlsx` 中标注的疑点用 sglang 源码裁决，见文末）。

## Reward 总体设计（依据 SGLang 官方实践建议 6 条 Rule，尤其 Rule 3「按 bound 读」）

**reward = bound-aware roofline 利用率 ∈ [0,1] = 实测吞吐 ÷ roofline 天花板**
- AI = FLOPs / HBM字节；ridge = 峰值算力 ÷ 8TB/s；AI≥ridge→compute-bound，否则 memory-bound（自动判定）。
- roofline 天花板 = min(峰值算力, AI×8TB/s)（Williams roofline）；reward = 实测算力 ÷ 天花板。
  - compute-bound ⇒ reward ≡ Tensor-Core 利用率；memory-bound ⇒ reward ≡ HBM 带宽利用率。
- 同一算子在 prefill(大 M) 多 compute-bound、decode(小 batch) 多 memory-bound，reward 自动切到「卡脖子」资源，可跨 shape/阶段统一比较；优化时最大化 reward 即逼近 roofline。低 reward(如 index_k decode≈0.05) 直接标注最大 headroom。
- 峰值(B200/SM100)：HBM 8 TB/s；FP8 TC 4.5 PFLOP/s；BF16 TC 2.25 PFLOP/s。ridge：FP8=562.5、BF16=281.25 FLOP/byte。
- 计时：CUDA Graph capture+replay（5 warmup + 20 replays 均值），与 llm_flops 一致，覆盖真实 dispatch 路径（Rule 2「先固定 benchmark」）。
- ABI 对齐（Rule 5）：每算子调用与 sglang 运行时**完全相同**的后端 kernel/dtype/scale/layout。FP8 blockwise 在 B200 用 UE8M0 scale（`DEEPGEMM_SCALE_UE8M0=True`）。
- 每算子输出诊断：latency(ms)、TFLOP/s、GB/s、AI、bound、util%、reward。

## 指标表（util% 为代表性 shape 实测：prefill M=4096 / decode batch=32，S=65536）

| 算子 | 类型（prefill/decode） | 反馈指标设计 | 设计原因 |
|---|---|---|---|
| fused_qkv_a_proj | prefill | FP8 compute 利用率（compute-bound, AI≈2077, ~53%）| deep_gemm.fp8_gemm_nt [M,6144]×[6144,2624]；大 M 计算密集，卡 tensor-core。 |
| q_b_proj | prefill | FP8 compute 利用率（compute-bound, AI≈1558, ~53%）| deep_gemm.fp8_gemm_nt [M,2048]×[2048,16384]，大 N，compute-bound。 |
| absorbed_W_UK | prefill | HBM 带宽利用率（**bf16** memory-bound, AI≈135, ~71%）| **torch.bmm bf16**（B200 上 kv_b_proj 被反量化为 bf16，走 torch.bmm 而非 bmm_fp8）；K=192 小，带宽 bound。 |
| absorbed_W_UV | prefill | HBM 带宽利用率（**bf16** memory-bound, AI≈164, ~79%）| torch.bmm bf16 [64,M,512]×[64,512,256]，搬运主导，带宽 bound。 |
| o_proj | prefill | FP8 compute 利用率（compute-bound, AI≈3745, ~61%）| deep_gemm.fp8_gemm_nt [M,16384]×[16384,6144]，K/N 大，强 compute-bound。 |
| dsa_prefill_attn | prefill | **FP8** HBM 带宽利用率（trtllm-gen, memory-bound, AI≈221, ~72%）| **flashinfer trtllm_batch_decode_with_kv_cache_mla(trtllm-gen, sparse_mla_top_k)** on FP8 KV（B200 默认路径）；每 query 独立读 topk=2048 KV 行(无 dedup)，KV 读主导→带宽 bound。 |
| index_k_proj | prefill | **bf16** HBM 带宽利用率（memory-bound, AI≈125, ~75%）| **torch.F.linear bf16** [S,6144]×[6144,128]（GLM-5.2 indexer 融合路径 wk 走 bf16 F.linear，非 fp8 deep_gemm）；读大激活主导，带宽 bound。 |
| index_q_upproj | prefill | FP8 compute 利用率（compute-bound, AI≈1358, ~40%）| deep_gemm.fp8_gemm_nt [M,2048]×[2048,4096]，M=4096 compute-bound；利用率偏低=有空间。 |
| index_weights_proj | prefill | **bf16→f32** HBM 带宽利用率（memory-bound, AI≈31, ~81%）| **torch.mm bf16-in/f32-out**（cuBLAS，非 deep_gemm.bf16_gemm_nt）[M,6144]×[6144,32]，N=32 极小→带宽 bound。 |
| index_score | prefill | FP8 compute 利用率（compute-bound, AI≈2000, ~44%）| deep_gemm.fp8_mqa_logits，q[M,32,128]·k[S,128]→logits[M,S]，UE8M0；M 大 compute-bound。 |
| moe_gate_proj | prefill | FP8 compute 利用率（compute-bound, AI≈1864, ~57%）| deep_gemm.fp8_m_grouped_gemm_nt_masked [8×,6144]×[6144,2048]，UE8M0；prefill token 多，compute-bound。 |
| moe_up_proj | prefill | FP8 compute 利用率（compute-bound, AI≈1864, ~57%）| 同 gate（运行时与 gate 融合为单个 w13 N=4096 GEMM，此处按 per-op 粒度各计一半）。 |
| moe_down_proj | prefill | FP8 compute 利用率（compute-bound, AI≈1440, ~50%）| deep_gemm.fp8_m_grouped_gemm_nt_masked [8×,2048]×[2048,6144]，compute-bound。 |
| fused_qkv_a_proj | decode | HBM 带宽利用率（memory-bound, AI≈62, ~9%）| skinny-M(batch=32)，读 6144×2624 权重主导→带宽 bound；利用率低=权重流式带宽是瓶颈(大 headroom)。 |
| q_b_proj | decode | HBM 带宽利用率（memory-bound, AI≈62, ~18%）| skinny-M，读 2048×16384 权重主导。 |
| absorbed_W_UK | decode | bf16 HBM 带宽利用率（memory-bound, AI≈26, ~61%）| torch.bmm bf16，batch=64 但每 batch M=32，带宽 bound。 |
| absorbed_W_UV | decode | bf16 HBM 带宽利用率（memory-bound, AI≈27, ~79%）| torch.bmm bf16，带宽 bound。 |
| o_proj | decode | HBM 带宽利用率（memory-bound, AI≈63, ~28%）| skinny-M，读 16384×6144 权重主导（与 prefill 的 compute-bound 相反，reward 自动切换）。 |
| dsa_decode_attn | decode | FP8 HBM 带宽利用率（trtllm-gen, memory-bound, AI≈221, ~30%）| flashinfer trtllm-gen(fp8 KV, sparse_mla_top_k=2048)；每 query 读 topk KV 行，带宽/延迟 bound（B200 默认 decode DSA 路径）。 |
| index_k_proj | decode | bf16 HBM 带宽利用率（memory-bound, AI≈26, ~5% ⇒ launch-bound）| torch.F.linear bf16 [32,6144]×[6144,128]，计算量极小，被启动开销主导；reward≈0 标注最大 headroom，杠杆是融合/降启动。 |
| index_q_upproj | decode | HBM 带宽利用率（memory-bound, AI≈62, ~6%）| skinny-M，启动/带宽 bound（已验证自定义 split-K Triton kernel 可提升~2.5x）。 |
| index_weights_proj | decode | bf16→f32 HBM 带宽利用率（memory-bound, AI≈16, ~2% ⇒ launch-bound）| torch.mm bf16，N=32 计算量极小，launch-bound；杠杆是融合(运行时与 wk 融为 wk_weights_proj)。 |
| index_score | decode | FP8 HBM 带宽利用率（memory-bound, AI≈60, ~73%）| deep_gemm.fp8_paged_mqa_logits，读分页 fp8 KV(M×S×132B) 主导→强 memory-bound；73% 已近 roofline。 |
| moe_gate_proj | decode | HBM 带宽利用率（memory-bound, AI≈62, ~33%）| grouped masked，decode 每专家 token 少→读 8×专家权重(~100MB) 主导，带宽 bound。 |
| moe_up_proj | decode | HBM 带宽利用率（memory-bound, AI≈62, ~33%）| 同 gate（运行时融合为 w13）。 |
| moe_down_proj | decode | HBM 带宽利用率（memory-bound, AI≈62, ~33%）| grouped masked，权重流式带宽 bound（deep_gemm 仅~33% HBM，有 headroom；已验证 warp-specialized Triton 可提升~1.14x）。 |

## 对齐核对与裁决（sglang 源码，adversarial 校验）

`算子baseline确认.xlsx` 的三处疑点已用 GLM-5.2 sglang 源码逐一裁决并据此修正脚本：

1. **absorbed_W_UK / absorbed_W_UV**（xlsx 标"疑似 torch 实现"）→ **裁决：bf16 torch.bmm**。B200 上 `DEEPGEMM_BLACKWELL=True`，`deepseek_weight_loader.py` 走 `block_quant_dequant(..., torch.bfloat16)` 把 kv_b_proj 反量化为 bf16，故 `w_kc/w_vc.dtype==bfloat16`，`forward_mla.py` 取 `torch.bmm` 分支(非 bmm_fp8)。脚本已改用 `torch.bmm` + bf16 峰值(2.25PF)。
2. **index_k_proj (wk) / index_weights_proj**（Sheet1「torch F.linear 融合」vs Sheet2「deep_gemm fp8」冲突）→ **裁决：bf16，融合**。GLM-5.2 config `indexer_rope_interleave=True` ⇒ `is_neox_style=False` ⇒ `dsa_indexer.py:380 use_dsa_indexer_fusion=True`，wk 与 weights_proj 融合为**单个 bf16 `ReplicatedLinear` wk_weights_proj [6144,160]**（128 k + 32 weights），weights_proj 用 `torch.mm(..., out_dtype=float32)`。脚本按 per-op 粒度：index_k=bf16 F.linear[.,6144]→[.,128]、index_weights=bf16→f32 torch.mm[.,6144]→[.,32]（运行时实为二者融合的一次 x 读）。
3. **dsa_prefill_attn / dsa_decode_attn**（xlsx/llm_flops 用 flash_mla_sparse_fwd bf16）→ **裁决：B200 默认 trtllm-gen fp8**。GLM-5.2(GlmMoeDsaForCausalLM) 在 B200 `kv_cache_dtype` auto→fp8_e4m3 ⇒ DSA backend='trtllm' ⇒ `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(backend='trtllm-gen', sparse_mla_top_k=2048)`（`dsa_backend.py:2716`），FP8 query+FP8 分页 KV，sparse top-k 经 `block_tables[seqs,1,topk]` 传入。`flash_mla_sparse_fwd`(bf16) 仅在 `--kv-cache-dtype bfloat16` 时为默认，脚本保留其 builder 作为该配置下的备选(`build_sparse_mla`)。头数 64 无需 pad(trtllm-gen 原生支持 GLM-5 的 num_heads=64)。

其它已核对为**对齐**的项：fused_qkv_a / q_b / o_proj（deep_gemm.fp8_gemm_nt UE8M0，shape 与 forward_mla.py 一致）；index_q_upproj（fp8_gemm_nt）；index_score（prefill fp8_mqa_logits / decode fp8_paged_mqa_logits，UE8M0）；moe gate/up/down（fp8_m_grouped_gemm_nt_masked）。

已修正的正确性问题：
- **moe grouped buffer 越界**：随机 multinomial 的最大 per-expert bin 可能超过 `expected_m`（如 M=4096 时 4179>4096），导致 masked kernel 越界读写。已改为按 `max(ceil(total_m/E), max(counts))` 分配容量。
- **moe gate+up 融合**：运行时 sglang 把 gate+up 融合为单个 w13 grouped GEMM(N=2×2048=4096)；本脚本按 13 算子粒度分别计 gate/up（各为融合 GEMM 的一半，FLOP 等价），已在表中标注。

## 运行

```bash
cd glm5_ops_reward
../kernel-harness/.venv/bin/python bench_GLM5_ops_prefill.py       # M sweep 1024/2048/4096
../kernel-harness/.venv/bin/python bench_GLM5_ops_decode.py        # batch sweep 1/4/8/16/32/64
# 单 shape：--m 4096 / --m 32 ；输出 CSV：glm5_ops_{prefill,decode}_reward.csv
```
