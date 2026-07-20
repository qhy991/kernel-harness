# GLM-5 算子性能测试 Reward 指标设计

配套脚本：`bench_GLM5_ops_prefill.py` / `bench_GLM5_ops_decode.py`（共享 `glm5_ops_common.py`）。
13 个算子及其后端 kernel/dtype **与 `llm_flops/bench_glm5_prefill.py`、`bench_glm5_decode.py` 完全对齐**
（该 repo 是 H800+B200 双部署、特调过参数的基准参考）。少数在 sglang 里按 dtype/硬件分支的算子，
默认用 llm_flops 的选择，另按分支提供可切换的备选 baseline（见文末「dtype/硬件分支」）。

## Reward 总体设计（依据 SGLang 官方实践建议 6 条 Rule，尤其 Rule 3「按 bound 读」）

**reward = bound-aware roofline 利用率 ∈ [0,1] = 实测吞吐 ÷ roofline 天花板**
- AI = FLOPs / HBM字节；ridge = 峰值算力 ÷ 8TB/s；AI≥ridge→compute-bound，否则 memory-bound（自动判定）。
- roofline 天花板 = min(峰值算力, AI×8TB/s)；reward = 实测算力 ÷ 天花板。
  - compute-bound ⇒ reward ≡ Tensor-Core 利用率；memory-bound ⇒ reward ≡ HBM 带宽利用率。
- 同一算子 prefill(大 M) 多 compute-bound、decode(小 batch) 多 memory-bound，reward 自动切到「卡脖子」资源，可跨 shape/阶段统一比较；最大化 reward 即逼近 roofline。低 reward(如 index_k decode≈0.005) 直接标注最大 headroom。
- 峰值(B200/SM100)：HBM 8 TB/s；FP8 TC 4.5 PFLOP/s；BF16 TC 2.25 PFLOP/s。ridge：FP8=562.5、BF16=281.25 FLOP/byte。
- 计时：baseline 模式用 CUDA Graph capture+replay（5 warmup + 20 replays 均值），与 llm_flops 一致（Rule 2「先固定 benchmark」）；candidate 模式改用 **CUDA-event 计时**（对任意第三方 kernel 稳健，见文末）。
- ABI 对齐（Rule 5）：每算子调用与 llm_flops/sglang 相同的后端 kernel/dtype/scale/layout。FP8 blockwise 在本机 deep_gemm(B200) 强制 UE8M0 scale（`DEEPGEMM_SCALE_UE8M0=True`；kernel/dtype 与 llm_flops 相同，仅 scale 格式随硬件——llm_flops 的 fp32 scale 在本机 deep_gemm 上会 assert）。
- 每算子输出诊断：latency(ms)、TFLOP/s、GB/s、AI、bound、util%、reward。

## 指标表（backend 与 llm_flops 一致；util% 为代表性 shape 实测：prefill M=4096 / decode batch=32，S=65536）

| 算子 | 类型（prefill/decode） | 反馈指标设计 | 设计原因 |
|---|---|---|---|
| fused_qkv_a_proj | prefill | FP8 compute 利用率（compute-bound, AI≈2077, ~53%）| deep_gemm.fp8_gemm_nt [M,6144]×[6144,2624]；大 M 计算密集，卡 tensor-core。 |
| q_b_proj | prefill | FP8 compute 利用率（compute-bound, AI≈1558, ~53%）| deep_gemm.fp8_gemm_nt [M,2048]×[2048,16384]，大 N，compute-bound。 |
| absorbed_W_UK | prefill | FP8 HBM 带宽利用率（memory-bound, AI≈159, ~61%）| sgl_kernel.bmm_fp8 [64,M,192]×[64,192,512] per-tensor FP8；K=192 小，带宽 bound。 |
| absorbed_W_UV | prefill | FP8 HBM 带宽利用率（memory-bound, AI≈248, ~77%）| sgl_kernel.bmm_fp8 [64,M,512]×[64,512,256]，搬运主导，带宽 bound。 |
| o_proj | prefill | FP8 compute 利用率（compute-bound, AI≈3745, ~61%）| deep_gemm.fp8_gemm_nt [M,16384]×[16384,6144]，K/N 大，强 compute-bound。 |
| dsa_prefill_attn | prefill | BF16 compute 利用率（compute-bound, AI≈1719, ~41%）| sgl_kernel.flash_mla_sparse_fwd（bf16 稀疏注意力），每 query gather topk=2048；FLOPs 大且用 tensor-core，compute-bound。 |
| index_k_proj | prefill | FP8 HBM 带宽利用率（memory-bound, AI≈238, ~61%）| deep_gemm.fp8_gemm_nt [S,6144]×[6144,128]，N=128 小但读 S×6144 大激活→带宽 bound。 |
| index_q_upproj | prefill | FP8 compute 利用率（compute-bound, AI≈1358, ~40%）| deep_gemm.fp8_gemm_nt [M,2048]×[2048,4096]，M=4096 compute-bound；利用率偏低=有空间。 |
| index_weights_proj | prefill | BF16→F32 HBM 带宽利用率（memory-bound, AI≈31, ~30%）| deep_gemm.bf16_gemm_nt [M,6144]×[6144,32] BF16→F32，N=32 极小→带宽 bound。 |
| index_score | prefill | FP8 compute 利用率（compute-bound, AI≈2000, ~44%）| deep_gemm.fp8_mqa_logits，q[M,32,128]·k[S,128]→logits[M,S]，UE8M0；M 大 compute-bound。 |
| moe_gate_proj | prefill | FP8 compute 利用率（compute-bound, AI≈1864, ~57%）| deep_gemm.fp8_m_grouped_gemm_nt_masked [8×,6144]×[6144,2048]，UE8M0；prefill token 多，compute-bound。 |
| moe_up_proj | prefill | FP8 compute 利用率（compute-bound, AI≈1864, ~57%）| 同 gate（运行时与 gate 融合为单个 w13 N=4096 GEMM，此处按 per-op 粒度各计一半）。 |
| moe_down_proj | prefill | FP8 compute 利用率（compute-bound, AI≈1440, ~50%）| deep_gemm.fp8_m_grouped_gemm_nt_masked [8×,2048]×[2048,6144]，compute-bound。 |
| fused_qkv_a_proj | decode | HBM 带宽利用率（memory-bound, AI≈62, ~9%）| skinny-M(batch=32)，读 6144×2624 权重主导→带宽 bound；利用率低=权重流式带宽是瓶颈(大 headroom)。 |
| q_b_proj | decode | HBM 带宽利用率（memory-bound, AI≈62, ~18%）| skinny-M，读 2048×16384 权重主导。 |
| absorbed_W_UK | decode | FP8 HBM 带宽利用率（memory-bound, AI≈46, ~36%）| sgl_kernel.bmm_fp8，batch=64 但每 batch M=32，带宽 bound。 |
| absorbed_W_UV | decode | FP8 HBM 带宽利用率（memory-bound, AI≈51, ~44%）| sgl_kernel.bmm_fp8，带宽 bound。 |
| o_proj | decode | HBM 带宽利用率（memory-bound, AI≈63, ~28%）| skinny-M，读 16384×6144 权重主导（与 prefill 的 compute-bound 相反，reward 自动切换）。 |
| dsa_decode_attn | decode | BF16 HBM 带宽利用率（memory-bound, AI≈114, ~26%）| sgl_kernel.flash_mla_sparse_fwd，decode s_q=batch 很小，gather KV 带宽/延迟主导，memory-bound。 |
| index_k_proj | decode | FP8 HBM 带宽利用率（memory-bound, AI≈50, ~0.6% ⇒ launch-bound）| deep_gemm.fp8_gemm_nt [32,6144]×[6144,128]，计算量极小，被启动开销主导；reward≈0 标注最大 headroom，杠杆是融合/降启动。 |
| index_q_upproj | decode | HBM 带宽利用率（memory-bound, AI≈62, ~6%）| skinny-M，启动/带宽 bound（已验证自定义 split-K Triton kernel 可提升~2.5x）。 |
| index_weights_proj | decode | BF16→F32 HBM 带宽利用率（memory-bound, AI≈16, ~0.8% ⇒ launch-bound）| deep_gemm.bf16_gemm_nt [32,6144]×[6144,32]，N=32 计算量极小，launch-bound；杠杆是融合。 |
| index_score | decode | FP8 HBM 带宽利用率（memory-bound, AI≈60, ~73%）| deep_gemm.fp8_paged_mqa_logits，读分页 fp8 KV(M×S×132B) 主导→强 memory-bound；73% 已近 roofline。 |
| moe_gate_proj | decode | HBM 带宽利用率（memory-bound, AI≈62, ~33%）| grouped masked，decode 每专家 token 少→读 8×专家权重(~100MB) 主导，带宽 bound。 |
| moe_up_proj | decode | HBM 带宽利用率（memory-bound, AI≈62, ~33%）| 同 gate（运行时融合为 w13）。 |
| moe_down_proj | decode | HBM 带宽利用率（memory-bound, AI≈62, ~33%）| grouped masked，权重流式带宽 bound（deep_gemm 仅~33% HBM，有 headroom；已验证 warp-specialized Triton 可提升~1.14x）。 |

## 特殊算子的 dtype/硬件分支（sglang 源码已核对；默认对齐 llm_flops，备选可切换）

sglang 对这几个算子按 KV dtype / 硬件(H800 SM90 vs B200 SM100) / indexer 融合配置走不同 kernel。
脚本**默认（active）= llm_flops 选择**（H800+B200 特调部署参考）；每个算子下方以注释给出**按 dtype 分支的备选
baseline**，两套 builder 均在 `glm5_ops_common.py`，取消注释即可切换测该分支。

| 算子 | 默认 baseline（active，= llm_flops） | 备选分支（脚本内注释，按 dtype/HW） | sglang 分支依据 |
|---|---|---|---|
| absorbed_W_UK / absorbed_W_UV | `sgl_kernel.bmm_fp8`（per-tensor FP8）| `torch.bmm`（bf16）| B200/SM100 `DEEPGEMM_BLACKWELL=True` → kv_b_proj 反量化 bf16 → `torch.bmm`（`deepseek_weight_loader.py`、`forward_mla.py`）；fp8-bmm 仅当 H800/SM90 或 `SGL_USE_DEEPGEMM_BMM=1` |
| index_k_proj (wk) | `deep_gemm.fp8_gemm_nt`（fp8）| `torch.F.linear`（bf16）| 非融合(`is_neox_style=True`)走独立 fp8；GLM-5.2 `indexer_rope_interleave=True`→`use_dsa_indexer_fusion=True`→与 weights 融合为 bf16 `wk_weights_proj[6144,160]`（`dsa_indexer.py:380/410`）|
| index_weights_proj | `deep_gemm.bf16_gemm_nt`（bf16→f32）| `torch.mm`（bf16→f32）| 非融合独立 bf16→f32；融合路径并入 `wk_weights_proj`（`dsa_indexer.py:471-482` `torch.mm(...,out_dtype=float32)`）|
| dsa_prefill_attn / dsa_decode_attn | `sgl_kernel.flash_mla_sparse_fwd`（bf16）| `flashinfer trtllm-gen`（fp8 KV, `sparse_mla_top_k`）| `--kv-cache-dtype bfloat16`→flash_mla_sparse；B200 默认 `kv_cache_dtype` auto→fp8_e4m3→DSA backend `trtllm`（`dsa_backend.py:2716`, `overrides.py:1141`）|

其它已核对为**与 llm_flops/sglang 对齐**：fused_qkv_a / q_b / o_proj（deep_gemm.fp8_gemm_nt UE8M0）；index_q_upproj（fp8_gemm_nt）；index_score（prefill fp8_mqa_logits / decode fp8_paged_mqa_logits）；moe gate/up/down（fp8_m_grouped_gemm_nt_masked）。

已修正的正确性问题（不改变 kernel/dtype，仅保证正确）：
- **moe grouped buffer 越界**：随机 multinomial 最大 per-expert bin 可能超过 `expected_m`（如 M=4096 时 4179>4096）导致 masked kernel 越界。已改为按 `max(ceil(total_m/E), max(counts))` 分配容量（无越界时与 llm_flops 一致）。
- **moe gate+up 融合**：运行时 sglang 融合为单个 w13 grouped GEMM(N=4096)；本脚本按 13 算子粒度分别计 gate/up（各为其一半，FLOP 等价），表中已注明。

## 运行

两个脚本各有**两种模式**（同一份 `glm5_ops_common.py`）：

```bash
cd glm5_ops_reward
# ── 模式 1：baseline（默认）——测 13 个 sglang/llm_flops 参考算子，作为 reward 分母 ──
../kernel-harness/.venv/bin/python bench_GLM5_ops_prefill.py       # M sweep 1024/2048/4096
../kernel-harness/.venv/bin/python bench_GLM5_ops_decode.py        # batch sweep 1/4/8/16/32/64
# 单 shape：--m 4096 / --m 32；输出 CSV：glm5_ops_{prefill,decode}_reward.csv
# 切换某算子到 dtype 备选分支：在脚本 build_ops() 里注释掉 active 行、启用其下方 alt 注释行

# ── 模式 2：candidate（--kernels-dir）——输入「给定的优化算子文件夹」，输出一个大 CSV ──
../kernel-harness/.venv/bin/python bench_GLM5_ops_prefill.py --kernels-dir best-kernels-reward-bench
../kernel-harness/.venv/bin/python bench_GLM5_ops_decode.py  --kernels-dir best-kernels-reward-bench
# --kernels-dir 也可直接指向【单个算子目录】（其下直接有 solution.py），供 flow 里单算子调用：
../kernel-harness/.venv/bin/python bench_GLM5_ops_prefill.py --kernels-dir best-kernels-reward-bench/dsa_prefill_attn
# prefill 脚本只测 phase==prefill 的候选，decode 脚本只测 phase==decode 的候选
# 每个候选的结果：① 终端逐行打印（带 UTC 时间戳）② 追加进【该算子目录下的 reward_bench.csv】（带时间戳的历史）
#                 ③ 汇总进大 CSV：glm5_ops_{prefill,decode}_candidates.csv
# 选项：--repeat N（取最快）、--no-baseline（只出 reward，不算 speedup）、--round K、--csv 自定义汇总文件
```

## Candidate 模式 —— 输入「优化算子文件夹」，两个脚本各自输出一个大 CSV（本次接口）

**接口要求**：测试对象是「给定的优化算子」——一个装有很多算子实现的文件夹（**不是**从 sglang 里抽取），
prefill 脚本与 decode 脚本各自对属于本阶段的候选直接测分、汇总成一个大 CSV。
**本模块只做性能评测**：正确性测试是它之前的独立环节（我们不实现，也不在此写占位/门禁）。

**输入结构**（参照 `best-kernels-reward-bench.zip` 解压树）：`--kernels-dir` 可指向父目录（下含多个候选子目录），
**也可直接指向单个算子目录**（其下直接有 `solution.py`）——flow 里通常一次只测一个算子，用后者。
```
<kernels_dir>/<candidate>/solution.py   # 必需：def run(...)；可选 def get_inputs(axes,device)
                          /task.json     # op/phase/family/sweep/K/N/E（可为空，缺省按内置推断）
                          /META.md       # "reward operator: <op>" 归属映射
                          /reward_bench.csv   # 【本脚本写入】该算子的带时间戳结果历史（追加）
```
算子仍是**每阶段一类 13 个**；提供的样例文件夹不全（只覆盖其中一部分），未提供候选的算子在
candidate 模式下自然缺席，其 roofline 参考值见 baseline 模式的 `*_reward.csv`。

**流程（每个候选 × 每个 shape）**：
1. 阶段过滤：`task.json.phase` 或文件夹名含 `decode` → decode，否则 prefill；只测与本脚本阶段一致的候选（单算子模式下阶段不符会明确提示改用另一个脚本）。
2. 归属算子：`META.md` 的 `reward operator:` > `task.json.llm_flops_op/op` > 文件夹名推断。
3. 取 `get_inputs`：**优先候选自带的**（它编码了该 kernel 期望的精确 scale/layout，如 o_proj 的 packed-int32 UE8M0）→ 同 family 的兄弟候选 → 本 bench 内置的 canonical。
4. 构造输入 → 以**按参数名绑定**的方式调用候选 `run(*inputs[params])`（兼容各 family 不同 run 签名：fp8-linear 4 参 / grouped-moe 9 参含 m_indices,layout / dsa-prefill 3 参 / sparse-mla-decode 7 参）。
5. baseline（可选，供 speedup）：按 family 调用 live 参考 kernel（deep_gemm.fp8_gemm_nt / fp8_m_grouped_gemm_nt_masked / bf16_gemm_nt / flash_mla_sparse_fwd / trtllm_batch_decode_with_kv_cache_mla），得 `base_us`；`--no-baseline` 可跳过。
6. 计时：**CUDA-event 计时（非 graph capture）**——对任意候选 kernel 稳健（Triton autotune / host-side routing / .item() / 动态 launch 都不会像 graph capture 那样污染上下文）；含真实 launch/dispatch 开销，贴近服务路径。
7. reward = bound-aware roofline 利用率（从候选实测延迟 + 由**实际输入张量**推出的 FLOP/字节模型算得，**不依赖 baseline**）；speedup = base_us / sol_us（仅当算了 baseline）。

**结果去向**（每个候选）：① 终端逐行打印，行首带 **UTC 时间戳**；② **追加**写入该算子目录下的 `reward_bench.csv`（带时间戳，供 agent 多轮测试累积历史）；③ 汇总进大 CSV（`glm5_ops_{prefill,decode}_candidates.csv`，覆盖写）。

**CSV 列**（性能列 + 时间戳，无正确性列）：
`ts, round, candidate, task, operator, phase, family, M, K, N, S, sol_us, base_us, speedup, achieved_tflops, achieved_gbps, arithmetic_intensity, bound, pct_fp_peak, util_pct, reward`

已在 `best-kernels-reward-bench/`（9 个候选样例）验证跑通：prefill 脚本命中 3 个 prefill 候选（9 行），
decode 脚本命中 6 个 decode 候选（20 行），0 崩溃；数值与各候选自带 `benchmark.csv` 吻合（如 o_proj_prefill
M=4096 sol≈844µs vs 其记录 823µs），speedup 也与各 `META.md` 的 WIN/no-win 结论一致（moe_gate/moe_down/routed_down >1；o_proj_prefill ~0.33×）。
