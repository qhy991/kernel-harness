# Kimi / MiniMax 算子性能测试汇总

日期: 2026-07-10  
机器: NVIDIA B200  
环境: `/home/qinhaiyan/kernel-harness/activate_env.sh` (glm52 venv + SGLANG_DIR)

说明:
- 参考表 `sglang_b200_operator_backend_inventory.xlsx` 实际是 GLM-5.2 / DeepSeek-V4-Pro 清单；
  Kimi K2.x 在 harness 中映射为 DSA ops 1–27（与 DeepSeek V3.2 同族）。
- MiniMax M3（ops 28–43）在本机 `/home/qinhaiyan/sglang` 中缺少 `minimax_sparse_ops` / `bench_minimax_*.py`，无法实测。
- 下列数字来自 `scripts/bench_kimi_real_kernels.py` 真实 kernel 微基准（DeepGEMM / sgl_kernel / torch.bmm）。
- decode 小 M 场景受 GPU 上其他进程与 launch overhead 影响，约有 ~50µs 地板噪声。

## Kimi DSA 真实 kernel 结果 (25 OK / 0 FAIL / 0 SKIP)

| op | 名称 | phase | latency (µs) | TFLOPS | backend |
|---:|------|--------|-------------:|-------:|---------|
| 1 | Q_a (fused w/ KV_a) | prefill | 444.30 | 1218.01 | deep_gemm w8a8_block_fp8 |
| 2 | Q_b | prefill | 151.80 | 905.37 | deep_gemm w8a8_block_fp8 |
| 4 | KV_b | prefill | 149.03 | 403.46 | deep_gemm w8a8_block_fp8 |
| 27 | FlashAttn causal MHA (prefill) | prefill | 7952.07 | 138.27 | flashinfer.single_prefill_with_kv_cache |
| 5 | O_proj | prefill | 354.61 | 1162.75 | deep_gemm w8a8_block_fp8 |
| 6 | Index_Q | prefill | 693.66 | 1188.82 | deep_gemm w8a8_block_fp8 |
| 7 | Index_K | prefill | 66.54 | 387.29 | deep_gemm w8a8_block_fp8 |
| 8 | Index_Score (ragged prefill) | prefill | 2668.81 | 823.97 | deep_gemm.fp8_mqa_logits |
| 9 | Dense GateUp | prefill | 996.20 | 1241.67 | deep_gemm w8a8_block_fp8 |
| 10 | Dense Down | prefill | 327.85 | 943.22 | deep_gemm w8a8_block_fp8 |
| 11 | MoE Router | prefill | 72.48 | 711.05 | deep_gemm w8a8_block_fp8 |
| 12 | MoE GateUp GroupGEMM | prefill | 902.02 | 914.20 | deep_gemm grouped |
| 13 | MoE Down GroupGEMM | prefill | 1546.81 | 266.56 | deep_gemm grouped |
| 14 | Q_a (fused w/ KV_a) | decode | 55.54 | 9.51 | deep_gemm w8a8_block_fp8 |
| 15 | Q_b | decode | 53.70 | 19.99 | deep_gemm w8a8_block_fp8 |
| 17 | q_nope absorb BMM | decode | 50.23 | 4.01 | torch.bmm bf16 |
| 18 | v absorb BMM | decode | 50.77 | 5.29 | torch.bmm bf16 |
| 19 | O_proj | decode | 67.89 | 47.45 | deep_gemm w8a8_block_fp8 |
| 20 | Index_Q | decode | 55.57 | 14.49 | deep_gemm w8a8_block_fp8 |
| 21 | Index_K | decode | 54.88 | 0.46 | deep_gemm w8a8_block_fp8 |
| 22 | Index_Score (paged decode) | decode | 125.21 | 2.14 | deep_gemm.fp8_mqa_logits |
| 23 | MoE Router | decode | 55.22 | 0.91 | deep_gemm w8a8_block_fp8 |
| 24 | MoE GateUp GroupGEMM | decode | 147.48 | 43.69 | deep_gemm grouped |
| 25 | MoE Down GroupGEMM | decode | 140.92 | 22.86 | deep_gemm grouped |
| 26 | Flash Decoding MLA sparse MQA | decode | 157.57 | - | sgl_kernel.flash_mla_sparse_fwd |

说明：B200 上 `sgl_kernel` FA3 不支持 sm100；op27 用 FlashInfer（与 DSA SM100+ 生产路径一致）。Dao `flash-attn` 正在后台源码编译安装（`/tmp/flash_attn_install.log`）。

## MiniMax M3
本机 sglang checkout **无** MiniMax JIT / sparse_ops / `bench_minimax_*.py` / `test_minimax_*.py`，ops 28–43 全部无法跑。需要带 MiniMax 源码的分支或外部包后再测。

## 产物路径
- 汇总: `logs/kimi_minimax_20260710-070816/`
- 真实 kernel JSON: `kimi_real_kernel.json`
- baseline: `baseline_results.{json,csv}`
- 脚本: `scripts/bench_kimi_real_kernels.py`, `scripts/run_kimi_minimax_perf.sh`
