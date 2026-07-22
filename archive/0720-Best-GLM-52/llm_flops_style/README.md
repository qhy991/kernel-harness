# llm_flops-style layer benches (CUDA Graph DROP-IN)

Decode / prefill 层延时，与 `llm_flops/bench_glm5_{decode,prefill}.py` **严格对齐**：

1. 计时：CUDA Graph（warmup=5, runs=20）
2. 未替换算子：直接调 llm_flops `bench_*`
3. 替换算子：**用 llm_flops 同一套量化构造冻结 tensor**，再在该输入上分别测 stock 与 candidate

| 脚本 | 作用 |
|------|------|
| `bench_decode.py` | Decode drop-in |
| `bench_prefill.py` | Prefill drop-in |
| `_common.py` | swap 表 + 同输入 builder |

## Decode swaps

- `q_b_proj` ← `best/q_b_decode`
- `o_proj` ← `best/o_proj_decode_hbm35`
- `index_k_proj` ← `best/index_k_proj_decode`
- `fused_qkv_a` / `index_q_upproj` / `dsa` ← PR5 `best-hechenxi-0720/`
- `moe_gate/up/down` ← `best/moe_*_decode_hbm40`

## Prefill swaps

- PR#3 扁平 `.py`：`fused_qkv_a`、`q_b`、`index_q`、`index_weights`、`dsa`、`index_score`、…
- **与 decode 共用** `best/`：`o_proj_decode_hbm35`、`index_k_proj_decode`、`moe_up/down_decode_hbm40`
- `moe_gate` 仍 stock（pack 在 M4096 Graph 可能回退）

## 运行

```bash
cd /home/qinhaiyan/Kernel-Harness/archive/0720-Best-GLM-52/llm_flops_style
CUDA_VISIBLE_DEVICES=0 ../../../.venv/bin/python bench_decode.py
CUDA_VISIBLE_DEVICES=0 ../../../.venv/bin/python bench_prefill.py
```

结果见 [`COMPARISON_TABLE.md`](COMPARISON_TABLE.md)（关键算子延时表）与 [`REPORT.md`](REPORT.md)。  
协作接入：[`COLLAB.md`](COLLAB.md)。
