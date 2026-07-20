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

- `q_b_proj` ← `q_b_decode`
- `o_proj` ← `o_proj_decode_hbm35`
- `index_q_upproj` ← `index_q_upproj_decode_hbm15`
- `moe_gate/up/down` ← 对应 decode_hbm40

## Prefill swaps

- `index_k_proj` ← `index_k_prefill_bw70`（rows=S）
- `o_proj` ← `o_proj_prefill`
- `moe_down_proj` ← `moe_down_proj_prefill_mfu65`

## 运行

```bash
cd /home/qinhaiyan/Kernel-Harness/archive/0720-Best-GLM-52/llm_flops_style
CUDA_VISIBLE_DEVICES=0 ../../../.venv/bin/python bench_decode.py
CUDA_VISIBLE_DEVICES=0 ../../../.venv/bin/python bench_prefill.py
```

结果见 [`COMPARISON_TABLE.md`](COMPARISON_TABLE.md)（关键算子延时表）与 [`REPORT.md`](REPORT.md)。  
协作接入：[`COLLAB.md`](COLLAB.md)。
