# GLM-5.2 E2E Prefill Profile vs Kernel-Harness（2026-07-22）

> **共享给同伴**：本次在 MI300X 上跑通的 **TP=8 真实 e2e prefill torch-profile**，
> 以及与 `kernel-harness` AMD campaign / baseline 的对照。  
> **结论先行：算子清单有重叠，但优先级与 baseline 含义差得大——尤其 AllReduce、MLA 路径、FP8 GEMM 基线。**

| 项 | 值 |
|----|----|
| Run ID | `20260722_172150` |
| 硬件 | 8× MI300X (gfx942) |
| 模型 | `/mnt/public/qinhaiyan/models/GLM-5.2-FP8` |
| 配置 | TP=8 · `--dsa-prefill/decode-backend aiter` · `--moe-runner-backend triton` · KV=`bfloat16` · `max_total_tokens=65536` |
| 形状 | `input_len ∈ {1024,2048,4096}` · `output_len=1` · `profile-stage=prefill` |
| Wall TTFT | 1024→393.8 ms · 2048→480.3 ms · 4096→**738.2 ms** |
| Chrome traces | `/mnt/public/qinhaiyan/glm52_profile_traces/glm52_prefill_20260722_172150_*_prefill.trace.json.gz` |
| 本目录 CSV/JSON | 见下方文件表 |

口径：GPU 时间 = torch profiler `cat=kernel` 求和（单 rank timeline），排除 memcpy/memset/runtime。占比是 **GPU kernel 时间占比**，不是 wall clock。

---

## 1. E2E Top 算子（主形状 M=4096）

| 排名 | 算子族 | GPU ms | 占比 | Profile kernel 名 |
|------|--------|--------|------|------------------|
| 1 | **AllReduce (TP)** | 184.6 | **34.9%** | `aiter::cross_device_reduce_2stage_*` |
| 2 | **MoE fused GEMM** | 82.8 | **15.7%** | `fused_moe_kernel` (triton) |
| 3 | ATen elementwise/misc | 76.6 | 14.5% | 多种 `at::native::elementwise_*` |
| 4 | **MLA attention (dense)** | 72.8 | **13.8%** | `aiter::mla_dec_stage1_bf16_*` |
| 5 | **FP8 GEMM (linear)** | 57.0 | **10.8%** | `fp8gemm_*BpreShuffle` + CK bpreshuffle |
| 6 | Quant / act | 18.5 | 3.5% | `per_token_group_quant_8bit` 等 |

短序列时 AllReduce 占比更高（M=1024/2048 约 **50–54%** GPU 时间）。完整表见 `e2e_prefill_op_share.csv`。

**工况提醒**：这是 **空 KV 上的短 prefill**（dense aiter MLA），**不是** S=65536 sparse extend。因此 `flash_mla_sparse` / `unified_attention_sparse_mla` **不会**出现在 top 列表。

---

## 2. 与 Kernel-Harness 选核的差别（重点）

### 2.1 Harness 当前怎么选

| 来源 | 内容 |
|------|------|
| Campaign 主攻（层占比>5%） | `o_proj` · `index_k` · `dsa_attn`（decode）— 见 `../amd-glm-object.csv` |
| ROCm taskset | 另含 MoE total / index_score / dsa_prefill_attn 等 — `../../../tasksets/glm52_rocm_local.json` |
| 部署假设 | **`MI300X-DP1-TP1-EP32`（TP=1）** — 无 TP AllReduce |
| Attention 基线语义 | 常按 **S=65536 sparse MLA**（`pa_decode_sparse` / flash sparse） |
| FP8 GEMM 基线 | 文档多写 **Triton / `gemm_a8w8_blockscale`**；注明 ASM bpreshuffle 有时不可编 |

### 2.2 Baseline / 范围不对齐（请同伴必读）

| # | 问题 | E2E 实测 | Harness 现状 | 影响 |
|---|------|----------|--------------|------|
| **B1** | **AllReduce 缺失** | TP=8 下 **最大瓶颈 ~35%** | **完全 OUT of scope**（TP1） | 单算子 campaign **解释不了 e2e TTFT** |
| **B2** | **MLA 路径名同实异** | 短 prefill → **`mla_dec_stage1` dense** | `dsa_attn` 基线偏 **sparse @ S=64k** | 优化 sparse decode **不直接**覆盖本次 e2e prefill |
| **B3** | **FP8 GEMM 基线偏软** | E2E 已走 **ASM/CK bpreshuffle**（`fp8_fast_path`） | AITER baseline CSV 多按 **Triton/CK plain** | harness「还能再快 2×」对当前生产路径要 **打折** |
| **B4** | **MoE 优先级偏低** | `fused_moe_kernel` **#2（~16%）** | 在 taskset，但 **不在 3 个 campaign 主攻** | e2e 收益可能被低估 |
| **B5** | **o_proj / index_k 单算子占比** | 淹没在「FP8 GEMM 合计 ~11%」里 | Campaign 主攻 | 单算子仍有价值，但 **不是 e2e 第一瓶颈** |
| **B6** | **KV / 上下文假设** | pool=64k，但实际 seq=input_len | 微基准固定 **S=65536 attend** | indexer / attn 成本量级不同 |

详细对照表：`e2e_vs_harness_baseline_mismatch.csv`。

---

## 3. 优化相关代码定位（E2E 真实路径）

### 3.1 AllReduce — `cross_device_reduce_2stage`

```
sglang/.../layers/communicator.py
  → moe_tensor_model_parallel_all_reduce / attention_*_all_reduce
sglang/.../distributed/parallel_state.py::_all_reduce_out_place
sglang/.../device_communicators/custom_all_reduce.py
  → SGLANG_USE_AITER_AR → aiter CustomAllreduce
aiter/dist/device_communicators/custom_all_reduce.py::all_reduce
  → ops.all_reduce → cross_device_reduce_2stage_*
```

### 3.2 MoE — `fused_moe_kernel`

```
sglang/.../moe/moe_runner/triton.py::TritonRunnerCore.run
  → triton_utils/fused_moe.py::_fused_moe_kernel_sequence
  → fused_moe_triton_kernels.py::invoke_fused_moe_kernel
  → @triton.jit fused_moe_kernel   # profile 名即此
```

Harness 对应：`moe_total` / `moe_gate|up|down`（`testbench/tasks/glm52/moe_*`）。

### 3.3 MLA dense — `mla_dec_stage1_bf16_*`

```
sglang/.../attention/dsa_backend.py
  · forward_extend → _forward_aiter_extend   # 本次 prefill
  · _run_aiter_mla_decode_fwd                # decode
  → aiter.mla.mla_decode_fwd
aiter/mla.py::mla_decode_fwd
  → aiter.mla_decode_stage1_asm_fwd
aiter/csrc/py_itfs_cu/asm_mla.cu
```

Harness 名：`dsa_attn` / `dsa_prefill_attn` — **请标注 baseline 是 dense stage1 还是 sparse**。

### 3.4 FP8 GEMM — ASM/CK bpreshuffle

```
sglang/.../quantization/fp8_utils.py::aiter_w8a8_block_fp8_linear
  · gfx942 + 大 M → gemm_a8w8_blockscale_bpreshuffle_asm
  · 否则 → gemm_a8w8_blockscale_bpreshuffle_ck / plain CK
aiter/ops/gemm_op_a8w8.py
E2E 本次额外：/root/glm5-flops-amd/fp8_fast_path.py  (monkey-patch，默认开)
```

Harness 对应：`o_proj` / `index_k` / `fused_qkv_a` / `q_b` 等；**应用 ASM 路径重测 baseline**，勿继续只用 Triton 分母宣称 speedup。

完整 API 表：`e2e_kernel_api_map.csv`。

---

## 4. 建议同伴怎么用

1. **看 e2e TTFT / 通信**：先承认 TP=8 AllReduce；harness TP1 优化不能单独闭环 e2e。  
2. **改 harness baseline 文档**：在 `amd-glm-object.csv` / `operator_mapping.md` 旁注「生产路径已 ASM」与「e2e prefill=dense MLA」。  
3. **Campaign 补强（可选）**：  
   - 增加 **MoE fused** 优先级（已有 task，抬到主攻）  
   - 增加 **dense `mla_dec_stage1` prefill** 工况（与 sparse decode 分开）  
   - 单独跟踪 **AR**（或 TP 策略），即使不做 kernel contest  
4. **复现 e2e**：脚本与参数见 `REPRODUCE.md`；traces 在 NFS（本仓不入库大文件）。

---

## 5. 本目录文件

| 文件 | 说明 |
|------|------|
| `README.md` | 本说明（共享入口） |
| `REPRODUCE.md` | 复现 e2e profile 命令 |
| `e2e_prefill_op_share.csv` | 三 shape 算子族 GPU ms / 占比 |
| `e2e_prefill_top_kernels.csv` | M=4096 top kernels |
| `e2e_vs_harness_baseline_mismatch.csv` | baseline/范围不对齐台账 |
| `e2e_kernel_api_map.csv` | profile kernel → 代码 API → harness op |
| `op_share.json` | 机器可读完整份额 |
| `profile_result_20260722_172150.jsonl` | bench_one_batch 墙钟结果 |
| `analyze_chrome_trace.py` | 从 chrome trace 重算份额（可选） |

维护人备注：2026-07-22 · MI300X e2e · 对照 `rewardbench/amd` campaign。
