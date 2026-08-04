# MI300X baseline 与 PR 测试方法差异分析（2026-08-04）

本文记录 GLM-5.2 MI300X 算子结果无法完整复现、以及远程 PR 测试看起来比历史测试更慢的原因。目标是让维护 `kernel-harness`、SGLang ROCm 和 AITER 的同事可以直接复核问题，不再把不同 shape、不同 dispatch 或不同计时边界的数据放在同一张表中比较。

## 1. 结论摘要

1. **PR #17 没有修改测试方法。** 它只修改 `testbench/tasks/glm52_amd/moe_total_decode/candidate.py`；`run.sh`、`evaluate_task.py` 和 ROCm timer 与 main 相同。
2. **当前正式 timer 本身不比旧 rewardbench graph timer 慢。** 对同一 callable 的实测差异在约 0.5%～2% 内，当前 timer 还略快。
3. **MoE baseline 变慢的主因是 alignment 实现退化。** 当前 `sgl_kernel 0.4.3` 没有注册 `torch.ops.sgl_kernel.moe_align_block_size`，SGLang 回退到含逐 token Python 循环和大量 `.item()` 的 PyTorch 实现。它既无法 HIP graph capture，也制造大量 device 同步。
4. **当前几个 baseline 的实际 dispatch 与历史表不同。** 历史 O Projection / Index K 表的分母是直接调用 AITER Triton fallback；当前正式 harness 先进入 SGLang production dispatch，可能选择 CK 或 bpreshuffle ASM。
5. **Index K 默认 baseline 当前无效。** 普通权重被送入要求预重排权重的 bpreshuffle ASM，虽然延时只有约 89～93 us，但输出与数学 oracle 完全不一致，不能作为分母。
6. **旧表只有 MLA candidate 延时可精确复现。** `1192 us` 复测为 `1191.328 us`；其他项受到 backend、API、输入契约或调用边界变化影响。

## 2. 复测环境

| 组件 | 版本 / 提交 |
|---|---|
| 机器 | MI300X 节点 `is-dc6nbl3mwwjj3mfs-devmachine-0` |
| kernel-harness main | `050571e5ff15b097b86eb3c751bd99ab54a1e809` |
| PR #17 head | `441cfb28eb85c40a3fc01849dbc987623b5f813e` |
| AITER source | `754e82edf9d5048f21a9332935e9db2a07d6d0d1` |
| SGLang source | `20fc529abdb9c9da36dbf7f2789fa285be3663f4` |
| installed SGLang | `0.5.13.post1` |
| installed sgl_kernel | `0.4.3` |
| PyTorch / HIP | `2.10.0+rocm7.0` / `7.0.51831` |
| Triton | `3.6.0` |
| 正式协议 | HIP graph 或 event fallback，`warmup=3`、`repeat=10`、`iterations=30` |

测试前 GPU 使用率和 VRAM 占用均为 0。正式隔离复测固定 GPU 0；MoE 首轮正式结果固定 GPU 1，随后在 GPU 0 上用独立 HIP Event 和 GPU profile 复核。

## 3. 历史结果与当前正式结果

延时单位均为 us。Index K 的 default baseline 已确认错误，因此另列禁用 bpreshuffle 后的有效结果。

| 算子 / shape | 历史 baseline | 历史 KDA | 当前有效 baseline | 当前 candidate | 当前结论 |
|---|---:|---:|---:|---:|---|
| MLA Prefill Attn, M=1024 | 1667 | 1192 | 1769.483 | 1191.328 | candidate 复现；baseline 慢 6.15%；当前 1.4853x |
| Routed Expert Total Prefill, M=1024 | 1365 | 1231 | 603995.483 | 603569.611 | 当前 alignment 路径异常，不可直接比较 |
| DSA Index Score, M=1024 | 930 | 649 | 726.442 | 724.036 | candidate 静默 fallback；neutral 1.0033x |
| Attention O Projection, M=1024 | 946 | 428 | 622.968 | 473.169 | 当前 CK baseline 有效；1.3166x |
| DSA Index K, M=1024 | 568 | 289 | 208.195 | 191.814 | 禁用错误 bpreshuffle 后 1.0854x |
| DSA Index K, M=2048 | 540 | 287 | 208.561 | 191.867 | 禁用错误 bpreshuffle 后 1.0870x |

两个历史表问题需要同时修正：

- `1667 / 1192 = 1.3985x`，不是表中的 `1.378x`。
- `540 -> 287 us` 对应 M=2048；历史 M=1024 是约 `568 -> 289 us`。

## 4. PR 测试方法是否更慢

### 4.1 PR #17 与 main 使用同一 runner

PR #17 相对 main 只有一个文件变化：

```text
testbench/tasks/glm52_amd/moe_total_decode/candidate.py
```

因此不存在独立的“PR timer”。PR 与 main 都通过任务的 `run.sh` 进入 `testbench/harness/evaluate_task.py`，再进入同一个 ROCm `RocmBenchTimer`。

### 4.2 相同 callable 的三种计时结果

为了排除计时器差异，使用完全相同的输入、candidate 和 reference，分别运行：

- 当前正式 `time_runnable`；
- 旧 `rewardbench/amd/amd_glm5_ops_common.py::graph_bench`；
- 旧 cold-L2 HIP Event timer。

| Callable | 当前正式 graph | 旧 rewardbench graph | 旧 cold-event |
|---|---:|---:|---:|
| O Projection candidate | 464.226 | 472.583 | 547.753 |
| O Projection reference | 616.891 | 621.140 | 635.370 |
| PR #17 MoE decode candidate, M=16 | 220.197 | 221.146 | 未测 |

结论：对同一个 callable，当前正式 graph timer 没有更慢。输入 clone 在 `setup()` 中完成，不在计时区间内；正确性检查、结果持久化、GPU lock 和十个 outer repeats 也不会加到单次延时中。

旧 cold-event 反而比 graph 更慢。历史结果绝对值较低，是因为它常常直接调用单个 AITER/Triton kernel，计时边界比当前完整 `run(inputs)` 更窄，而不是因为其 timer 更“宽松”。

### 4.3 当前 timer 的一个可观测性问题

ROCm timer 先尝试 HIP graph；捕获失败就退回 HIP Event。但 `result.json` 只记录统一字符串：

```text
hipgraph-or-event-median
```

它没有分别记录 candidate 和 reference 最终使用了 graph 还是 event。这使得结果表看起来像双方使用同一协议，实际上 MoE PR 中是：

- PR candidate：graph capture 成功；
- SGLang reference：graph capture 失败，自动使用 event。

这不一定不公平——event 路径中的 GPU idle/sync 也是完整调用的真实成本——但必须在结果中明确披露。

## 5. MoE baseline 变慢的根因

### 5.1 缺少已注册的 alignment custom op

当前环境状态：

```text
sgl_kernel.__version__ = 0.4.3
hasattr(torch.ops.sgl_kernel, "moe_align_block_size") = False
```

`sgl_kernel.moe.moe_align_block_size()` 先调用：

```python
torch.ops.sgl_kernel.moe_align_block_size.default(...)
```

失败后回退 `_moe_align_block_size_pytorch()`。该 fallback 对 `M * top_k` 个 routing entry 执行 Python 循环，并反复进行：

```python
e = int(topk_flat[idx].item())
offset = int(cumsum[e].item()) + int(fill_counts[e].item())
```

对于 prefill M=1024、top-k=8，这意味着至少 8192 轮 Python routing 循环，以及大量 host/device 同步。

### 5.2 为什么 graph capture 失败

reference 捕获时出现：

```text
HIP error: operation not permitted when stream is capturing
hipErrorStreamCaptureUnsupported
```

触发位置就是 `cumsum[num_experts].item()`。正式 timer 捕获异常后转向 HIP Event，因此完整 alignment 链被计入 GPU 时间线。

### 5.3 profile 证据

M=1024 的一轮 GPU profile 显示：

- 两个 `fused_moe_kernel` 合计约 `1.201 ms`；
- 约 24,600 次 device copy / scalar dense 操作；
- 8192 次 `aten::add_`；
- 完整调用约 `590～604 ms`。

历史 `1231～1365 us` 与两个 fused GEMM kernel 的量级非常接近。因此历史表大概率使用了不同/正常的 alignment 实现，或只覆盖 fused kernel sequence；不能与当前退化后的 total-call baseline 直接比较。

### 5.4 PR #17 为什么非常快

PR #17 用 AITER Triton `moe_align_block_size_triton` 替换上述 graph-unsafe PyTorch alignment，再调用 SGLang 自己的 fused MoE kernel sequence。它没有缓存 routing 或结果，正确性 `calc_diff=0`。

正式 decode gate：

| Shape | PR candidate | reference | speedup |
|---|---:|---:|---:|
| M=16 | 216.114 | 10483.096 | 48.5072x |
| M=32 | 235.663 | 20038.407 | 85.0301x |

这个巨大倍数主要反映 baseline alignment fallback 的退化，并不代表 GEMM 计算本身提升了 48～85 倍。

## 6. 其他 baseline 与候选问题

### 6.1 MLA Prefill Attention

实际 baseline 加载的是 AITER ASM：

```text
mla_dec_stage1_bf16_a16w16_subQ128_mqa128
```

但结果元数据仍写 `unified_attention_sparse_mla`。candidate `1191.328 us` 与历史 `1192 us` 几乎完全一致；baseline `1769.483 us` 相对历史 `1667 us` 慢 6.15%，属于 backend/environment 变化或机器状态差异，而不是 timer 公式差异。

### 6.2 DSA Index Score

candidate fast path调用 `_mqa_mod._gfx942_tile_fits_lds(...)`，但当前 AITER 模块中不存在该 symbol。`run()` 的宽泛 `except Exception` 吞掉 AttributeError 并回到 reference，因此：

```text
baseline 726.442 us
candidate 724.036 us
calc_diff 0
verdict neutral
```

历史 `930 -> 649 us` 优化没有在当前环境真正执行。

### 6.3 Attention O Projection

当前 SGLang gfx942 dispatch 在 M=1024 选择 AITER CK。baseline 已直接与数学 oracle 验证，`calc_diff=4.91e-9`，是有效分母。

为了与历史口径对齐，另用同一正式 harness 强制直接调用 AITER Triton fallback：

```text
current SGLang/CK baseline:       622.968 us
current direct AITER Triton:      831.266 us
historical direct AITER Triton:   946 us
current KDA candidate:            473.169 us
historical KDA candidate:         428 us
```

因此差异同时来自 SGLang dispatch 变化和 AITER/Triton 版本变化。

### 6.4 DSA Index K 默认 baseline 无效

当前 SGLang 对大行数输入启用 gfx942 bpreshuffle ASM，但 harness 提供的是普通 blockwise 权重。直接把普通权重作为 preshuffled 权重使用，得到很低延时但错误输出。

baseline 与数学 oracle 的直接比较：

```text
calc_diff = 0.999813
cosine = 0.000187
elementwise_failed = 4,172,891 / 4,194,304
```

所以默认报告中的 `88～93 us` 不能作为性能分母。使用 SGLang 支持的开关：

```bash
SGLANG_DISABLE_GFX942_BPRESHUFFLE=1 \
  testbench/tasks/glm52_amd/index_k_prefill/run.sh --M 1024
```

得到完整、正确且稳定的正式门禁：

| Shape | 有效 baseline | candidate | speedup | conservative | verdict |
|---|---:|---:|---:|---:|---|
| M=1024 | 208.195 | 191.814 | 1.0854x | 1.0824x | WIN |
| M=2048 | 208.561 | 191.867 | 1.0870x | 1.0812x | WIN |

## 7. 输入契约不一致

当前 checked-in 文件：

```text
testbench/tasks/glm52_amd/o_proj_prefill/problem.json: S=65536
testbench/tasks/glm52_amd/index_k_prefill/problem.json: S=65536
```

但当前：

```bash
run.sh --describe --json
```

以及所有正式结果都使用 `S=32768`。旧 rewardbench prefill 默认 `S=65536`。这意味着即使 M 相同，DSA/Indexer 等读取 S 的任务也可能不是同一 workload。

## 8. 建议修复顺序

### P0：先保证 baseline 正确、可解释

1. 在计时前增加 baseline-versus-math correctness 检查；不能只验证 candidate。
2. 在 Index K 的 frozen inputs 尚未提供正确 preshuffle 权重前，默认禁用 gfx942 bpreshuffle；或者同时提供 raw/preshuffled 权重和正确 scale layout。
3. 在 `result.json` 中分别记录 candidate/reference 的实际 timer：`graph` 或 `event`，并记录 graph capture fallback 原因。
4. MoE 环境检查必须验证 `torch.ops.sgl_kernel.moe_align_block_size` 是否存在；只验证 `import sgl_kernel` 不够。

### P1：消除静默 fallback 和环境漂移

1. candidate fallback 必须记录原因；至少在结果中标记 `fast_path_engaged=false`，不要吞掉 API AttributeError 后仍看起来像一次有效优化测试。
2. 固定并记录 kernel-harness、AITER、SGLang、sgl_kernel、PyTorch、HIP、Triton 的版本/提交和实际加载路径。
3. 重新生成 AMD task 的 `problem.json` / README，使 S 与运行时一致。
4. 将“kernel-only latency”和“operator-call total latency”作为两个不同指标，不要复用同一列名。

### P2：再做性能比较

公平比较至少要同时固定：

- phase（prefill/decode）；
- M、S、top-k 和 expert 数；
- frozen inputs 和权重 layout；
- callable 边界；
- baseline backend（AITER Triton / CK / ASM / SGLang wrapper）；
- 实际 timer（graph/event）；
- 软件提交和编译扩展。

## 9. 复现命令

```bash
cd /root/repos/kernel-harness

export ROCM_TORCH_PYTHON=/root/venvs/rocm-torch/bin/python

# 当前正式 M=1024 任务
testbench/tasks/glm52_amd/dsa_prefill_attn/run.sh --M 1024 --device cuda:0
testbench/tasks/glm52_amd/moe_total_prefill/run.sh --M 1024 --device cuda:0
testbench/tasks/glm52_amd/index_score_prefill/run.sh --M 1024 --device cuda:0
testbench/tasks/glm52_amd/o_proj_prefill/run.sh --M 1024 --device cuda:0

# Index K：使用 correctness-valid baseline
SGLANG_DISABLE_GFX942_BPRESHUFFLE=1 \
  testbench/tasks/glm52_amd/index_k_prefill/run.sh --M 1024 --device cuda:0
SGLANG_DISABLE_GFX942_BPRESHUFFLE=1 \
  testbench/tasks/glm52_amd/index_k_prefill/run.sh --M 2048 --device cuda:0
```

PR #17 的正式 gate 运行在独立 worktree，head 为 `441cfb28eb85c40a3fc01849dbc987623b5f813e`。

## 10. 对外沟通建议

可以用下面这段话概括：

> 这次差异不是 PR 使用了更慢的计时公式。相同 callable 在新旧 graph timer 下只差约 1%。主要原因是当前 ROCm 环境缺少 SGLang MoE alignment custom op，baseline 回退到逐 token 的 PyTorch 实现并失去 graph capture；同时部分 baseline 从直接 AITER Triton 切换到了 SGLang CK/ASM dispatch。Index K 的默认 bpreshuffle ASM 还在普通权重上产生错误输出，因此该延时不能作为分母。只有在统一 phase、shape、权重 layout、dispatch 和实际 timer 后，speedup 才能横向比较。
