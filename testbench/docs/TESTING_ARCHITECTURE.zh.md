# Kernel-Harness 测试架构（以 `dsa_prefill_attn` 为例）

本文档回答四个问题：

1. 整个测试架构长什么样？
2. 一次 `./run.sh` 里发生了什么？
3. 怎么写 / 换 candidate 做对比？
4. 具体到 `dsa_prefill_attn`，从零开始怎么跑通、怎么读结果？

面向对象：**第一次接触 kernel-harness 的合作者**。前置知识：了解 PyTorch/Triton、见过 GPU kernel benchmark、知道 speedup 是什么。

---

## 1. 架构总览

分成三层：**任务定义**（声明式）· **算子契约**（唯一真相源）· **通用 runner**（编排器）。

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: 任务定义（每个 op 一份）                                │
│  testbench/tasks/glm52_amd/<op>/                                │
│    ├── task.json      只声明 WHICH problem 和 gate 阈值          │
│    ├── problem.json   由 harness 生成，冻结契约（不手改）         │
│    ├── candidate.py   ★ 你写/改的优化实现，唯一要动的东西 ★      │
│    ├── workload.jsonl M sweep 列表                              │
│    └── run.sh         一行入口，指向 evaluate_task.py            │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: 算子契约（single source of truth）                     │
│  testbench/harness/glm52_ops_amd.py                             │
│    build_inputs(op, phase, M, S, dev, seed)  造 frozen 输入      │
│    reference(op, phase, inputs)              baseline 实现       │
│    _sparse_mla_math_oracle(...)              fp32 精度参考       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: 通用 runner                                            │
│  testbench/harness/                                              │
│    evaluate_task.py       669 行编排器，所有 op 共用             │
│    candidate_loader.py    加载 candidate.py / HIP 扩展           │
│    timing.py              HIP graph capture+replay / event      │
│    reward_hack.py         L2 flush / poison buffer 反作弊       │
│    backends/rocm_amd.py   MI300X FP8/BF16 peaks + roofline      │
└─────────────────────────────────────────────────────────────────┘
```

### 关键设计原则

**"Every operator definition — inputs, reference, thresholds, masks, cost model, peaks — lives in `glm52_ops`. This file only orchestrates."**  ——  `evaluate_task.py:2`

即 task 目录只声明"我是哪个 op、哪个 phase"，剩下所有细节（tensor 形状、dtype、正确性 tolerance、cost model、GPU peaks）**全部**由 `glm52_ops_amd.py` 独家定义。这样 candidate 不可能"改契约来让自己过 gate"，因为契约根本不在它能修改的目录里。

---

## 2. 一次 `./run.sh` 里发生了什么

`run.sh` 内部就是一句：

```bash
python testbench/harness/evaluate_task.py <this_task_dir> [args...]
```

`evaluate_task.py:244 def evaluate()` 是主循环。对 `workload.jsonl` 里每个 M 依次执行 **6 步**：

### Step 1 — 造 frozen 输入

```python
inputs = ops.build_inputs("dsa_attn", "prefill", M=4096, S=32768, device=dev, seed=0)
# → {"q":       [4096, 64, 576] bf16,
#    "kv":      [32768, 576]   bf16,
#    "indices": [4096, 2048]   int64,
#    "sm_scale": 0.04166..., "d_v": 512}
```

**同一个 dict 既喂 reference 也喂 candidate** —— fairness 的基础。种子固定，每次运行输入完全一致。

### Step 2 — 跑 reference，然后"下毒"

```python
ref_out = ops.reference("dsa_attn", "prefill", inputs).clone()
poison_buffer_in(inputs)   # inputs 里的共享 out buffer 被写成 NaN
```

**为什么下毒**（`evaluate_task.py:21-30`）：GEMM/MoE 家族的 op 用共享输出 buffer，reference 会把结果写到 `inputs["out"]`。一个偷懒 candidate 写 `return inputs["out"]` 也能过正确性，还能 latency ≈ 0，把 roofline reward 顶到天花板。下毒后这种作弊立刻变成 all-NaN 出错。

`dsa_prefill_attn` 没有共享 buffer，这一步实际是 no-op，但流程是通用的。

### Step 3 — 跑 candidate，过 3 层正确性 gate

```python
cand_out = candidate.run(inputs)
```

三层校验（`problem.json` 的 `correctness.layers`）：

| 层 | 检查 | 门槛 | 拦不住什么就补下一层 |
|---|---|---|---|
| **L1 anomaly** | `inf/-inf/nan` 位置和 reference 完全对齐 | 硬 gate | inf 位置错就是算法错 |
| **L2 elementwise** | 每个元素：`abs_err < abs_tol` **OR** `rel_err < rel_tol` | `rel=0.0157, abs=1e-4·|ref|.max()` | "OR" 让大元素走相对误差、近零元素走绝对误差 |
| **L3 calc_diff** | `‖x-y‖² / (‖x‖² + ‖y‖²)` | `≤ 5e-6` | scale-sensitive，能抓到 `k·reference` 这种均匀 scale 错 |

只有三层都过才进 Step 4。**若不过，直接返回 exit 2（`INCORRECT_OR_INCOMPLETE`）**，不进入计时。

### Step 4 — 计时

时序协议（`problem.json` 的 `performance`）：

- **timer**: HIP graph capture+replay（默认）→ 环境不支持时 fallback HIP event
- **默认参数**: `warmup=3, repeat=10, iterations=30`
- **每次 iteration 前**：clone 输入 + flush L2（都在测量窗口外，不算进 latency）
- **测量对象**：设备端 kernel 时间，**不是** wall-clock

**为什么必须是 device time**（`evaluate_task.py:36-49`）：reward 是 hardware-utilisation ratio。wall-clock 会把 host dispatch 也算进去 —— 这个 op 上 wall-clock ~99µs 里 ~52µs 是 host stall，用 wall-clock 算出来的"利用率"就是虚的。CUDA graph 或 CUPTI 才能只测 device span。

采集 K=10 个样本，得到分位数：

- `reference_us` (median), `reference_us_p10`（最有利于 reference 的读数）
- `candidate_us` (median), `candidate_us_p90`（最不利于 candidate 的读数）

### Step 5 — Post-timing recheck

计时后**重新** `build_inputs` + 跑一次 candidate，重新过 L2+L3。抓的是"kernel 在计时循环里跨 iteration 慢慢污染 state"的 bug（比如无意中改了输入张量）。

### Step 6 — Shape verdict

```
win     : reference_p10 / candidate_p90 > 1.0    保守读数下 candidate 也赢
regress : reference_p90 / candidate_p10 < 1.0    有利读数下 candidate 还是输
neutral : 都不满足                                 噪声带内，不否决
```

**为什么用分位数而不是 median 或 max/min**（`evaluate_task.py:52-68`）：

- median-vs-median：K=1 时噪声 ±5%，identical-to-reference 的 candidate 也能有一半概率 > 1.0，形同虚设
- max/min：单个 outlier 就能翻盘。K=10 时 sp_cons 0.347 但 median 0.999 —— 已经观察到
- p10-vs-p90：K 越大 gate 越紧，才是合理设计

---

## 3. 跨 shape 聚合 → 最终判决

一个 task 的 `workload.jsonl` 里所有 shape 跑完后：

- **run-level gate**（`task.json:performance_gate`）：`≥1 shape win` **且** `0 shape regress`
- `aggregate.geomean_speedup` = 各 shape speedup 的几何平均
- `aggregate.min_speedup` = 最差 shape 的 speedup

**Terminal state**：

| state | exit | 含义 |
|---|---|---|
| `COMPLETE_WIN` | 0 | 正确 + ≥1 win + 0 regress |
| `NO_WIN_WITH_EVIDENCE` | 1 | 正确但全 neutral（跟 reference 打平） |
| `PARTIAL_OR_REGRESSED_WITH_EVIDENCE` | 1 | 正确但有 shape regress |
| `INCORRECT_OR_INCOMPLETE` | 2 | 正确性挂了 or workload 没跑完 |

Exit code 直接可以接 CI。

---

## 4. Candidate 的写法与加载

### 4.1 ABI

`candidate.py` 只需要暴露一个函数：

```python
def run(inputs: dict):
    # inputs 就是 build_inputs 那个 dict —— 直接消费，别 re-quantize / re-seed / rebuild
    return output_tensor
```

**"frozen inputs" 原则**：`inputs` 里的张量已经量化、种子、layout 都定好了。**不要**在 `run()` 里再 quantize / seed / rebuild —— 那样你测的问题就跟 reference 测的问题不一样了。允许的操作：`.contiguous()`、view/reshape、切片。

### 4.2 两种形态

`candidate_loader.py` 支持：

1. **单文件 `.py`**：`candidate.py` 里定义 `run`。参见 `dsa_prefill_attn/candidate.py`（178 行，纯 Triton）。
2. **HIP/C++ 扩展目录**：目录里放 `candidate.py`（负责 `torch.utils.cpp_extension.load` 编译 kernel）+ `.cpp/.hip` 源码。**编译放在 `run()` 之外**，不算进 kernel 计时。

### 4.3 加载优先级

`candidate_loader.py`：`candidate.py → impl.py → solution.py (legacy) → reference`。最后一档 fallback 到 reference 意味着"测 backend vs 自己"，speedup ≈ 1.0，是"有意义的基线"而非"通过"。

### 4.4 用外部 candidate 测（不动 task）

```bash
./run.sh --candidate /path/to/my_kernel.py
./run.sh --candidate /path/to/my_kernel_dir/    # 目录里必须有 candidate.py
```

这样你可以把 candidate 放到任何地方 —— task 目录完全不用碰，也不会污染 git working tree。

### 4.5 fallback 是允许的

`run()` 里 **可以**按 shape 分派：

```python
def run(inputs):
    M = inputs["q"].shape[0]
    if M <= 1024:
        return my_fast_kernel(inputs)
    else:
        return glm52_ops.reference("dsa_attn", "prefill", inputs)  # 大 M 走 reference
```

这就是 SGLang 自己 `deepgemm_w8a8_block_fp8_linear_with_fallback` 的做法。fallback shape 判 `neutral`，不否决。**但每个 shape 都 fallback 等于零 wins，还是 fail**。

---

## 5. 完整案例：`dsa_prefill_attn`

以下是**同事从零开始上手**完整流程。

### 5.1 前置

- 硬件：AMD MI300X（gfx942），ROCm 7.0
- Python venv（此机器上是 `/root/venvs/rocm-torch/`）里有：
  - `torch 2.10.0+rocm7.0`
  - `amd-aiter`（editable）
  - `sglang-kernel 0.4.3`
  - `triton 3.6.0`
- sglang 需是 fork **`qhy991/SGLang-DGMK:decode-fusion-r1`**（而不是官方 sglang），否则 aiter 生产 baseline 拉不起来
- aiter 需要一个 checkout（此机器上是 `/root/repos/aiter`）

### 5.2 拿代码

```bash
git clone git@github.com:qhy991/kernel-harness.git
cd kernel-harness
git checkout amd     # 或 main —— PR #13 之后两个分支都包含
```

确认 tip 上有 commit `8d034d5 glm52_amd: land op-level winners for index_score / moe_total / dsa_prefill`。

### 5.3 配环境变量

```bash
export ROCM_TORCH_VENV=/root/venvs/rocm-torch       # 含 torch/aiter/triton
export SGLANG_DIR=/root/repos/sglang                # decode-fusion-r1 fork
export AITER_PATH=/root/repos/aiter
export HIP_VISIBLE_DEVICES=0                        # 挑一张空闲 MI300X
# 以下 run.sh 会自动设，如需覆盖可自己 export
# KERNEL_HARNESS_PLATFORM=rocm
# KERNEL_HARNESS_PROFILE=amd-mi300x
# KERNEL_HARNESS_PROVIDER=aiter-torch-reference
# KERNEL_HARNESS_TIMER=event
```

**检查 GPU 是否空闲**（避免和别的 workload 互相干扰）：

```bash
rocm-smi --showuse    # GPU use (%) 应该都 ≤ 5%
```

### 5.4 看题面

```bash
cd testbench/tasks/glm52_amd/dsa_prefill_attn
./run.sh --describe
```

会打出：

```
TASK  dsa_attn/prefill — DSA Sparse Attention
  GLM-5.2, MI300X-DP1-TP1-EP32.  family=mla  S=32768  seed=0

  MATH   sparse MLA: q[M,64,576] attends the top-2048 of kv[32768,576] -> out[M,64,512]
  WORKLOAD   M in [1024, 2048, 4096]
  BASELINE   sglang aiter unified_attention_sparse_mla (paged)
             glm52_ops.reference('dsa_attn', 'prefill', inputs)
  ...
```

`./run.sh --describe --json` 会打完整的 `problem.json`（机器可读）。

### 5.5 跑完整 gate

```bash
./run.sh 2>&1 | tee /tmp/dsa_prefill_attn.log
echo "exit=$?"
```

**注意不要用 `| tail -N`**，会把中间几个 shape 的数据吃掉；用 `tee` 保完整 log，事后再 tail/grep。

预期跑 ~10 分钟（M=1024/2048/4096，每个 warmup 3 + repeat 10 + iterations 30 + 编译 Triton kernel 的开销）。

### 5.6 快速迭代（改代码时用）

```bash
# 只跑单个 shape，几十秒出结果
./run.sh --M 4096

# 快速探针（不能当结论，见 problem.json 里 repeat_note）
./run.sh --M 4096 --repeat 3

# 用外部 candidate（不改仓库文件）
./run.sh --candidate /tmp/my_v2/candidate.py --M 4096
```

### 5.7 判读结果

log 末尾 `RESULT_JSON_BEGIN … RESULT_JSON_END` 之间是完整 JSON。关键字段：

```json
{
  "per_shape": [
    {
      "axes": {"M": 4096, "S": 32768},
      "reference_us": 6459.1, "reference_us_p10": 6446.5,
      "candidate_us":  4540.6, "candidate_us_p90": 4548.0,
      "speedup": 1.4225, "speedup_conservative": 1.4174,
      "shape_verdict": "win",
      "correct": true, "calc_diff": 1.87e-06, ...
    },
    ...
  ],
  "aggregate": {
    "geomean_speedup": 1.4087,
    "min_speedup":     1.3258,
    "shapes_won":      3, "shapes_regressed": 0, "shapes_neutral": 0
  },
  "verdict": {
    "correct": true, "performance_ok": true,
    "status": "CORRECT",
    "exit_code": 0,
    "terminal_state": "COMPLETE_WIN"
  }
}
```

**主看这些**：
- `aggregate.geomean_speedup` — 总加速比
- `verdict.terminal_state` — 期望是 `COMPLETE_WIN`
- 每个 shape 的 `shape_verdict` — 都应是 `win`（或至少 1 个 `win` + 其余 `neutral`）
- `per_shape[i].calc_diff` — 应远小于 5e-6

**Exit code**：0（COMPLETE_WIN）· 1（正确但没赢） · 2（不正确） · 3（infra 挂了）。

### 5.8 dsa_prefill_attn 为什么能赢 1.41x

**baseline 的问题**：aiter 生产实现走 ASM `mla_decode_fwd`，gfx942 上没有 bf16 gqa=64 的 kernel，只能把 q 从 64 head **pad 到 128 head** 调 128-head kernel → 白算 ~2× 的 QK^T + P·V FLOPs（实测 MFU ~15%）。

**candidate 的思路**（`candidate.py:6-13` 的 docstring）：

1. 写一个 **native-64-head** Triton flash-attention kernel（`_dsa_prefill_fused`），做一半的 compute
2. 关键 AMD 调优：`matrix_instr_nonkdim=16` MFMA 指令（16×16×16），Triton 默认选的 32×32×8 在 skinny-M sparse-MLA dot 上慢 ~40%
3. 精度处理：QK 用 bf16 MFMA + fp32 accumulator（`tl.dot` 默认）；PV 也用 bf16 MFMA（gate 通过后确认安全，DSA_PV_F32=0 是 winner 配置）

结果：M=1024/2048/4096 分别 1.36/1.30/1.30x（保守读数下），calc_diff 1.87e-6（gate 5e-6，2.7× margin），三 shape 都 `win`。

### 5.9 只测 kernel、不走 harness

`_dsa_prefill_fused` + `_run_triton` 是纯 `torch + triton`，可以复制到独立脚本用 `torch.randn` 造输入直接调用。缺点：没有 aiter baseline 对比、没有 3 层正确性 gate、没有 poison buffer 反作弊。**改代码调试时可以这么用，出结论时必须回到 harness**。

### 5.10 baseline 代码在哪（3 层 dispatch 链）

`dsa_prefill_attn` 的 baseline **不是单一文件**，是一个 3 层的 dispatch 链，每一层存在不同 repo/目录：

```
候选 candidate.py 的 fallback 分支
       │  glm52_ops.reference("dsa_attn", "prefill", inputs)
       ▼
┌────────────────────────────────────────────────────────────────┐
│ 层 1 — harness dispatcher（只做转发）                            │
│ testbench/harness/glm52_ops_amd.py:636  reference()             │
│   → OPERATOR_PROVIDER.reference(op, phase, family, inputs)      │
└────────────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────────────┐
│ 层 2 — AMD provider（按 family/phase 选实际实现）                 │
│ testbench/harness/backends/rocm_amd.py:461  reference()         │
│                                                                  │
│   family == "mla" AND phase == "prefill":                        │
│     1. _try_aiter_asm_mla_decode(inputs)     ← 主路径 ★          │
│        rocm_amd.py:302                                          │
│     2. _try_sglang_tilelang_sparse_mla       ← fallback #1       │
│        rocm_amd.py:251                                          │
│     3. _sparse_mla_reference(inputs)         ← 兜底 torch 参考   │
│        rocm_amd.py:693                                          │
└────────────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────────────┐
│ 层 3 — 实际 GPU kernel（在 aiter 这个第三方 repo 里）             │
│ /root/repos/aiter/aiter/mla.py:197  mla_decode_fwd()             │
│   Python wrapper，内部调编译好的 ASM stage1 + Triton stage2      │
│                                                                  │
│ ASM stage1 kernel:                                              │
│   mla_dec_stage1_bf16_a16w16_subQ128_mqa128                     │
│   编译产物 → aiter/aiter/jit/module_aiter_core.so                │
│                                                                  │
│ Triton stage2 kernel:                                           │
│   _fwd_kernel_stage2_asm  (aiter/aiter/mla.py:19)                │
└────────────────────────────────────────────────────────────────┘
```

**关键的 45 行**：`rocm_amd.py:302-345` 的 `_try_aiter_asm_mla_decode()` 是本 op baseline 的"胶水层"，做 3 件事：
1. **把 q 从 64 头 pad 到 128 头**（`:326-330`）—— 这正是 aiter 白算一半的根因，也是我们 1.41× 加速的来源
2. 把 kv 重塑成 paged 布局 `[num_pages, page_size=1, nhead_kv=1, d_qk]`
3. 调 `aiter.mla.mla_decode_fwd(...)` 做实际 attention

#### 文件定位表

| 层 | 文件 | 行 | 内容 |
|---|---|---|---|
| harness 分发 | `testbench/harness/glm52_ops_amd.py` | `:636` | `reference()` 入口，只做 family 分派 |
| AMD provider 分发 | `testbench/harness/backends/rocm_amd.py` | `:461-532` | `AmdRewardbenchProvider.reference()` |
| **MLA prefill 主路径** ★ | `testbench/harness/backends/rocm_amd.py` | **`:302-345`** | `_try_aiter_asm_mla_decode()` |
| MLA prefill fallback #1 | `testbench/harness/backends/rocm_amd.py` | `:251-299` | `_try_sglang_tilelang_sparse_mla()`（gfx942 编译不了，实际用不到） |
| MLA prefill fallback #2 | `testbench/harness/backends/rocm_amd.py` | `:693` | `_sparse_mla_reference()`（纯 torch 兜底） |
| 数学 oracle（精度校验，**不参与计时**） | `testbench/harness/backends/rocm_amd.py` | `:713` | `_sparse_mla_math_oracle()` — 全 fp32 einsum |
| aiter Python wrapper | `/root/repos/aiter/aiter/mla.py` | `:197` | `mla_decode_fwd()` 定义 |
| 编译好的 kernel `.so` | `/root/repos/aiter/aiter/jit/module_aiter_core.so` | — | ASM stage1 二进制 |

#### 快速定位命令

```bash
# 1. harness 入口
sed -n '636,650p' testbench/harness/glm52_ops_amd.py

# 2. MLA 分派
sed -n '516,532p' testbench/harness/backends/rocm_amd.py

# 3. 实际 baseline wrapper（关键 45 行）
sed -n '302,346p' testbench/harness/backends/rocm_amd.py

# 4. aiter 侧的 Python 入口
sed -n '197,260p' /root/repos/aiter/aiter/mla.py

# 5. 运行时确认 aiter 装在哪
/root/venvs/rocm-torch/bin/python -c "import aiter.mla; print(aiter.mla.__file__)"
```

#### 想改哪一层？

- **改 baseline 胶水**（比如跳过 64→128 pad、换到 aiter v4 API）→ 改 `rocm_amd.py:302`
- **改实际 GPU kernel** → 去 `/root/repos/aiter/` 改 HIP/ASM 源码并 `pip install -e .` 重装
- **改 candidate**（推荐的常规操作）→ 只动 `testbench/tasks/glm52_amd/dsa_prefill_attn/candidate.py`

---

## 6. 常见坑

| 症状 | 原因 | 解决 |
|---|---|---|
| `RuntimeError: No aiter …` | `SGLANG_DIR` 指到了官方 sglang | 换 `qhy991/SGLang-DGMK:decode-fusion-r1` fork |
| speedup 忽高忽低差异巨大 | GPU 被别人占着 | `rocm-smi --showuse` 确认，换空闲卡 |
| 单跑很快，`repeat=10` 慢 10 倍 | 忘了 warmup、每 iteration 编译 Triton | Triton kernel 定义放模块级，别放 `run()` 里 |
| calc_diff 边缘（5e-6 附近） | fp32 → bf16 中间 round 掉精度 | max/sum/acc 归一化 state 保持 fp32，见 `candidate.py:15-30` |
| Exit 3 | 环境 / 契约不一致 | 先跑 `./run.sh --describe` 看能不能正常输出 |
| log 里没有 `RESULT_JSON_BEGIN` | 用了 `| tail -N` 截断了 | 改用 `tee` |
| candidate 报 "No module named 'harness.inputs'" | Python 把 harness 目录放到了 `sys.path[0]` shadowing 了 stdlib | 已经修好（`evaluate_task.py:90-98`），别自己再改 sys.path |

---

## 7. 参考

- `evaluate_task.py:1-72` — runner 顶部 docstring，讲了所有设计取舍
- `candidate.py:1-36`（dsa_prefill_attn）— winning kernel 的完整 rationale
- `problem.json` — 每个 task 的冻结契约
- `testbench/docs/BACKENDS.md` — backend / profile / provider / timer 四件套
- `testbench/docs/GLM52_CANDIDATES.md` — candidate ABI 细节

---

**一句话总结**：

> 每个 op 有 1 份契约（`glm52_ops_amd.py`），1 个 runner（`evaluate_task.py`），N 个 candidate。runner 用同一份 frozen 输入喂 reference 和 candidate，先过 3 层正确性 gate，再用同一套计时协议对比，最后按 `p10 vs p90` 保守判定 win / regress / neutral，聚合到 run-level `terminal_state`。
