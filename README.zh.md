# kernel-harness

面向 Agent 的 **SGLang 内核优化任务集**（Kimi-K2.7 / MiniMax-M3）。

[`testbench/tasks/`](testbench/tasks) 下每个目录是一个 `(算子, phase)` 任务，带完整 shape
sweep。你（或 agent）只需编辑该目录的 `solution.py`，使其**对齐 SGLang 真实内核输出**，并在
sweep 上**打赢其延迟**。正确性与性能基线是同一份东西：任务里的 `reference.py`（真实 SGLang
内核）。若 `solution.py` 在每个 shape 上都正确且更快，即可作为该内核的候选 drop-in。

> 新人请先读本中文 README（或英文 [`README.md`](README.md)）。Agent 请读
> [`AGENTS.md`](AGENTS.md)。评测细节见 [`testbench/README.md`](testbench/README.md) 与
> [`testbench/docs/HARNESS_DESIGN.md`](testbench/docs/HARNESS_DESIGN.md)。

## 1. 这个仓库是什么

- **任务集，不是某个优化框架。** 共 82 个任务、24 个 family。任意 agent / 人类跑同一套循环。
- **真实 oracle。** `reference.py` 就是 SGLang 生产内核：正确性比对其输出，性能对比其 CUPTI
  测得的 device-kernel 延迟。
- **自包含评测。** 评估器、正确性、计时、防作弊都在 [`testbench/harness/`](testbench/harness)，
  只需 torch（可选 cupti），不依赖外部测试框架。
- **WIN 只是候选，不是上线。** `evaluate.py` 证明该 I/O 合约上有更好内核；`integrate.py`
  再证明它能挂进 SGLang 的 dispatch。

## 2. 前置条件

- 一台 **GPU 节点**，CUDA 与你的 SGLang checkout 匹配（任务会跑真实 Hopper/Blackwell 内核）。
- 已安装 [`uv`](https://docs.astral.sh/uv/)。
- 可用的 **SGLang checkout**（默认 `../sglang`，与本仓库同级）。它提供匹配的 torch / Triton /
  sgl-kernel / DeepGEMM / FlashInfer。
- *可选：* `../sglang-m3`（含 MiniMax DSA 稀疏栈）——仅 MiniMax DSA 任务需要。

## 3. 安装

```bash
git clone git@github.com:qhy991/kernel-harness.git
cd kernel-harness

cp testbench/harness.env.example testbench/harness.env   # 路径不同再改
./testbench/setup_env.sh                                 # 用 uv 创建仓库内 .venv
.venv/bin/python testbench/bin/check_env.py              # 环境自检
```

`harness.env` 可覆盖 `SGLANG_DIR` / `MM_M3_SGLANG_DIR` / `CUDA_HOME`；相对路径相对仓库根解析。
**唯一支持的环境是仓库内 `.venv`**，不要复用无关 conda/venv（ABI 不一致会毁掉正确性与计时）。

## 4. 选一个任务

一个任务 = `testbench/tasks/<model>/<name>/` 下的一个目录。查看清单：

```bash
.venv/bin/python testbench/bin/inventory.py
ls testbench/tasks/kimi_k27 testbench/tasks/minimax_m3
```

**推荐起步任务（优先 memory-bound / fused）。** 先别打 FP8 DeepGEMM：那是高度手调的
Blackwell 基线，故意很难赢。

- `testbench/tasks/kimi_k27/input_embedding_decode` — 访存 gather，最易上手
- `testbench/tasks/kimi_k27/q_a_layernorm_decode` — RMSNorm（同 family 还有 fused-add）
- `testbench/tasks/kimi_k27/mla_qk_rope_decode` — RoPE，有融合空间
- `testbench/tasks/kimi_k27/q_nope_absorb_bmm_decode` — 小批量 absorb BMM

## 5. 如何把任务发给 Agent

**一次只优化一个任务。** 把下面模板里的 `<TASK_DIR>` 换成 §4 的路径后粘贴给 agent：

```text
Optimize ONE kernel task only:
  <TASK_DIR>

Rules:
- Edit only <TASK_DIR>/solution.py
- Do not modify reference.py, workload tolerances, evaluate.py, or any harness code
- Read AGENTS.md and the task's task.json / reference.py first
- Loop: edit solution.py -> run evaluate.py -> read VERDICT_JSON -> iterate
- Quick check:  .venv/bin/python testbench/bin/evaluate.py <TASK_DIR> --max-workloads 1 --no-baseline
- Full gate:    .venv/bin/python testbench/bin/evaluate.py <TASK_DIR> --repeat 3
- Exit 0 = candidate WIN (correct AND faster on every shape). Then run integrate.py.
- Prefer real algorithmic/kernel improvements; do not game tolerances or timing.
```

**如何判断成功。** `evaluate.py` 会打印逐 shape 表 + 机器可读 JSON，并用退出码表示结果：

```
VERDICT_JSON_BEGIN
{"task":"...","correct":true,"win":true,"geomean_speedup":1.23,"min_speedup":1.08,"per_shape":[...]}
VERDICT_JSON_END
```

| 退出码 | 含义 |
|---|---|
| `0` | **WIN** — 每个 shape 都正确且更快 |
| `1` | 正确，但并非处处更快 |
| `2` | 不正确 / 报错 / reward-hack / sweep 不完整 |

`--repeat 3` 用多次独立进程的最坏加速比门控，避免把噪声当加速。内环可用**仅供方向参考**的
快速探针：

```bash
PYTHONPATH=testbench .venv/bin/python -m harness.profile <TASK_DIR> --shape M
```

它**不决定胜负**；权威结论永远是 `evaluate.py`。

## 6. WIN 之后

```bash
.venv/bin/python testbench/bin/integrate.py <TASK_DIR>   # exit 0 = SGLang drop-in 验证通过
.venv/bin/python testbench/bin/migrate.py  <TASK_DIR>    # 可选：生成可回滚的 SGLang patch
```

`integrate.py` 会热替换真实 dispatch 符号并跑真实 SGLang forward，确认候选被调用、输出对齐、
可还原。`migrate.py` 在 integrate 未绿时会拒绝。即便 integrate 通过，仍**不等于可上线**——还要
过更广 shape、真实模型精度、SGLang 单测、CUDA Graph、AOT/JIT 与非 Blackwell 回退等门禁。详见
[`testbench/README.md`](testbench/README.md)。

## 7. 相关文档

- [`AGENTS.md`](AGENTS.md) — Agent 优化循环（给 agent 读）
- [`README.md`](README.md) — 英文版根 README（与本文件同内容）
- [`testbench/README.md`](testbench/README.md) — 任务合约、evaluate / integrate / migrate、防作弊
- [`testbench/docs/HARNESS_DESIGN.md`](testbench/docs/HARNESS_DESIGN.md) — 正确性 / 计时 / 双层探针设计

## 8. 旧版 proxy 基线（不是 Agent oracle）

更早的轻量 **proxy** 微基准（纯 `torch` / `flash_attn` / `torch._scaled_mm`，不需要
`sgl_kernel`/`deep_gemm`）在 [`docs/legacy_proxy_baselines.md`](docs/legacy_proxy_baselines.md)。
适合看 shape、做粗 sanity check；**不能**当 SGLang 加速比分母，也**不是**正确性 oracle。不要在那里开优化循环。

## License

与你指向的 SGLang 仓库一致 — Apache 2.0。本 harness 本身为 BSD-3，便于放入内部仓库。
