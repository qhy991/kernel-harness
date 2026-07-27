# GLM-5.2 gpt-5.6 Goal 实验结果汇总

> 维护说明：分析完一个 `g56u-*` 任务后，更新「总览表」对应行，并在下方追加/完善该任务专节。  
> 证据以各 goal 目录内 `REPORT.md` / commits 为准；本文件只做人工可读摘要。  
> 最后更新：2026-07-23（已记录 goal-01–08、22–25）

## 门禁约定（各任务通用）

| 项 | 定义 |
|---|---|
| 晋级门槛 | paired p50 ≥ **1.03×**（相对 stock ≥ **3%**） |
| 计量方式 | 同进程 A/B 交替；取配对加速比的 p50，不是最好一次 |
| 重复 | 每个 bucket（通常 M16 / M32）**重复 3 轮**，均需过线 |
| 权威模式 | **CUDA graph replay**（serving decode 线上形态）；eager 仅作参考 |
| Eager vs Graph | Eager ≈ 算子裸性能；Graph ≈ 上线 replay 链（固定开销会稀释局部优化） |
| 处置 | **win / replace**：过门禁且可接入生产；**no-replacement**：不过线或不够稳，保留 stock；**blocked**：外部条件不足未完成权威验收；**local-reject + blocked**：本地 leaf 已否决且生产验收外阻；**leaf-win + no-replacement**：leaf 过线但因集成/拓扑门禁未 enable；**in-progress**：基建/战役未完成，尚无权威性能结案 |

## 总览表

| ID | 任务 | Session | 处置 | 是否获得可上线优化 | 备注 | 分析状态 |
|---:|---|---|---|---|---|---|
| 01 | dsa_decode_value_path | g56u-01-dsa-value | **no-replacement** | **否** | FlashMLA combine `MAX_SPLITS` 收紧；eager 有增益，graph 无 | ✅ 已分析 |
| 02 | dsa_decode_attention | g56u-02-dsa-attn | **no-replacement** | **否** | scheduler 变体全面回退；bank-layout 正确性失败 | ✅ 已分析 |
| 03 | dsa_prefill_attention | g56u-03-dsa-prefill | **local-reject + blocked** | **否** | Q32/Q16 tactic 大回退；缺 checkpoint / 八卡，正式生产验收外阻 | ✅ 已分析 |
| 04 | dsa_decode_score_path | g56u-04-dsa-score | **no-replacement** | **否** | page64 坐标扁平化；eager 回退，graph ~1.0×；page64-002 证据已作废 | ✅ 已分析 |
| 05 | indexer_k_weights_prefill | g56u-05-index-kw | **no-replacement（inner-gate）** | **否** | TGV 大回退；torch.mm/单流不过重复 1.03×；TP4/八卡外阻 | ✅ 已分析 |
| 06 | moe_w2_decode_pack_launch | g56u-06-w2-launch | **local no-replacement + blocked** | **否** | 无 online pack 可删；direct floor 非合法优化；registry 更慢；EP8/verifier 外阻 | ✅ 已分析 |
| 07 | moe_w2_decode_kernel | g56u-07-w2-kernel | **leaf-win + no-replacement** | **否** | BM16 leaf 1.06–1.09×，但 process-global alignment + TP8 外阻，未 swap | ✅ 已分析 |
| 08 | moe_w2_prefill | g56u-08-w2-prefill | **leaf-win + no-replacement** | **否** | PSUM 组件 1.064×；EP4 诊断过；缺 checkpoint/八卡，未 enable | ✅ 已分析 |
| 09 | moe_w13_prefill | g56u-09-w13-prefill | | | | ⏳ 待分析 |
| 10 | attn_o_decode_baseline | g56u-10-o-base | | | | ⏳ 待分析 |
| 11 | attn_o_decode_packed_port | g56u-11-o-packed | | | | ⏳ 待分析 |
| 12 | attn_o_decode_source_tuning | g56u-12-o-source | | | | ⏳ 待分析 |
| 13 | attn_o_prefill | g56u-13-o-prefill | | | | ⏳ 待分析 |
| 14 | attn_q_b_decode_packed | g56u-14-qb-packed | | | | ⏳ 待分析 |
| 15 | indexer_wq_b_decode | g56u-15-index-wq | | | | ⏳ 待分析 |
| 16 | indexer_score_decode | g56u-16-index-score-d | | | | ⏳ 待分析 |
| 17 | indexer_score_prefill | g56u-17-index-score-p | | | | ⏳ 待分析 |
| 18 | moe_w13_decode_scale_path | g56u-18-w13-scale | | | | ⏳ 待分析 |
| 19 | moe_w13_decode_kernel | g56u-19-w13-kernel | | | | ⏳ 待分析 |
| 20 | moe_w13_prefill_graph | g56u-20-w13-graph | | | | ⏳ 待分析 |
| 21 | attn_q_b_decode_source_fork | g56u-21-qb-source | | | | ⏳ 待分析 |
| 22 | dsa_flashmla_kv_production | g56u-22-flashmla | **no-replacement** | **否** | combine `max_num_splits=32`；eager 不稳、graph ~1.0×；与 01 同路 | ✅ 已分析 |
| 23 | dp_allgather_production | g56u-23-allgather | **no-replacement + blocked** | **否** | TP4 基线有；候选 oracle 零席；原子调度升级后仍 exit 75；SGLang 未改 | ✅ 已分析 |
| 24 | tp_allreduce_reachability | g56u-24-allreduce | **in-progress（基建就绪，战役未完成）** | **否（尚无结案）** | analyzer/对齐已修；新 TP4 因 exit 75 未启动；Codex 额度耗尽 | 🔄 进行中 |
| 25 | deepep_dispatch_combine | g56u-25-deepep | **no-replacement** | **否** | 源码候选回退；EP4 joint Config ~1.36× 仅诊断；SGLang 未改；EP8 外阻 | ✅ 已分析 |

**当前已确认可上线优化数：0 / 11 已结案任务（01–08、22、23、25）；goal-24 进行中不计入结案**  
**其中有实测加速但未上线：goal-07（BM16）、goal-08（PSUM）、goal-25（EP4 joint Config ~1.36×，诊断-only）**

---

## Goal-01 · dsa_decode_value_path

| 字段 | 内容 |
|---|---|
| Session | `g56u-01-dsa-value` |
| 目录 | `~/glm52-goal-runs/01-dsa_decode_value_path/` |
| 运行状态 | pane dead（约 2026-07-22 14:18）；monitor 已 close；`status.sh` 可能仍显示 running（陈旧） |
| 分支 (harness) | `goal/glm52-prod-01-dsa_decode_value_path` @ `5c359d1` |
| 处置 | **no-replacement** |
| 是否获得可上线优化 | **否** |
| 未 push | 是 |

### 目标路径

- 生产 value-path 实际走 **FlashMLA KV**（`flash_fwd_splitkv_mla*` + combine），不是 TRT-LLM
- SGLang 字面类名：`DeepseekSparseAttnBackend._forward_flashmla_kv`
- Plan 标签：`DSAAttentionBackend._forward_flashmla_kv`（仅命名差异）
- 符号：`sgl_kernel.flash_mla.flash_mla_with_kvcache`
- Workload：`dsa_flashmla_kv_decode_m16` / `dsa_flashmla_kv_decode_m32`（`serving_native/workloads.py`）
- TRT-LLM workload/runner 与 parent **字节级一致**（未改、未重标）

### 尝试的优化

- Candidate：收紧 sparse decode combine 的 `MAX_SPLITS`（stock 模板选 160，观测只需 8/4 splits）
- 数值正确（graph 下 BF16 `max_abs_diff = 0`）
- 主耗时仍在 split-KV kernel（带宽 / long-scoreboard）；combine 优化带不动整条 graph 链

### 门禁结果

| Bucket | Eager gate | Graph gate（权威） | 政策 |
|---|---|---|---|
| M16 | 过（约 1.039–1.045×） | **失败**（约 0.995–1.000×） | stock |
| M32 | 未全过（一轮 1.027×） | **失败**（约 0.998–1.001×） | stock |

Eager 代表数据（paired p50）：

| Bucket/run | Ref p50 µs | Cand p50 µs | paired p50 |
|---|---:|---:|---:|
| M16/1 | 44.160 | 42.560 | 1.0446× |
| M16/2 | 44.768 | 42.896 | 1.0393× |
| M16/3 | 44.416 | 42.592 | 1.0417× |
| M32/1 | 56.432 | 54.336 | 1.0363× |
| M32/2 | 49.168 | 47.424 | 1.0300× |
| M32/3 | 49.104 | 47.632 | **1.0271×**（未过） |

Graph 代表数据（paired p50）：

| Bucket/run | Ref p50 µs | Cand p50 µs | paired p50 |
|---|---:|---:|---:|
| M16/1–3 | ~30.4–30.7 | ~30.3–30.7 | **0.995–1.000×** |
| M32/1–3 | ~35.8–36.3 | ~35.9–36.3 | **0.998–1.001×** |

### 关键 commits（本地）

1. `245ff19` — `serving_native: add explicit flashmla_kv decode workloads`
2. `081e58a` — `evidence: record DSA value-path no-replacement result`
3. `5c359d1` — `evidence: clarify FlashMLA KV reachability scope`（仅证据标注，未重跑 GPU）

### 关键证据路径

- `kernel-harness/profile/dsa-flashmla-kv-experiment/REPORT.md`
- `kernel-harness/profile/dsa-flashmla-kv-stock/`
- `kernel-harness/profile/dsa-trtllm-stock-v0612/`（负证据：默认会落到 TRT-LLM）

### 阻塞项（未削弱门禁）

- TP8/DP8/EP8 端到端门禁：主机仅 4×B200，且会话只授权物理 GPU 1 → 未跑八卡验收
- 因 graph 门禁已失败，缺端到端也**不改变** no-replacement 结论

### 一句话

Candidate 在 eager 上有可见加速，但 **CUDA graph 权威门禁下无实质收益**，故不替换 stock `flashmla_kv`。

---

## Goal-02 · dsa_decode_attention

| 字段 | 内容 |
|---|---|
| Session | `g56u-02-dsa-attn` |
| 目录 | `~/glm52-goal-runs/02-dsa_decode_attention/` |
| 运行状态 | pane dead（约 2026-07-22 15:37）；已完成 |
| 分支 (harness) | `goal/glm52-prod-02-dsa_decode_attention` @ `d91dc16` |
| 分支 (sglang) | 同名分支 @ `5db482acd`（isolated FlashMLA namespace hook） |
| 处置 | **no-replacement** |
| 是否获得可上线优化 | **否** |
| 未 push | 是 |
| 用时 / token | ~1h 33m；约 1.33M tokens |

### 目标路径

- 与 goal-01 同属 **显式 `flashmla_kv` decode**（非 TRT-LLM / 非 `flash_mla_sparse_fwd`）
- 调用链：`DeepseekSparseAttnBackend._forward_flashmla_kv` → `flash_mla_with_kvcache` → `fwd_kvcache_mla` → split-KV main + combine
- Workload：exact M16/M32 FlashMLA-KV（`538aceb` 加入）
- 权威测量：immutable campaign `screen3`（单 flexible-GPU wrapper，物理 GPU 1）

### 尝试的优化（两次，均 reject）

1. **Scheduler / useful-partition 缩减**（配置型）
   - 假设：减少 combine 读的 partial 输出，同时保留足够 main 并行
   - M16 useful partitions 128→120；M32 128→112（metadata 行数仍 148）
   - 正确性：过（eager + fresh-input graph replay）
   - 效果：combine DRAM/耗时下降，但 **main kernel 变慢更多** → 整体回退

2. **V32 group-major raw-NoPE bank rotation**（源码型）
   - 假设：旋转 shared-memory bank，降低实测 bank conflict / excessive wavefronts
   - 布局：每组四行 + 16B pad → **group stride = 2064 bytes**
   - 正确性：**未过线即失败**（未进入 timing）
   - 根因：TMA `UTMALDG.2D.GATHER4` 要求 128B shared 对齐；2064 使第二组目的地址落到 `0x24810`（未对齐）
   - 修正空间不足（128B pad 会撞 shared 上限），无可信后续变体可测

### 门禁结果

| Bucket / candidate | Eager paired p50（3 轮） | Graph | 正确性 | 决策 |
|---|---|---|---|---|
| M16, scheduler 120 parts | **0.958 / 0.878 / 0.950×**（全回退） | 单次 1.035×，但 p10–p90 宽且 eager 全负 | Pass | Reject |
| M32, scheduler 112 parts | **0.871 / 0.885 / 0.875×**（全回退） | 0.895× | Pass | Reject |
| M16/M32 bank-layout | 未跑 | 未跑 | **Fail**（M16 首次调用） | Reject before timing |

Profiler 要点（scheduler）：M16 main 22.3→24.1 µs；M32 main 30.4→35.5 µs。combine 略省，盖不住 main 损失。

Stock graph 正确性通过（oracle max err ≈ 3.05e-5；graph median ≈ 30.8 / 36.0 µs）。

### 关键 commits（本地）

- Kernel-Harness：`538aceb`（exact FlashMLA-KV workloads）、`d91dc16`（no-replacement 证据）
- SGLang：`5db482acd`（isolated operator namespace）
- Isolated FlashMLA：`beeba02f`（bank-layout 实验；生产仍回滚 parent）

### 关键证据路径

- `kernel-harness/profile/dsa_flashmla_kv_production_20260722/REPORT.md`
- `.../attempt_ledger.json`
- `.../validation_summary.json`
- `.../validation_blockers.json`
- 另有未跟踪目录：`dsa_flashmla_kv_bank_conflict_20260722/`、`dsa_flashmla_kv_scheduler_campaign_20260722/` 等（证据保留）

### 阻塞项

- TP8/DP8/EP8 + 全模型验收：外部不可用（4×B200、无 GLM-5.2 weights）
- **未削弱门禁**；且 leaf 已双失败，缺端到端不改变 no-replacement
- Fallback：`SGLANG_GLM52_OPT=0`，M16/M32 均 stock；生产 dispatch 未改

### 与 goal-01 对照

| | Goal-01 | Goal-02 |
|---|---|---|
| 路径 | 同：`flashmla_kv` | 同 |
| 主要招数 | combine `MAX_SPLITS` 收紧 | scheduler 分区缩减；shared bank 旋转 |
| Eager | M16 过 3%，M32 差一点 | **全面回退** |
| Graph | ~1.00×，不过线 | 单点噪声/回退，不晋级 |
| 正确性 | Pass | scheduler Pass；bank-layout **Fail** |
| 结论 | no-replacement | no-replacement |

### 一句话

两条候选都没过 leaf 门禁：**scheduler 变体正确但全面变慢，bank-layout 因 2064B TMA 对齐错误直接挂**；M16/M32 继续 stock `flashmla_kv`。

---

## Goal-03 · dsa_prefill_attention

| 字段 | 内容 |
|---|---|
| Session | `g56u-03-dsa-prefill` |
| 目录 | `~/glm52-goal-runs/03-dsa_prefill_attention/` |
| 运行状态 | pane dead（exit 0）；已完成可行本地工作 |
| 分支 (harness) | `goal/glm52-prod-03-dsa_prefill_attention` @ `640f7e4` |
| 分支 (sglang) | 同名分支 @ `5a444f66c`（PDL trial 已 revert，与 stock 源码一致） |
| 处置 | **local-reject + production externally blocked** |
| 是否获得可上线优化 | **否**（未 enable 任何 SGLang 优化；stock FlashInfer Q64 仍活跃） |
| 未 push | 是；两 worktree clean |

### 目标路径

- Prefill 默认解析到 **FlashInfer TRTLLM-gen**，不是 FlashMLA sparse fwd
- 调用链：`DeepseekSparseAttnBackend.forward_extend` → `_forward_trtllm(is_prefill=True)` → `trtllm_batch_decode_with_kv_cache_mla(backend="trtllm-gen")`
- 决策 leaf：exact **513-page** raw pool（含 leading dummy page，offset-64），M4096 / ctx32768 / FP8
- 冻结任务 `dsa_prefill_attn`（flat BF16 `flash_mla_sparse_fwd`）判定 **operationally mismatched**，不用于决策

### 尝试的优化

1. **PDL off**（外部 candidate / 历史 SGLang trial）  
   - 约 **0.997×**，中性偏慢 → reject  
   - SGLang 侧 `b03db3f` 曾加 guarded PDL policy，**无持久化性能跑**后由 `5a444f66` revert

2. **强制 Q32 / Q16 `PersistentSwapsAb` tactic**（相对 stock Q64 `PersistentKeepsAb`）  
   - 正确性：runner comparison 通过  
   - 性能：**大回退**（见下表）→ reject，未集成

### 门禁结果（决策 leaf：513-page raw-pool）

| Tactic | Candidate p50（runs 中位） | Paired speedup p50 | 决策 |
|---|---:|---:|---|
| stock Q64 Keeps（control） | 0.887 ms | 1.0006× | neutral |
| Q32 Swaps | 1.608 ms | **0.541×**（约 −46%） | reject |
| Q16 Swaps | 2.924 ms | **0.297×**（约 −70%） | reject |

Backend-class checkpoint-free region：CUDA-event mean **0.947472 ms**（attention ≈ 88%，RoPE/FP8 ≈ 11%）。

NCU 诊断：Q32/Q16 与 Q64 做同样 tensor math / 读同样 bytes，但 waves / shared 指令 / bank conflict 暴增 → 更慢。

### 关键 commits（本地）

- Harness（用户摘要要点）：`d001817`（workload / source trial / benchmarks / profiler）、`640f7e4`（audited report + knowledge）  
  - 同分支另有更早收尾提交：`8610fad`…`7f5253a` 等
- SGLang：停在 rollback HEAD `5a444f66…`（无启用优化）

### 关键证据路径

- `kernel-harness/evidence/glm52_prod_03_dsa_prefill_attention/FINAL_REPORT.md`
- `.../paired_summary_v2.md`
- `.../external_validation_blocker.md`
- `.../source_overlay_manifest.md`
- `.../attempt_ledger.md`

### 阻塞项（正式生产完成外阻）

| 项 | 状态 |
|---|---|
| GLM-5.2 checkpoint | **缺失**（配置模型目录空） |
| 真请求 reachability / live indexer / full DSA region | blocked |
| Graph / E2E prefill | blocked |
| TP8/DP8/EP8 八卡验收 | blocked（主机仅 4×B200） |
| 已完成的 DP4 诊断 | 明确 **diagnostic only**，不替代八卡门禁 |

报告明确：**不声称** plan 的正式 production no-replacement 结案；本地只做到「候选否决 + stock 保留」，正式生产验收仍外阻。

### 与 01/02 对照

| | 01 / 02 | 03 |
|---|---|---|
| 阶段 | decode `flashmla_kv` | **prefill TRTLLM-gen** |
| 本地 leaf | 有完整 paired 结论 | Q32/Q16 **decisive reject** |
| 生产结案 | 可写 no-replacement（leaf 已否决） | **外阻**：缺权重 + 缺八卡，graph/E2E/真请求未跑 |

### 一句话

可行本地工作已做完并 commit：**Q32/Q16 大回退、PDL 无增益，stock Q64 保留**；但缺 checkpoint 与八卡，**正式生产完成被外部阻塞**，未 enable 任何优化。

---

## Goal-04 · dsa_decode_score_path

| 字段 | 内容 |
|---|---|
| Session | `g56u-04-dsa-score` |
| 目录 | `~/glm52-goal-runs/04-dsa_decode_score_path/` |
| 运行状态 | pane dead（exit 0）；已完成并本地 commit |
| 分支 (harness) | `goal/glm52-prod-04-dsa_decode_score_path` @ `9d2f5db` |
| 分支 (sglang) | 同名分支 @ `d33ad5bf4`（isolated FlashMLA namespace 支持；**无**实验 namespace引用） |
| 处置 | **no-replacement** |
| 是否获得可上线优化 | **否** |
| 未 push | 是；两 worktree clean |

### 目标路径

- 与 01/02 同属 decode **`flashmla_kv`** 融合路径（score 嵌在 split-KV main 内，不可单独计时）
- 调用链：`_forward_flashmla_kv` → `flash_mla_with_kvcache` → `fwd_kvcache_mla` → SM100 V32 split-KV + combine
- Score 相关源码区：online softmax / split write-out / tcgen05 累加 / sparse-index+TMA producer（`v32.cu` / `kernel.cuh`）+ `combine.cu`
- Bucket：生产本地 M16/M32（attention TP=1）

### 尝试的优化

**page64 flattened coordinate**（改 FlashMLA baseline 源码，非手写新 kernel）：

- 假设：page size 64 时，物理 token index 可直接当 TMA 坐标 / scale 地址（`index*656`），省掉整数坐标重建
- 实现：FlashMLA experiment commit `5fa2b1f`（base `05e26647`）；patch `0001-page64-flattened-coordinate.patch`
- 形态：隔离 DSO `sgl_kernel_goal04_page64`；stock 先加载，候选 `RTLD_LOCAL` + **`-Wl,-Bsymbolic`（`DT_SYMBOLIC`）** 防符号抢占

### 证据治理（重要）

| Attempt | 状态 | 说明 |
|---|---|---|
| page64-001 / 003 | invalid | 过程作废 |
| **page64-002** | **证据作废（保留审计）** | 缺 `DT_SYMBOLIC`；弱符号 launcher 有 `JUMP_SLOT`，可能绑到已加载的 **stock** → 计时/profiler **不能**证明跑了候选 |
| **page64-004** | **权威** | artifact SHA `063882d…c853`；无 sparse-launcher 动态重定位；决策仅以此为准 |

### 门禁结果（page64-004）

正确性：M16/M32 过（含 CUDA Graph mutation replay、interspersed `-1`、非默认 stream、unsupported-scale → stock fallback）。

| 模式 | paired speedup | 相对 1.03× |
|---|---|---|
| Eager M16 | **0.905 / 0.933 / 0.924×** | 全 fail（回退） |
| Eager M32 | **0.908 / 0.916 / 0.910×** | 全 fail（回退） |
| Graph M16 | 0.997 / 0.992 / 1.001× | 全 fail |
| Graph M32 | 0.993 / 0.982 / 0.987× | 全 fail |

Profiler 解读：主核仍受 long-scoreboard / barrier / 低 eligible warps 束缚；去掉坐标重建**打不中**绑定瓶颈 → eager 反而变慢。

### 关键 commits（本地）

- `34f612e` — 初版 no-win 文档（其后发现隔离缺陷，视为 provisional）
- `9d2f5db` — **correct FlashMLA isolation evidence**（权威收尾）
- 临时 nested FlashMLA checkout 已删；可由 patch + bundle + reconstruction script 恢复

### 关键证据路径

- `kernel-harness/profile/dsa-flashmla-score-page64-b200-20260722/REPORT.md`
- `.../attempts/page64-004/`（权威）
- `.../attempts/page64-002/analysis/EVIDENCE_INVALID.md`
- `.../source/0001-page64-flattened-coordinate.patch`

### 阻塞 / 政策

- Region / E2E / TP8·DP8·EP8：**微基准已拒绝后未推进**（非削弱门禁）
- Enable：无；M16/M32/unsupported/完整 server 均 stock
- 无 tracked SGLang 源码引用实验 namespace；fallback 校验通过

### 一句话

**page64 坐标扁平化**在正确隔离重建后确认：正确性过，但 eager 全面回退、graph 近 1.0×，不过 3% 门禁；先前 page64-002 因 ELF 符号抢占已作废。Stock 继续。

---

## Goal-05 · indexer_k_weights_prefill

| 字段 | 内容 |
|---|---|
| Session | `g56u-05-index-kw` |
| 目录 | `~/glm52-goal-runs/05-indexer_k_weights_prefill/` |
| 运行状态 | 本地工作已 commit 收尾（~5h / ~2.86M tokens） |
| 分支 (harness) | `goal/glm52-prod-05-indexer_k_weights_prefill` @ `1e38cef` |
| 分支 (sglang) | 同名分支 @ `2fbd443a1`（K-before-Q trial 已 revert；`dsa_indexer.py` 与 stock 字节一致） |
| 处置 | **no-replacement（validated rank-local inner gate）**；非 TP8 生产验收 |
| 是否获得可上线优化 | **否**（stock dual-stream 仍是唯一启用实现） |
| 未 push | 是；两 worktree clean |

### 目标路径

- 目标：`Indexer.forward_cuda` → `_fused_q_prepare_and_store`（fused prepare/store 子区）
- 固定模型：`GLM-5.2-NVFP4@aec724e8…`；rank-local prefill **M4096**；indexer `wq_b` / `wk_weights_proj` 为 **BF16 UnquantizedLinear**（非 FP8）
- Stock 调度：**dual-stream**（current: wk→wait(wq)→Q…；alternate: wq→wait(wk)→K/cache…）
- 权威 workload：`indexer_fused_prepare_store_prefill_m4096_eager_dual_stream`（+ 孤立投影 `indexer_wk_weights_prefill_m4096`）
- 注：早期 FP8 `wq_b` + generic RoPE 战役已 **superseded**，不参与最终决策

### 尝试的优化（多为外部 candidate / 调度，非手写新 kernel）

| # | 候选 | 类型 | 做法 |
|---|---|---|---|
| 1 | CuTe-DSL **TGV** | 换已有 GEMM tactic | 只替换 `wk_weights_proj` BF16 backend |
| 2 | FlashInfer 多 backend 扫 | 库 tactic 扫 | auto / cuBLASLt / cuDNN / CUTLASS / TGV（孤立投影） |
| 3 | **`torch.mm`** | 换调用形态 | `torch.mm(x, weight.t())`，其余 region/stream 仍 stock |
| 4 | **single-stream** | 调度开关 | `enable_dual_stream=False`，仍用 stock linear |
| 5 | K-before-Q（历史） | **改 SGLang 源码** launch 顺序 | `a75a772a2` → 因错误 ABI 战役排除决策 → `2fbd443a1` revert |

### 门禁结果（immutable campaign，权威）

晋级要求：**三次重复全部 ≥1.03×**（不是“有一次过线”）。

| Attempt | 三次 paired speedup | 决策 |
|---|---|---|
| TGV · fused region | **0.564 / 0.563 / 0.554×** | reject（稳定大回退） |
| TGV · isolated | 0.329 / 0.322 / 0.361× | reject |
| `torch.mm` · fused region | 1.004 / **1.033** / 1.003× | reject（**仅 1/3 过 1.03×**） |
| `torch.mm` · isolated | 0.984 / 0.996 / 0.996× | reject |
| single-stream · fused region | 1.013 / 0.985 / 0.978× | reject（无一稳定过线） |

FlashInfer 孤立扫：全部远低于 1.0×（0.11–0.32× 量级）。

### 关键 commits（本地）

- Harness：`727cc58`（exact validation）、`95060f3`（TP4 provenance/venv 路径修复）、`1e38cef`（finalize inner-gate）
- SGLang：`a75a772a2`（trial）/ `2fbd443a1`（revert）

### 关键证据路径

- `evidence/glm52_prod_05_indexer_k_weights_prefill/FINAL_REPORT.md`
- `.../paired_results_summary.md`
- `.../attempt_ledger.md`
- `.../hardened_runs/20260722T174049Z-immutable/validation.json`

### 阻塞项

| 项 | 状态 |
|---|---|
| TP4/DP4/EP4 live | **未执行**：180 次 allocation 均 exit-75；非验收替代 |
| Q/K NCU retry | 三次 exit-75（四卡车道占用） |
| TP8/DP8/EP8 | **外部阻塞**（主机仅 4×B200）；未削弱/未重标 |
| 生产 enable | 无；一切 shape/ABI/graph/topology 回退 stock dual-stream |

### 一句话

在固定模型 BF16 dual-stream prepare/store 内部门禁上：**TGV 大回退，`torch.mm`/单流都无法三次稳定 ≥1.03×**；stock dual-stream 保留。TP4 未跑成，八卡验收外阻。

---

## Goal-06 · moe_w2_decode_pack_launch

| 字段 | 内容 |
|---|---|
| Session | `g56u-06-w2-launch` |
| 目录 | `~/glm52-goal-runs/06-moe_w2_decode_pack_launch/` |
| 运行状态 | 四卡主机上可做工作已 commit；goal 标为 **blocked（非 complete）** |
| 分支 (harness) | `goal/glm52-prod-06-moe_w2_decode_pack_launch` @ `8e6d8ca` |
| 分支 (sglang) | 同名分支 @ `5fefdc10f`（mocked overlap/ABI tests；无生产 enable） |
| 处置 | **本地 no-replacement** + **EP8 / GPU verifier 外阻** |
| 是否获得可上线优化 | **否**（M16/M32/其他 bucket 均 stock；`SGLANG_GLM52_OPT=0`） |
| 未 push | 是 |

### 目标路径 / 设定目标

- 焦点：MoE **W2 decode pack/launch handoff**（SwiGLU+quant → DeepGEMM grouped W2），不是重写 W2 算术核
- 初始假设之一：去掉 online float→packed UE8M0 scale pack → **本地证伪**：native packed int32 UE8M0 已到 W2，`online_w2_scale_adapter_required=false`
- 绑定瓶颈：host / inter-kernel gap（约几十 µs），设备两核合计约 **81–82 µs**；测窗内无 `cudaMalloc*`/`cudaFree*`
- 晋级仍要 ≥1.03× 且 **保留** wrapper / recipe / overlap / signal / return 等生产合约

### 尝试的优化（多为 launch/包装层，非新手写 kernel）

| 候选 | 类型 | 结果 |
|---|---|---|
| **direct DeepGEMM launch** | 绕过 SGLang grouped wrapper | current M16/M32 **1.082× / 1.086×** —— 只作 **诊断下界**，非可上线优化（丢合约） |
| **output reuse** | 预热复用 W2 输出缓冲 | **1.033× / 1.002×** —— 不稳健、并发/stream/graph 不安全 → reject |
| **fail-closed registry** | 走已有 GLM52 registry 集成 | **0.894× / 0.904×** —— handoff 全面更慢 → reject |
| Isolated W2 变体 | 同设备核不同调用路径 | **~1.00×** 中性 —— 差异来自 host/gap，不在核内 |

### 关键 stock 数字（current-source 5/9，pooled p50）

| Bucket | isolated W2 | handoff |
|---|---:|---:|
| M16 | 0.0910 ms | 0.1327 ms |
| M32 | 0.0911 ms | 0.1241 ms |

（另保留 plan catalogue `expected_m=4/8` 与 current `5/9` 为独立 workload；部署 EP8 的真实 `expected_m` 仍需外线确认。）

### 战役与验证

- Locked campaign：`artifacts/moe_w2_decode_pack_launch_20260722_a/` — **215/215** rows，26/26 local graph checks，strict reduce 零错误
- EP4：仅 **diagnostic**（64 local experts 等，非 EP8 合约）
- GPU `verify_harness.py`：多次 all-GPU-lock **exit 75** 未启动 —— 记为本地调度验证缺口，未用裸 CUDA 绕过
- EP8 production validation：**不可用**（仅 4×B200）→ goal **blocked, not complete**

### 关键 commits（本地）

- Harness：`8e6d8ca`（final verifier refusal 文档；战役时点更早为 `eeef459` 等）
- SGLang：`5fefdc10f`（masked MoE overlap fallback ABI tests）

### 关键证据路径

- `serving_native/evidence/06_moe_w2_decode_pack_launch/final_report.md`
- `.../measurement_status.md`
- `.../hypotheses_and_attempts.md`
- `.../ep8_external_acceptance_runbook.md`
- `artifacts/moe_w2_decode_pack_launch_20260722_a/campaign_summary.md`
- `testbench/knowledge/entries/glm52--moe_down_proj_decode--b200--20260722a.json`

### 政策 / 回滚

- Stock 对 M16、M32、其他 bucket、未知 overlap 状态全部有效
- `$SGLANG_GLM52_ENV_FILE`、默认 `glm52_opt.env`、仓库 `runtime.env` 须缺失或解析为 OPT0，再设 `SGLANG_GLM52_OPT=0`

### 一句话

**没有可删的 online scale pack**；合法 registry 路径更慢，direct floor 虽 >3% 但绕过生产合约不可上线；本地 no-replacement，**EP8 与 GPU verifier 使 goal 仍 blocked**。

---

## Goal-07 · moe_w2_decode_kernel

| 字段 | 内容 |
|---|---|
| Session | `g56u-07-w2-kernel` |
| 目录 | `~/glm52-goal-runs/07-moe_w2_decode_kernel/` |
| 运行状态 | ~5h 21m 收尾；本地 commit clean；未 push |
| 分支 (harness) | `goal/glm52-prod-07-moe_w2_decode_kernel` @ `8bca9f9` |
| 分支 (sglang) | 同名分支 @ `49dc279b5`（**test-only** overlap/fallback 合约测试；无生产 swap） |
| 处置 | **leaf-win + no-replacement**（报告标题：No replacement） |
| 是否获得可上线优化 | **否**（stock DeepGEMM/SGLang 仍活跃；无 bucket enable） |
| Leaf 是否有实测加速 | **是**（BM16，约 6–9%） |

### 目标路径

- Leaf：`grouped_gemm_nt_f8f8bf16_masked` ← DeepGEMM `fp8_m_grouped_gemm_nt_masked`
- ABI：E32 / slab1024 / K2048 / N6144；FP8 + packed int32 UE8M0；BF16 out
- 前置链：DeepEP LL dispatch → fused W13 → SwiGLU+quant → **W2** → combine
- Workload：plan `expected_m=4/8` 与 current-source `5/9` 分开命名

### 尝试的优化

**DeepGEMM M-alignment 扫描**（改 DeepGEMM 运行时 tile 配置，非从零手写新 kernel）：

| Alignment | 结果 |
|---|---|
| **BM16**（选中） | 四 workload 全部过 3% 门禁，且最快 |
| BM32 | 四行过 3%，但慢于 BM16 |
| BM64 | 仅两行 M16 过线 |
| BM96 / BM128 | 不过 / 中性对照 |

机制：stock 默认 **BM128**；候选 **BM16**（load-M8、12 stages）。固定 mask 下 active M 很小（≤14），BM16 砍掉大量 padded M / epilogue 写出（DRAM write 约 40→8 MB；输出 `UTMASTG` 16→2），核时约 76→69 µs。

实现形态：调用 `set_mk_alignment_for_contiguous_layout()`；测完恢复 128。**process-global**，不是 fail-closed 按 bucket 的 SGLang oracle。

### Leaf 门禁结果（BM16，权威 paired p50）

| Workload | Stock p50 | BM16 p50 | Paired p50 | 正确性 |
|---|---:|---:|---:|---|
| M16 plan m=4 | 0.0982 ms | 0.0914 ms | **1.080×** | exact / graph / edge-mask PASS |
| M16 current m=5 | 0.1030 ms | 0.0950 ms | **1.087×** | 同上 |
| M32 plan m=8 | 0.1034 ms | 0.0964 ms | **1.076×** | 同上 |
| M32 current m=9 | 0.1006 ms | 0.0968 ms | **1.062×** | 同上 |

Graph leaf：capture/replay 30×，`max_abs=0`；edge masks（空 expert、多种 count）active-row exact。

### 为何仍 no-replacement / 未上线

1. **集成不安全**：alignment 是进程全局开关，可能误伤其他 grouped GEMM / mask；热路径上读 host 做 per-bucket 选择被禁止  
2. **TP8/DP8/EP8 region + SGLang E2E**：四卡主机 **BLOCKED**；未削弱门禁  
3. **TP4 诊断**：strict PASS，但 DeepEP 落在 **fallback 环境**（20-SM comm、IBGDA 失败等），且 **无 candidate**、非 overlap、非 TP8 证据  

→ 政策：全部 bucket disabled；stock 即 active 亦 rollback。

### 关键 commits（本地）

- Harness：`ac9d47f`（profile validator）、`224a826`（leaf campaign evidence）、`8bca9f9`（close as no replacement）
- SGLang：`49dc279b5`（lock MoE overlap fallback contract tests）

### 关键证据路径

- `evidence/glm52_prod_07_moe_w2_decode_kernel/FINAL_REPORT.md`
- `.../paired_alignment_summary.json`
- `.../external_validation_blocker.md`
- `profile/moe-w2-packed-baseline/analysis/stock128_vs_bm16_profile_comparison.md`
- `.../tp4_diagnostic/tp4_20260722T185932Z_1629770_10420/summary.json`

### 一句话

**首个 leaf 真加速**：DeepGEMM **BM16** 在四条 packed W2 workload 上约 **1.06–1.09×** 且正确性过；但因 **全局 alignment 不能安全上线** + **八卡验收外阻**，最终仍 **no-replacement，未做 kernel swap**。

---

## Goal-08 · moe_w2_prefill

| 字段 | 内容 |
|---|---|
| Session | `g56u-08-w2-prefill` |
| 目录 | `~/glm52-goal-runs/08-moe_w2_prefill/` |
| 运行状态 | ~3h 48m / ~2.54M tokens；两 worktree clean；未 push |
| 分支 (harness) | `goal/glm52-prod-08-moe_w2_prefill` @ `10190d8` |
| 分支 (sglang) | 同名分支 @ `07802235f`（opt-in PSUM 控件/测试；**默认仍 stock `{}` kwargs**） |
| 处置 | **leaf-win + no-replacement**（complete — no replacement；TP8 验收外阻） |
| 是否获得可上线优化 | **否**（stock row-wise contiguous W2 仍唯一生产路径） |
| Leaf/组件是否有实测加速 | **是**（PSUM ≈ **1.064×**） |

### 目标路径（相对 plan 的 ABI 纠正）

- Prefill + normal DeepEP：**不是** masked decode 路径
- 实际：`ep_scatter` → flat expert-major + row-wise `m_indices` → `grouped_gemm_nt_f8f8bf16_contig` → `m_grouped_fp8_gemm_nt_contiguous`
- Workload：`moe_w2_grouped_prefill_m4096`（32 local experts，K2048/N6144，FP8 + packed UE8M0，128-row alignment）
- Fixture：单 rank 的 provisional EP8 router-contract replay（32982 valid / 35200 aligned）；**非 live EP8**

### 尝试的优化

**DeepGEMM PSUM layout**（库内已有 specialization / GemmType=5，非从零手写新核）：

- 假设：用 `ep_scatter` 已算好的 cumulative endpoints，减少 row-layout 查找与 tail/epilogue 无效行工作
- 调用：`use_psum_layout=True`，`compiled_dims=nk`，expected M 1024，不做 output-gap zeroing
- 重要：PSUM **不能减少 M tile 数**（`ceil(raw_m/128)==aligned_m/128`）；收益来自 layout/tail/output，非更高 occupancy
- 变体：expected-M unset / `mnk` 与 primary 几乎同速；zero-gap 略慢（≈1.055×）

SGLang：`07802235` 暴露 keyword-only 实验控件，但生产调用仍构造 stock `{}`。

### 组件门禁结果（权威）

| 项 | 数值 |
|---|---|
| Stock / PSUM p50 | 0.336816 → **0.316400 ms** |
| Paired median | **1.064482×**（p10–p90 **1.059–1.066×**） |
| Identity 噪声 | 0.999–1.001× |
| Valid-row 正确性 | PASS（仅 `ep_gather` 消费行） |
| NCU 核时 | 331.296 → **313.312 µs（1.057×）** |

### EP4 / 外阻

- EP4：6 组 dispatch/combine identity 过（pinned DeepEP overlay）；**未跑 PSUM / 全 MoE region**；仍是 EP4 诊断
- 曾修：`6180228` — 同步 DeepEP `async_finish=False` 时错误 `current_stream_wait` 的 fail-closed 行为
- **TP8/DP8/EP8 + checkpoint**：主机四卡、无 GLM-5.2 权重 → full-region / E2E **外阻**，未削弱

### 为何仍 no-replacement

1. 组件 3% 只算 **development evidence**  
2. 生产还需：真 scatter endpoint 无 adapter 税、live EP8、graph/overlap、全 region、E2E  
3. 政策：无 bucket enable；stock contiguous W2 继续

### 关键 commits（本地）

- Harness：`10190d8`（close no-replacement）；战役相关 `83e0352` / `6180228` / `382e1f5` / `bcfc77a` 等
- SGLang：`07802235f`（opt-in grouped PSUM controls）

### 关键证据路径

- `serving_native/evidence/08_moe_w2_prefill/FINAL_REPORT.md`
- `.../final_policy.md`
- `.../external_validation_blocker.md`
- `profile/moe-w2-prefill-psum-vs-stock-20260722b/REPORT.md`

### 与 goal-07 对照

| | 07 decode W2 | 08 prefill W2 |
|---|---|---|
| 路径 | masked grouped GEMM | **contiguous** + PSUM |
| 招数 | DeepGEMM **BM16** alignment | DeepGEMM **PSUM** layout |
| Leaf 加速 | ~1.06–1.09× | ~**1.064×** |
| 上线？ | 否 | 否 |
| 主阻因 | 全局 alignment + 八卡 | endpoint 集成 + 八卡/缺权重 |

### 一句话

**PSUM 组件真实快约 6.4%**（NCU 也确认），但是 provisional 单卡 replay；缺 live EP8 / checkpoint / 八卡验收，**stock contiguous W2 未替换**。

---

## Goal-22 · dsa_flashmla_kv_production

| 字段 | 内容 |
|---|---|
| Session | `g56u-22-flashmla` |
| 目录 | `~/glm52-goal-runs/22-dsa_flashmla_kv_production/` |
| 运行状态 | pane dead；monitor 已 close；Codex 曾报 Goal achieved（~2h）后 conversation interrupted |
| 分支 (harness) | `goal/glm52-prod-22-dsa_flashmla_kv_production` @ `c7db20d`（**权威结案**）；其后有未提交 flexible-campaign harness WIP |
| 分支 (sglang) | 同名 @ `d9fb72325`（exact-ABI tests）；`?? third_party/FlashMLA-goal22/` 未入库 |
| 处置 | **NO REPLACEMENT** |
| 是否获得可上线优化 | **否** |
| 未 push | 是 |

### 目标路径

与 goal-01 / 02 / 04 同路：

```text
--dsa-decode-backend flashmla_kv
  DeepseekSparseAttnBackend._forward_flashmla_kv
  -> sgl_kernel.flash_mla.flash_mla_with_kvcache
  -> torch.ops.sgl_kernel.fwd_kvcache_mla
  -> flash_fwd_splitkv_mla_fp8_sparse_kernel*
  -> flash_fwd_mla_combine_kernel*
```

- Workload：`dsa_flashmla_kv_decode_m16` / `m32`（Q `[M,1,64,576]` BF16，paged KV FP8 `[2049,64,1,656]`，page64，2048 sparse slots）
- 实测：M16 实际 8 splits/req，M32 实际 4；stock 仍按 `num_sm_parts=148` 选 combine bound **160**

### 尝试的优化

| 候选 | 做法 | 结果 |
|---|---|---|
| FlashMLA `max_num_splits=32` combine（M16 专用 fail-closed） | 隔离重建 `d18ff63`；**未**原地改已装 package | 代码/资源如预期缩小（ptxas shared 5KiB→1KiB），但门禁不过 |

正确性：M16/M32、夹杂 −1 indices、非默认 stream、CUDA graph capture/replay/mutation — 均过。

### 门禁结果

| Mode | Bucket | session paired p50 | ≥1.03 席位 |
|---|---|---:|---:|
| eager | M16 | 1.037 / 1.016 / 1.021 | **1/3** |
| eager | M32 | 0.990 / 0.994 / 1.011 | **0/3** |
| CUDA Graph（权威） | M16 | 0.998 / 0.987 / 0.983 | **0/3** |
| CUDA Graph（权威） | M32 | 0.994 / 0.994 / 0.999 | **0/3** |

- 唯一有利的 eager M16 未复现，且在 graph 下反转
- Nsys 两核链（main+combine，PDL 重叠）：candidate M16 chain 仅约 **−1.18%**（单次 profile，不作验收）
- NCU：combine 仍是 128-block、~16.8 MB gather + long-scoreboard；降 shared 未消掉主导延迟

### 关键 commits（本地）

- harness：`c7db20d` — `perf(dsa): validate production FlashMLA KV decode path`（结案）
- SGLang：`d9fb72325` — exact FlashMLA KV ABI tests
- **WIP（不计入结案）**：dirty flexible-campaign / runner 改动；`third_party/FlashMLA-goal22/` 未跟踪

### 关键证据路径

- `kernel-harness/profile/dsa-flashmla-kv-stock-b200-20260722/REPORT.md`
- `.../analysis/paired_measurements_summary.md`
- `.../analysis/flashmla_d18ff63.patch`
- `.../analysis/external_validation_blockers.md`

### 阻塞项（未削弱门禁）

- checkpoint 目录空；主机仅 4×B200 → 完整 server / region / **TP8·DP8·EP8** 外阻
- graph 门禁已失败 → 外阻**不改变** no-replacement

### 与 goal-01 关系

同一 `flashmla_kv` + 同类「收紧 combine `MAX_SPLITS` / `max_num_splits`」假设；本任务补齐 production-ABI 可达性、隔离 rebuild、graph 权威重复，结论一致：**不替换 stock**。

### 一句话

生产 ABI 可达且正确，`max_num_splits=32` 有代码效果，但 **eager 不稳、CUDA graph ~0.98–1.00×**，全部 bucket 保持 stock FlashMLA。

---

## Goal-23 · dp_allgather_production

| 字段 | 内容 |
|---|---|
| Session | `g56u-23-allgather` |
| 目录 | `~/glm52-goal-runs/23-dp_allgather_production/` |
| 运行状态 | pane dead（exit 0）；本地收尾；未 push |
| 分支 (harness) | `goal/glm52-prod-23-dp_allgather_production` @ `fdc227a` |
| 分支 (sglang) | 同名分支 @ `f93f8867b`（**完全未改**，diff 空） |
| 处置 | **NO_REPLACEMENT** + 本地/外部调度与拓扑阻塞 |
| 是否获得可上线优化 | **否** |
| 权威测量拓扑 | 仅 **TP4/DP4 诊断**；不得晋升为 TP8 |

### 目标路径

- Decode：`dp_gather_replicate` → `_dp_gather_via_all_gather` → `GroupCoordinator.all_gather_into_tensor` → graph PyNCCL `ncclAllGather`
- ABI：BF16 `[M,6144]` → `[world*M,6144]`，caller-owned output，rank-major
- 默认 eager **prefill 不是 AllGather**：源码选 SUM_LEN → `_dp_gather_via_all_reduce`；`dp_allgather_prefill` 仅是显式 MAX_LEN 旁路

### 尝试的优化（通信调度层，无设备核改动）

| 候选 | 结果 |
|---|---|
| Direct PyNCCL identity | 噪声对照；严格 oracle **排除**（缺 topology/hash 硬化） |
| c10d torch.distributed | **源码拒绝** decode graph（SGLang 禁 capture） |
| Grouped rank broadcasts | 正确性过，但 bundle **INCOMPLETE**/污染；严格 timing **零席** |
| 强制 NCCL algo/proto | 与 auto 已选 **Ring/LL** 等同，无配对 delta |
| Symmetric/registered mem | 当前 recipe 禁用 → reject |
| 自定义/multimem kernel | ABI 不匹配 → reject；**未写新核** |

### 已承认的 TP4 基线（非候选 oracle）

| Bucket | 三 session rank-max p50 (ms) |
|---|---|
| decode M16 | 0.101 / 0.093 / 0.099 |
| decode M32 | 0.084 / 0.094 / 0.107 |

Profiler（stock M16）：`ncclDevKernel_AllGather_RING_LL`，小消息 launch/协调瓶颈，NVLink 低利用率；CUPTI 扰动 timing **不入性能表**。

### 你贴的收尾（原子调度合约）

升级后：两 TP4 launcher 必须继承 **intent-lock FD 8** + GPU-lock **FD 9–12**；CPU 回归 21 测通过。

三次 post-upgrade atomic retry（`atomic-retry{,2,3}-20260722.log`）：

- 均 **立即 exit 75**（他卡占用 / 外部 CUDA）
- **未**调用 wrapped command → 无 ranks / CUDA / 结果目录
- **不**产生性能证据，也**不**把 TP4 重标为 TP8

严格 topology×M×backend 候选 oracle：**0 合格 session → 无 enable**。

### 关键 commits（本地）

- 证据/oracle：`522a3b8` 等；原子合约：`05d0585` / `5e8a81a`
- 拒绝回执：`4df0157`、`fdc227a`（atomic-retry3）
- SGLang：无变更

### 关键证据路径

- `profile/dp-allgather-production-b200-20260722/FINAL_REPORT.md`
- `.../paired_summary.md` / `attempt_ledger.md` / `manifest.json`
- `.../scheduler/atomic-retry3-20260722.log`

### 一句话

通信侧试了一圈，**没有可晋级候选**；后半段主要在硬化四卡原子锁并记录 exit 75。Stock AllGather 继续；**TP8 验收仍外阻**。

---

## Goal-24 · tp_allreduce_reachability

| 字段 | 内容 |
|---|---|
| Session | `g56u-24-allreduce` |
| 目录 | `~/glm52-goal-runs/24-tp_allreduce_reachability/` |
| 运行状态 | **仍 active / in-progress**；pane 存活但 Codex 已触达 **usage limit**（提示 Jul 29 再试）；停在模型切换选项 |
| 分支 (harness) | `goal/glm52-prod-24-tp_allreduce_reachability` @ `bea4a8c` |
| 分支 (sglang) | 同名分支 @ `e49664113` |
| 处置 | **尚未结案**（ledger：`RUNTIME EVIDENCE IN PROGRESS`）；stock fallback 全程有效 |
| 是否获得可上线优化 | **否**（无完整 clean campaign 性能门禁结果） |
| 未 push | 是；两树 clean |

### 目标路径

- TP AllReduce reachability / backend 选择（decode M16/M32 graph + prefill eager）
- 诊断拓扑：**TP4 only**；不得重标为 TP8
- 生产门槛仍需 ≥1.03× paired + 正确性 + 后续 region/E2E（未跑到）

### 已完成的工作（基建为主，非性能结案）

| 阶段 | 结果 |
|---|---|
| P-1 CPU-barrier 对齐 | **拒**：rank-start envelope 超 500µs → 战役中止 |
| P0 公共 scheduled-start deadline | **验收为测量基建**（非 backend 结果） |
| P1 shutdown hang / long-tail | 部分战役拒；修复后待 clean 验证 |
| Graph-reset / teardown probes | 有拒收 + 修复收据 |
| `20260722T192433Z` clean campaign | 跑完 reachability / semantics / baselines / c10d ABI；在 **analyzer 合约漂移**处失败退出（exit 1） |
| Analyzer 修复 | **15/15** CPU tests；offline replay：inplace 选中、outplace 拒；**不能**复活旧战役 → 必须新 locked campaign |
| AllReduce tracing | **12/12** tests |
| 拒收产物 | 已审计归档（`PARTIAL_EVIDENCE` / receipts） |

c10d 侧（该 aborted campaign，**性能数字不入门禁**）：inplace 过 ABI；cloned outplace 因 alias/poststate 不过（预期可拒）。

### 你贴的当前卡点

- 新一轮 clean TP4 campaign：**未能启动**
- `with_all_gpus_lock.sh` 多次 **exit 75**（四卡被无关 CUDA 占用）
- **未**创建 evidence root；**未**削弱门禁；**未**动无关进程
- 下一步：额度恢复后继续 **retry clean locked campaign**

### 关键 commits（本地进度点）

- Harness：`bea4a8c`（harden TP4 paired lifecycle）等一串 serving_native/timing 硬化
- SGLang：`e49664113`（analyzer 与 retry receipts 对齐）等

### 关键证据路径

- `sglang/glm52_opt/history/tp_allreduce_reachability/attempt_ledger.md`
- `.../runtime/tp_allreduce_reachability_20260722T192433Z_aborted_analyzer_contract/`（`PARTIAL_EVIDENCE.md`、`ANALYZER_FIX_RECEIPT.json`）
- 多份 `*_aborted_*` / probe 目录（对齐、shutdown、graph-reset）

### 一句话

**还在路上**：测量基建与 analyzer 已修好并 commit，但权威 TP4 性能战役因 **analyzer 拒收旧跑 + 新跑抢不到四卡（exit 75）+ Codex 额度** 未完成；**无优化上线，stock 继续，goal 仍 active**。

---

## Goal-25 · deepep_dispatch_combine

| 字段 | 内容 |
|---|---|
| Session | `g56u-25-deepep` |
| 目录 | `~/glm52-goal-runs/25-deepep_dispatch_combine/` |
| 运行状态 | Goal achieved（~8h / ~4.29M tokens）；后续有调度升级提示与 usage limit，但主结论已 commit |
| 分支 (harness) | `goal/glm52-prod-25-deepep_dispatch_combine` @ `ab78c4c` |
| 分支 (sglang) | 同名分支 @ `f93f8867b`（**未改**） |
| 处置 | **No replacement**（无生产替换） |
| 是否获得可上线优化 | **否**（`SGLANG_GLM52_OPT=0`，无 `--deepep-config`） |
| EP4 诊断加速？ | **有**：joint normal Config ≈ **1.36×**（仅诊断，不可晋升 EP8） |
| 未 push | 是 |

### 目标路径

- `DeepseekV2MoE.forward_deepep` → DeepEP dispatcher；AUTO：decode=low-latency，prefill=normal
- 含区：`dispatch → W13 → SwiGLU+quant → W2 → combine`
- 本地测量：**EP4**（4×B200，64 local experts；decode M16/M32；prefill M8192）
- 生产目标：**TP8/DP8/EP8**（32 local experts；prefill M4096）——保持分离，禁止重标

### 尝试的优化

1. **Source：compact low-latency geometry**（改 DeepEP overlay，EP4-guarded）  
   - dispatch/combine：更小 CTA/线程（及 combine 降 shared）  
   - 正确性/hash 过，但 **每个 bucket 至少一对回退** → reject  
   - 例：dispatch M16 region medians 有 0.945 / 3.13 / 0.266 → 门禁 fail

2. **Joint normal Config**（仍用 **stock binary**，只换 Config）  
   - dispatch `(24,32,256,6,128)` + combine `(24,16,256,6,128)`，匹配 24-SM/12-channel handle  
   - 六对 full-region：median **1.361×**，min **1.319×**，无回退，rank-wise hash 一致  
   - 核时：dispatch ~1.70→0.73 ms；combine ~1.84→1.08 ms  
   - **政策：EP4 frontier 记录，不 enable、不抄到 EP8**

### Stock 基线（EP4，rank-max p50）

| Op | M | p50 (ms) |
|---|---:|---:|
| LL dispatch / combine | 16 | 0.093 / 0.049 |
| LL dispatch / combine | 32 | 0.077 / 0.052 |
| normal dispatch / combine | 8192 | 1.911 / 1.743 |

### 为何仍 no-replacement

1. 源码候选在 full-region **不过门禁**  
2. Config 虽 EP4 过 3%，但缺 **EP8 / CUDA Graph / overlap / checkpoint / server**  
3. 源码 overlay **rank-4 guarded**，EP8 会 fallthrough stock，不能当 EP8 候选二进制  
4. LL **IBGDA init failed**；实际 fallback transport **未观测**，不作宣称  

### 关键 commits / 身份

- Harness：`ab78c4c`（finalize joint Config disposition）  
- SGLang：`f93f8867b` 不变；DeepEP stock ext SHA `3e857a1c…abedb0`

### 关键证据路径

- `serving_native/evidence/25_deepep_dispatch_combine/final_report.md`
- `.../oracle_fallback_policy.md`
- `.../summaries/glm52_goal25_joint_config_summary_20260722_d.md`
- `.../ep8_config_external_acceptance.md` / `ep8_external_acceptance.md`
- `.../SHA256SUMS` / `artifact_index.md`

### 一句话

**源码改 DeepEP 几何失败；调 Config 在 EP4 上能稳快约 36%，但只作诊断。** 生产仍 stock，EP8/graph/overlap/checkpoint/server 门禁未削弱。

---

## 专节模板（后续任务复制）

```markdown
## Goal-XX · <slug>

| 字段 | 内容 |
|---|---|
| Session | `g56u-XX-...` |
| 目录 | `~/glm52-goal-runs/XX-.../` |
| 运行状态 | |
| 分支 (harness) | |
| 处置 | win / no-replacement / blocked / ... |
| 是否获得可上线优化 | 是 / 否 |
| 未 push | |

### 目标路径
### 尝试的优化
### 门禁结果
### 关键 commits
### 关键证据路径
### 阻塞项
### 一句话
```
