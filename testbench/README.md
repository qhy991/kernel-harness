# testbench — the GLM-5.2 task suite

36 tasks: 12 leaf operators + 6 production fusion regions, each in prefill and
decode, on B200 / DP1 / TP1 / EP32 (`zai-org/GLM-5.2-FP8`).

```
testbench/
  harness/
    glm52_ops.py        the ONLY definition of the 12 leaf ops + 6 fusion regions:
                        frozen inputs,
                        reference, tolerances, masks, cost model, peaks, and the
                        generated problem statement
    evaluate_task.py    the runner — orchestrates only; defines nothing
    candidate_loader.py resolves candidate.py / --candidate PATH to run(inputs)
    result_store.py     run ids, environment capture, append-only persistence
    timing.py           CUPTI cold-L2 device-kernel timer; first sample discarded
    paired_stats.py     order-balanced pair ratios, quantiles, stability detector
    reward_hack.py      anti-cheat (monkey-patched timers, lazy outputs)
  bin/
    sync_glm52_tasks.py projects glm52_ops onto the 36 task dirs (--check for CI;
                        never overwrites candidate.py)
    selftest.py         GPU-free structural pre-flight
    accept_layer.py     acceptance only: swap a candidate into the 12-op layer
                        budget (PR1 allLatency / llm_flops style) and report the
                        end-to-end delta — does NOT gate a WIN
    inventory.py        list tasks by family
    knowledge.py        the recipe log
    check_env.py        verify GPU / CUDA / torch / deep_gemm / sgl_kernel
    verify_harness.py   GPU-free review/CI gate: compile, selftest, knowledge
                        freshness, generated-task sync, audit sweep, pointer
                        audit, diff hygiene
    supervise.py        process-group timeout/reaper for task and torchrun children
  tasks/glm52/<task>/   task.json · problem.json · workload.jsonl · candidate.py ·
                        run.sh · README.md
  knowledge/            append-only session entries + atomic experiment recipes
  docs/GLM52_CANDIDATES.md   worked Triton and CUDA .cu candidates, measured
```

The agent-facing guide is [`../AGENTS.md`](../AGENTS.md); a task describes itself with
`run.sh --describe`.

The 24 leaf tasks are synthetic numerical/leaf-kernel oracles whose f32-scale
DeepGEMM reference intentionally differs from SGLang's packed UE8M0 production
path. The 12 fusion tasks instead gate production regions using the actual fused
kernel or production-order sequence and every required side output. Use
[`../serving_native/`](../serving_native/README.md) with
`--execution-mode both` for the production ABI eager+CUDA-graph comparison, then
confirm the containing region and end-to-end serving workload before promotion.

Gate-eligible leaf timing uses at least 10 adjacent balanced `R/C,C/R` pairs,
warmup 8, and p10/p90 of per-pair ratios. A spread above 1.25× retries once at 3×
samples; continued instability is a no-verdict. Restricted sweeps and low-repeat
runs are probes, post-timing correctness uses a different seed, and a roofline
reward above 1.0 is a hard invalid measurement.

Every `run.sh` is supervised with a 30-minute wall timeout and process-group
cleanup. Override with `KERNEL_HARNESS_TIMEOUT_SECONDS=<seconds>`; use `0` only for
an intentionally unsupervised diagnostic.

Review harness changes with one GPU-free command:

```bash
python3 testbench/bin/verify_harness.py
python3 testbench/bin/verify_harness.py --json        # CI/agent-readable summary
python3 testbench/bin/verify_harness.py --strict-audit-sweep
python3 testbench/bin/verify_harness.py --strict-pointer-audit
python3 testbench/bin/verify_harness.py --audit-report
python3 testbench/bin/verify_harness.py --pointer-report
python3 testbench/bin/verify_harness.py --skip-audit-sweep
python3 testbench/bin/verify_harness.py --skip-pointer-audit
python3 testbench/bin/verify_harness.py --print-review-paths
python3 testbench/bin/verify_harness.py --print-review-files
python3 testbench/bin/verify_harness.py --print-review-files --with-status
python3 testbench/bin/verify_harness.py --print-review-files -0
```

## The six operator families

| family | ops | backend |
|---|---|---|
| gemm | `fused_qkv_a` `q_b` `o_proj` `index_q_upproj` `index_k` | `deep_gemm.fp8_gemm_nt` |
| bmm | `absorbed_W_UK` `absorbed_W_UV` | `sgl_kernel.bmm_fp8` |
| moe | `moe_gate` `moe_up` `moe_down` | `deep_gemm.fp8_m_grouped_gemm_nt_masked` |
| mla | `dsa_attn` | `flash_mla_sparse_fwd` |
| score | `index_score` | `fp8_mqa_logits` / `fp8_paged_mqa_logits` (differs by phase) |
| fusion | `norm_quant_qkv` `norm_quant_gate` | `fused_add_rmsnorm` → packed-UE8M0 quant → `fp8_gemm_nt` |
| fusion | `indexer_q_rope_quant` | fused Q RoPE → FP8 quant → head-gate scale |
| fusion | `indexer_k_norm_rope_store` | fused K LayerNorm → RoPE → paged-cache store |
| fusion | `moe_swiglu_quant` | masked SwiGLU → packed-UE8M0 quant |
| fusion | `router_gemm_topk` | FP32 router projection → sigmoid/correction/top-k |

Sweeps: prefill M∈{1024,2048,4096}, decode M∈{16,32}. `index_k` prefill is driven by
S=65536, not M, so its three prefill shapes are one GEMM. `index_score` runs a
different kernel per phase. Every decode shape is memory- or launch-bound and most
prefill shapes are compute-bound — the same operator with the opposite bottleneck,
which is why phases are separate tasks. The QKV fusion contract additionally gates
the normalized BF16 tensor consumed by the DSA indexer; the gate fusion contract has
no such side consumer. The Q/K indexer tasks use an existing KV context of S=65536:
prefill appends positions `[S,S+M)`, decode treats M as batch and uses position S for
each request, and K writes the assigned rows of a real page-size-64 cache. Thus these
tasks exercise cached incremental prefill/decode semantics rather than prefix-zero
full prefill, although serving-native and end-to-end confirmation are still required.

## Provenance

The definitions were merged from opbench (PR1) and rewardbench (PR2) after an
op-by-op comparison, taking the correct side of each disagreement (PR2's MoE capacity
guard, PR1's indexer q_scale fold) and PR2's cost model, verified bit-exact. The
leaf reference is deep_gemm's f32-blockwise-scale path, which is ~1.6x slower than
SGLang's production int32-ue8m0 dispatch. Fusion-region references use production
packed scales and do not inherit that null line. `run.sh --describe` states the
applicable caveat per task; details are in `glm52_ops.py`'s module docstring.

## Retired

The Kimi-K2.7 / MiniMax-M3 suite, its `solution.py` + `definition.json` contract, and
the tooling that served it (`evaluate.py`, `integrate.py`, `migrate.py`, `gen_tasks.py`,
`taskgen/`, `recipes/`, the provider A/B benchmarks) live under
[`../legacy/`](../legacy/README.md), along with that version of this README.
