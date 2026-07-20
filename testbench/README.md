# testbench — the GLM-5.2 task suite

24 tasks: 12 operators × 2 phases, B200 / DP1 / TP1 / EP32 (`zai-org/GLM-5.2-FP8`).

```
testbench/
  harness/
    glm52_ops.py        the ONLY definition of the 12 operators: frozen inputs,
                        reference, tolerances, masks, cost model, peaks, and the
                        generated problem statement
    evaluate_task.py    the runner — orchestrates only; defines nothing
    candidate_loader.py resolves candidate.py / --candidate PATH to run(inputs)
    result_store.py     run ids, environment capture, append-only persistence
    timing.py           CUPTI cold-L2 device-kernel timer
    reward_hack.py      anti-cheat (monkey-patched timers, lazy outputs)
  bin/
    sync_glm52_tasks.py projects glm52_ops onto the 24 task dirs (--check for CI;
                        never overwrites candidate.py)
    selftest.py         GPU-free structural pre-flight
    accept_layer.py     acceptance only: swap a candidate into the 12-op layer
                        budget (PR1 allLatency / llm_flops style) and report the
                        end-to-end delta — does NOT gate a WIN
    inventory.py        list tasks by family
    knowledge.py        the recipe log
    check_env.py        verify GPU / CUDA / torch / deep_gemm / sgl_kernel
  tasks/glm52/<task>/   task.json · problem.json · workload.jsonl · candidate.py ·
                        run.sh · README.md
  knowledge/            append-only session recipes
  docs/GLM52_CANDIDATES.md   worked Triton and CUDA .cu candidates, measured
```

The agent-facing guide is [`../AGENTS.md`](../AGENTS.md); a task describes itself with
`run.sh --describe`.

## The five operator families

| family | ops | backend |
|---|---|---|
| gemm | `fused_qkv_a` `q_b` `o_proj` `index_q_upproj` `index_k` | `deep_gemm.fp8_gemm_nt` |
| bmm | `absorbed_W_UK` `absorbed_W_UV` | `sgl_kernel.bmm_fp8` |
| moe | `moe_gate` `moe_up` `moe_down` | `deep_gemm.fp8_m_grouped_gemm_nt_masked` |
| mla | `dsa_attn` | `flash_mla_sparse_fwd` |
| score | `index_score` | `fp8_mqa_logits` / `fp8_paged_mqa_logits` (differs by phase) |

Sweeps: prefill M∈{1024,2048,4096}, decode M∈{16,32}. `index_k` prefill is driven by
S=65536, not M, so its three prefill shapes are one GEMM. `index_score` runs a
different kernel per phase. Every decode shape is memory- or launch-bound and most
prefill shapes are compute-bound — the same operator with the opposite bottleneck,
which is why phases are separate tasks.

## Provenance

The definitions were merged from opbench (PR1) and rewardbench (PR2) after an
op-by-op comparison, taking the correct side of each disagreement (PR2's MoE capacity
guard, PR1's indexer q_scale fold) and PR2's cost model, verified bit-exact. The
reference is deep_gemm's f32-blockwise-scale path, which is ~1.6x slower than SGLang's
production int32-ue8m0 dispatch — `run.sh --describe` says so per task. Details are in
`glm52_ops.py`'s module docstring.

## Retired

The Kimi-K2.7 / MiniMax-M3 suite, its `solution.py` + `definition.json` contract, and
the tooling that served it (`evaluate.py`, `integrate.py`, `migrate.py`, `gen_tasks.py`,
`taskgen/`, `recipes/`, the provider A/B benchmarks) live under
[`../legacy/`](../legacy/README.md), along with that version of this README.
