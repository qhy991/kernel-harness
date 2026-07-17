# Agent integration guide

How to tell whether a harness WIN is safe to ship into the real SGLang inference path.

## Three integration contracts

| Contract | Meaning | Agent action on WIN |
|---|---|---|
| `drop-in` | `integrate.py` patches a real sglang dispatch symbol and drives production forward | **Must** run `integrate.py`; record `integrate=pass` in knowledge |
| `fused-only` | No isolated symbol (fused backend path) | **Do not** fake integrate; record `integrate=no-recipe` + interface notes |
| `unsupported` | No recipe yet | Record `integrate=not-run`; treat as harness-only until recipe lands |

List contracts / pick high-leverage tasks before starting:

```bash
python3 testbench/bin/integration_status.py <model>
```

## GLM-5.2 does not use this document

This whole page describes the `solution.py` + `integrate.py` drop-in contract, which
GLM-5.2's 24 tasks do not follow — they have no `solution.py` and no integrate
recipe, and `integration_status.py` reports them `unsupported`. For GLM-5.2 read the
"different contract" section of [`AGENTS.md`](../../AGENTS.md) and ask the task
itself:

```bash
./testbench/tasks/glm52/o_proj_decode/run.sh --describe
```

## GLM-5.2 batch — integration truth table (HISTORICAL, tasks removed)

Kept as a record of the pre-consolidation runs. **Five of these six task directories
no longer exist**, and the sixth (`o_proj_decode`) now has a different contract,
different inputs and a different baseline — so none of these verdicts can be
reproduced or compared against a current run. `routed_down_decode` never existed in
this tree at all; the row below was already wrong when it was written.

| Task | evaluate | integrate contract | Production readiness | Status now |
|---|---|---|---|---|
| `routed_swiglu_prefill` | **WIN** | `drop-in` | DROP-IN VERIFIED (Float8 compare fixed) | removed (not one of the 12 ops) |
| `routed_down_decode` | **WIN** | `drop-in` | DROP-IN VERIFIED (active expert rows only) | never existed |
| `sparse_mla_decode` | **WIN** | `fused-only` | Interface-exact TRT-LLM; `integrate=no-recipe` | removed (not one of the 12 ops) |
| `routed_swiglu_decode` | no-win | `drop-in` | Production C++ kernel near floor | removed (not one of the 12 ops) |
| `o_proj_decode` | no-win | `drop-in` | DeepGEMM wrapper already optimal at small M | rebuilt on the glm52_ops contract |
| `routed_gateup_nvfp4_decode` | prior sessions | `drop-in` (nvfp4-moe) | FlashInfer TRT-LLM symbol has integrate recipe | removed (not one of the 12 ops) |

## Session closeout (recommended)

```bash
.venv/bin/python testbench/bin/agent_closeout.py glm52/<task> --repeat 3 \
  --owner qinhaiyan --record-tokens
```

Parses `VERDICT_JSON`, runs `integrate.py` on WIN when contract is `drop-in`, emits
`CLOSEOUT_JSON`, and optionally appends a row under `token-records/<owner>/`.

`evaluate.py` also prints an `INTEGRATION_CONTRACT:` line after every verdict.

## Common integrate failure modes (and fixes)

1. **Float8 `isinf` NotImplemented** — fixed: promote to float32 before finite checks.
2. **Grouped-MoE empty experts** — fixed: compare only rows where `masked_m[e] > 0`.
3. **NVFP4 MoE unsupported** — fixed: `family=nvfp4-moe` patches
   `flashinfer.trtllm_fp4_block_scale_routed_moe` (+ `fused_moe` alias).
4. **Fused-only families** — `sparse-mla-decode`, DSA fused attention. Do not invent a wrapper.

## Agent optimization priorities

1. Query contract + headroom first (`integration_status.py` / `inventory.py --headroom`).
2. Use glm52 `workload_metrics` / `performance_model` (advisory).
3. Record `result.integrate` honestly in knowledge (`pass|fail|no-recipe|not-run`).
4. Prefer **high** headroom families over small-M FP8 DeepGEMM for first wins.
5. Ledger tokens with `agent_closeout.py --record-tokens --owner <name>`.
