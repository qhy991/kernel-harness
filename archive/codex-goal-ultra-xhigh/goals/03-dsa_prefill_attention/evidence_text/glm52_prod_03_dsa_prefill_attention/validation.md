# Validation log

Date: 2026-07-22. Commands ran only from the isolated Kernel-Harness and SGLang
worktrees. Every decision-bearing CUDA series used one scheduler wrapper
allocation for its complete alternating series and profiler collection. No
delegated task ran GPU work.

## Final structural checks

| Check | Result |
|---|---|
| `python3 testbench/bin/brief.py dsa_prefill_attn` | completed before source work; inherited recipe was audited rather than trusted blindly |
| `env SGLANG_ROOT=<isolated-sglang> .venv/bin/python serving_native/selftest.py` | passed: 45 fixed workloads |
| `python3 testbench/bin/selftest.py` | passed: 24 tasks, 0 problems |
| `python3 -m py_compile ...` for all modified/new serving and profile Python drivers | passed |
| `bash -n ...` for all four new bundle scripts | passed |
| `python3 testbench/bin/knowledge.py add ...knowledge_entry_superseding_draft.json` | passed; installed append-only entry `glm52--dsa_prefill_attn--b200--20260722b` |
| knowledge `lint`, `index --check`, and `distill --check` | passed: 14 entries, 0 problems; all generated views current |
| `/home/qinhaiyan/glm52-goal-runs/with_flexible_gpu.sh -- python3 testbench/bin/verify_harness.py` | passed on scheduler-selected physical GPU 2, UUID `GPU-df8b1d78-b06c-39a2-54f0-66b9fabf3a99`; 24 task dirs in sync |
| verifier audit/pointer report | audit sweep clean with no run corpus; missing historical `runs/index.jsonl` reported as advisory pointer drift, verifier exit 0 |
| source/documentation whitespace checks | passed; immutable raw torchrun logs retain four upstream lines with trailing spaces and were excluded from the cached source check |
| isolated SGLang status and backend byte comparison | clean at rollback head; reached backend byte-identical to stock base |

The first CPU-only verifier invocation correctly reported stale generated
knowledge views immediately after the append-only entry was installed, and it
could not project CUDA-dependent task tensor tables. After regenerating only the
knowledge views, the final verifier was rerun through the required flexible-GPU
wrapper and passed without modifying generated task files.

## Decision-bearing GPU bundles

| Bundle | Scheduler allocation | Result |
|---|---|---|
| exact 513-page raw-pool leaf: 3x stock control, 3x Q32, 3x Q16, NSYS and full NCU | one `with_flexible_gpu.sh` allocation; physical GPU UUID `GPU-5b9be10b-5bfc-b658-9b31-f7ae8516dc54` | all correctness comparisons passed; Q32/Q16 rejected |
| checkpoint-free backend class: hit trace, 3x stock control, NSYS | one `with_flexible_gpu.sh` allocation; physical GPU UUID `GPU-30b619de-87f2-1862-0d07-a595da8fe417` | exact leaf hit and finite output passed; identity control neutral |
| four-rank raw-pool diagnostic: 3x rank-max stock control and NSYS | one `with_all_gpus_lock.sh` allocation over the four host B200s | all ranks completed; diagnostic neutral; not TP8 acceptance |

The exact raw-pool comparisons contain 90 paired samples per tactic (30 pairs
in each of three runs). Backend and DP4 controls contain 60 paired samples each.
Every persisted serving result was written only after the runner's mandatory
structure/shape comparison and FP32-cast `allclose` check. Finite BF16 output is
a separate assertion in the backend hit trace, not a generic runner guarantee.

The earlier compact-pool and PDL artifacts are retained. Their original scripts
predate the current scheduler rule and sometimes record a directly selected
physical device. They are corroborating historical evidence only; the final
source decision uses the newly bundled wrapper-compliant raw-pool series. The
PDL timing files measure an external leaf candidate, not the temporary SGLang
dispatch, as detailed in `AUDIT_CORRECTIONS.md`.

After the DP4 bundle completed, its scheduler precondition was generalized from
a literal device list to "exactly four scheduler-selected visible GPUs" so the
replay script does not reserve physical identifiers. The measured workload,
commands, timing body, and retained artifacts are unchanged.

## Scope of correctness and audit

- Exact raw-pool Q32 and Q16: passed the scattered causal distribution.
- Compact replay Q32 and Q16: passed the trailing causal distribution.
- Nonmatching eight-request compact workload: passed through stock fallback.
- Backend fixture: exactly one TRTLLM-gen leaf hit, exact dtype/shape metadata,
  finite BF16 output, eager stream behavior recorded.
- DP4: all four independent rank-local leaves passed and timing used rank max.

Serving-native JSON has a separate schema from the frozen task's `result.json`,
so `testbench/bin/audit_result.py` is not applicable and no synthetic official
gate result is claimed. A real checkpoint-backed request, full live DSA region,
SGLang end-to-end prefill, and TP8/DP8/EP8 remain explicitly unpassed.
