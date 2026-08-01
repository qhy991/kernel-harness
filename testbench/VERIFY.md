# Harness Review Verification

Use this file as the review handoff for harness-only changes. Commands run from
the repository root.

## Normal Review Lane

```bash
python3 testbench/bin/verify_harness.py
python3 testbench/bin/verify_harness.py --json
git diff --check -- AGENTS.md testbench/README.md testbench/VERIFY.md \
  testbench/setup_env.sh \
  testbench/bin testbench/harness testbench/knowledge testbench/tasks/glm52 \
  rewardbench serving_native \
  ':(exclude)testbench/tasks/glm52/*/candidate.py'
```

Expected current corpus summary:

- `selftest: 36 tasks, 0 problems`
- `knowledge lint: 13 entries, 0 problems`
- `index --check: 0 stale`
- `distill --check: up to date`
- `36 task dirs are in sync with glm52_ops`
- `serving_native selftest OK: 39 fixed workloads; production_source=checked`
- audit sweep: `audited=282 invalid=0 official=0 provisional=282`
- pointer audit: `index_rows=283 latest_files=27 stale_index=1 stale_latest=0 malformed=0 mismatched=0`

The stale pointer is historical and advisory in the normal lane:

```text
runs/glm52/moe_down_proj_decode/20260718T022352Z-3635ef/result.json
```

The 12 additive fusion-region tasks are covered by the same review lane. The
preserved Gate-prefill candidate's current complete B200 result is
`runs/glm52/norm_quant_gate_prefill/20260801T032655Z-bdb223/result.json`: 3/3
shapes won, 0 regressed, `calc_diff=0`. It is provisional because the integration
tree and candidate were uncommitted during measurement. QKV deliberately has a
three-output ABI; the saved two-output Gate kernel is rejected there.

The eight B300-derived Q/K-indexer, masked-SwiGLU/quant, and router/top-k tasks
passed all 20 canonical B200 shapes in low-repeat correctness sweeps, including
post-timing unseen-seed checks. The decode `router_jit_gemm` variant also passed a
complete default-protocol gate at
`runs/glm52/router_gemm_topk_decode/20260801T053943Z-4007d5/result.json`: M=16
2.721x median / 2.712x conservative, exact M=32 fallback neutral, `calc_diff=0`,
no regressions, stable pair spread. It remains provisional because the task and
candidate were uncommitted during measurement and the shared tree was dirty.

Schema 1.4 results additionally prove paired timing rows, distinct post-timing
seeds, minimum warmup/repeat/inner iterations, stability/no-verdict behavior, and
the physical reward limit. Schema 1.3 and older runs remain provisional rather
than being retroactively invalidated; unstable legacy wins and repeat<10 wins are
reported as caveats. The production leaf lane is separately exercised with:

```bash
serving_native/run.sh <workload> --candidate PATH --execution-mode both
```

Exit 0 there means eager+CUDA-graph leaf evidence only. The structured result still
requires containing-region and end-to-end SGLang confirmation.

## Strict Evidence Lanes

These are expected to fail on the current historical corpus until old evidence is
promoted or repaired:

```bash
python3 testbench/bin/verify_harness.py --strict-audit-sweep
python3 testbench/bin/verify_harness.py --strict-pointer-audit
python3 testbench/bin/verify_harness.py --audit-report
python3 testbench/bin/verify_harness.py --pointer-report
```

Use strict lanes for release notes, official evidence claims, and CI jobs that
must reject provisional or stale historical evidence. `--audit-report` lists every
non-official historical result with its warnings/errors; `--pointer-report` lists
every stale, malformed, or mismatched pointer. Both reports are read-only.

## Review Bucket

List the harness-reviewable files with:

```bash
python3 testbench/bin/verify_harness.py --print-review-files --with-status
```

Stage only that bucket with:

```bash
python3 testbench/bin/verify_harness.py --print-review-files -0 | xargs -0 git add --
```

The bucket intentionally excludes `archive/` and `testbench/tasks/glm52/*/candidate.py`.
