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
  ':(exclude)testbench/tasks/glm52/*/candidate.py'
```

Expected current corpus summary:

- `selftest: 24 tasks, 0 problems`
- `knowledge lint: 12 entries, 0 problems`
- `index --check: 0 stale`
- `distill --check: up to date`
- `24 task dirs are in sync with glm52_ops`
- audit sweep: `audited=273 invalid=0 official=0 provisional=273`
- pointer audit: `index_rows=274 latest_files=22 stale_index=1 stale_latest=0 malformed=0 mismatched=0`

The stale pointer is historical and advisory in the normal lane:

```text
runs/glm52/moe_down_proj_decode/20260718T022352Z-3635ef/result.json
```

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
