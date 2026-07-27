# Validation log

## CPU-only structural gates

- `python3 testbench/bin/selftest.py`: PASS, 24 tasks and zero problems.
- `SGLANG_ROOT=<isolated-sglang> .venv/bin/python serving_native/selftest.py`:
  PASS, 52 fixed workloads after adding the mixed containing-region scopes.
- `python3 -m py_compile` for the new workload, candidate, runner, and
  evidence drivers: PASS.
- `bash -n` for the single-GPU, mixed-confirmation, and four-GPU scripts:
  PASS.
- `git diff --check`: PASS.
- First campaign status: 72 recorded commands, all exit 0.
- First campaign and profiler SHA-256 manifests: PASS after verification.
- Mixed confirmation status: 32 recorded commands, all exit 0.
- Mixed confirmation campaign and profiler SHA-256 manifests: PASS after
  verification.
- Knowledge validation: 13 entries, zero lint problems; generated indices
  and distillation are current.

## Repository-wide verifier

`python3 testbench/bin/verify_harness.py` exits 1 at the pre-existing generated
task projection check. Its independent checks pass:

- harness Python compilation;
- 24-task structural selftest;
- knowledge lint (13 entries);
- knowledge index freshness;
- knowledge distillation freshness;
- scoped `git diff --check`;
- result audit sweep (zero invalid).

The failure is `sync_glm52_tasks.py --check`: without CUDA it omits generated
tensor tables and reports all 24 forbidden generated `problem.json`/README
pairs stale (48 files). The pointer audit also reports the pre-existing
missing `runs/index.jsonl`. This goal did not edit or regenerate any forbidden
task metadata, oracle, timing, or pointer file.

## Result-audit applicability

No frozen synthetic `result.json` is used or cited by this production goal.
The measurements are serving-native paired JSON files with their own raw
sample arrays, runtime metadata, source snapshots, and checksum manifests;
`audit_result.py` is therefore not applicable to them. The frozen
`index_score_prefill` task remains untouched.

## GPU gates

The initial exact baseline/source-attempt campaign and all profilers completed
under one flexible-GPU lease. The mixed-context confirmation then completed
all alternating score/complete/graph/DSA series, fallback controls, and Nsys
captures under a second single wrapper lease on the same physical GPU UUID.
All correctness checks passed. Only the score-only mixed bucket met the
pooled 3% threshold; the complete indexer, exact graph split, and selected
TRT-LLM DSA regions did not.

Four-GPU requests initially returned exit 75 for active or queued CUDA work.
No wrapper was bypassed. The first acquired run (`20260723T144048Z`) recorded
one failed series whose local correctness exception was masked by a
600-second NCCL timeout; series 2 was manually interrupted once the repeated
divergence was established, and its four exact worker PIDs were terminated.
No artifact was deleted, and that superseded directory has a verified
checksum manifest.

The distributed runner was then hardened only outside the timed interval:
each barrier names the logical local device and untimed correctness errors are
reduced across ranks before measurement. The fresh all-four-GPU run
(`20260723T145846Z`) completed three series without timeout. Each exited 1
before timing because the broad candidate changed the row-wise top-k set on
rank 1; ranks 1 and 3 failed in the last two series. Reference correctness
completed on every rank. The directory status, logs, source snapshot, and
checksum manifest verify; no four-rank latency is claimed.

CUDA-dependent repository-verifier retries were correctly denied by the
flexible wrapper while other four-GPU requests held priority. The CPU-only
verifier result above remains the recorded result.
