# Serving-native V2 CPU P0 report

Status: **PASS**

Captured at `2026-07-28T16:27:31+00:00`. No CUDA API, GPU probe, or
scheduler wrapper was invoked by this gate. The GPU-derived generated-task
projection is intentionally deferred to the fresh scheduler-wrapped smoke
script.

Source identity at the gate:

```text
$ git rev-parse HEAD
e614f325e5465fd481ae18d9bc80cf7c49f88ddb
$ git branch --show-current
goal/glm52-v2-00-serving-native-contract
```

The tested implementation was an uncommitted working tree at that base. Exact
tested source hashes:

```text
$ sha256sum serving_native/audit_result.py serving_native/contract_v2.py serving_native/runner.py serving_native/test_contract_v2.py serving_native/workloads.py testbench/bin/verify_harness.py
f434603384e0b2ff5a40a41788102d3e8166318dec13c9f4a0f6cc0f2bc873e9  serving_native/audit_result.py
ef30d34423019c381e6735cd7c81b60339129cc2cacd2f1b7d45c9e16463f962  serving_native/contract_v2.py
892889d3cda5bcae44cd1c032c991e7a20ec29308ba3c7b260db43e29a4ece75  serving_native/runner.py
cd4ad5066f297e7cd3f61255b156d750e01e859fc280269637ae8f08c117a256  serving_native/test_contract_v2.py
4f5ce503c21d4745e7faf674cd4706dfc5b944049470bfb14c3a22c0703b0bd3  serving_native/workloads.py
f47faf6c2577002acf6ff2d8dd6c430676d0ccb02c52fb975b26c1f9d8f57609  testbench/bin/verify_harness.py
```

## Adversarial auditor suite

```text
$ python3 -m unittest -v serving_native.test_contract_v2
test_call_totals_and_by_phase_counts_close_exactly (serving_native.test_contract_v2.ContractV2AuditTest.test_call_totals_and_by_phase_counts_close_exactly) ... ok
test_candidate_cannot_self_authorize_reference_delegation (serving_native.test_contract_v2.ContractV2AuditTest.test_candidate_cannot_self_authorize_reference_delegation) ... ok
test_candidate_hit_zero_is_rejected (serving_native.test_contract_v2.ContractV2AuditTest.test_candidate_hit_zero_is_rejected) ... ok
test_candidate_loader_preserves_sys_modules_provenance (serving_native.test_contract_v2.ContractV2AuditTest.test_candidate_loader_preserves_sys_modules_provenance) ... ok
test_candidate_result_path_must_match_hashed_artifact (serving_native.test_contract_v2.ContractV2AuditTest.test_candidate_result_path_must_match_hashed_artifact) ... ok
test_complete_eager_identity_artifact_is_valid_non_win (serving_native.test_contract_v2.ContractV2AuditTest.test_complete_eager_identity_artifact_is_valid_non_win) ... ok
test_complete_graph_identity_artifact_is_valid_non_win (serving_native.test_contract_v2.ContractV2AuditTest.test_complete_graph_identity_artifact_is_valid_non_win) ... ok
test_eager_kernel_identities_are_recomputed_from_events (serving_native.test_contract_v2.ContractV2AuditTest.test_eager_kernel_identities_are_recomputed_from_events) ... ok
test_execution_mode_mismatch_is_rejected (serving_native.test_contract_v2.ContractV2AuditTest.test_execution_mode_mismatch_is_rejected) ... ok
test_graph_capture_ids_are_bound_to_independent_round_robin_captures (serving_native.test_contract_v2.ContractV2AuditTest.test_graph_capture_ids_are_bound_to_independent_round_robin_captures) ... ok
test_graph_copy_and_adapter_nodes_are_rejected_after_recomputation (serving_native.test_contract_v2.ContractV2AuditTest.test_graph_copy_and_adapter_nodes_are_rejected_after_recomputation) ... ok
test_graph_metadata_is_recomputed_from_nodes_and_ids (serving_native.test_contract_v2.ContractV2AuditTest.test_graph_metadata_is_recomputed_from_nodes_and_ids) ... ok
test_graph_semantic_failures_are_rejected (serving_native.test_contract_v2.ContractV2AuditTest.test_graph_semantic_failures_are_rejected) ... ok
test_identity_cannot_claim_a_performance_pass (serving_native.test_contract_v2.ContractV2AuditTest.test_identity_cannot_claim_a_performance_pass) ... ok
test_jit_during_timing_is_rejected (serving_native.test_contract_v2.ContractV2AuditTest.test_jit_during_timing_is_rejected) ... ok
test_missing_correctness_and_provenance_are_rejected (serving_native.test_contract_v2.ContractV2AuditTest.test_missing_correctness_and_provenance_are_rejected) ... ok
test_non_filesystem_pseudo_module_is_not_recorded (serving_native.test_contract_v2.ContractV2AuditTest.test_non_filesystem_pseudo_module_is_not_recorded) ... ok
test_noncanonical_artifact_path_is_rejected (serving_native.test_contract_v2.ContractV2AuditTest.test_noncanonical_artifact_path_is_rejected) ... ok
test_old_candidate_controlled_delegation_escape_cannot_claim_win (serving_native.test_contract_v2.ContractV2AuditTest.test_old_candidate_controlled_delegation_escape_cannot_claim_win) ... ok
test_raw_ordering_and_summary_are_fail_closed (serving_native.test_contract_v2.ContractV2AuditTest.test_raw_ordering_and_summary_are_fail_closed) ... ok
test_runner_owned_config_api_can_carry_a_valid_win (serving_native.test_contract_v2.ContractV2AuditTest.test_runner_owned_config_api_can_carry_a_valid_win) ... ok
test_runner_owned_config_candidate_is_declarative (serving_native.test_contract_v2.ContractV2AuditTest.test_runner_owned_config_candidate_is_declarative) ... ok
test_self_consistent_but_noncanonical_workload_is_rejected (serving_native.test_contract_v2.ContractV2AuditTest.test_self_consistent_but_noncanonical_workload_is_rejected) ... ok
test_series_and_repeat_counts_close_exactly (serving_native.test_contract_v2.ContractV2AuditTest.test_series_and_repeat_counts_close_exactly) ... ok
test_silent_fallback_is_rejected_with_closed_totals (serving_native.test_contract_v2.ContractV2AuditTest.test_silent_fallback_is_rejected_with_closed_totals) ... ok
test_trusted_config_api_is_rejected_outside_owned_workloads (serving_native.test_contract_v2.ContractV2AuditTest.test_trusted_config_api_is_rejected_outside_owned_workloads) ... ok
test_unknown_workload_is_rejected (serving_native.test_contract_v2.ContractV2AuditTest.test_unknown_workload_is_rejected) ... ok
test_workload_hash_is_fail_closed (serving_native.test_contract_v2.ContractV2AuditTest.test_workload_hash_is_fail_closed) ... ok
test_wrong_artifact_hash_is_rejected (serving_native.test_contract_v2.ContractV2AuditTest.test_wrong_artifact_hash_is_rejected) ... ok

----------------------------------------------------------------------
Ran 29 tests in 0.069s

OK
[exit 0]
```

The delegated pseudo-win regression asserts both `valid == False` and the
auditor-owned returned `performance_gate_passed == False`.

## Structural selftests

```text
$ python3 serving_native/selftest.py
serving_native selftest OK: 40 fixed workloads
[exit 0]

$ python3 testbench/bin/selftest.py
selftest: 24 tasks, 0 problems
[exit 0]
```

## Complete CPU verifier

```text
$ python3 testbench/bin/verify_harness.py --skip-task-projection --json
{
  "ok": true,
  "checks": [
    {
      "cmd": ["/home/qinhaiyan/miniconda3/bin/python3", "-m", "py_compile", "testbench/bin/check_env.py", "testbench/bin/audit_result.py", "testbench/bin/selftest.py", "testbench/bin/brief.py", "testbench/bin/bw_ceiling.py", "testbench/bin/knowledge.py", "testbench/bin/kwiki_bridge.py", "testbench/bin/sync_glm52_tasks.py", "testbench/bin/verify_harness.py", "testbench/harness/evaluate_task.py", "testbench/harness/glm52_ops.py", "testbench/harness/gpu_lease.py", "testbench/harness/result_store.py", "serving_native/audit_result.py", "serving_native/contract_v2.py", "serving_native/launch.py", "serving_native/runner.py", "serving_native/selftest.py", "serving_native/test_contract_v2.py", "serving_native/workloads.py"],
      "returncode": 0,
      "stdout": "",
      "stderr": ""
    },
    {
      "cmd": ["/home/qinhaiyan/miniconda3/bin/python3", "testbench/bin/selftest.py"],
      "returncode": 0,
      "stdout": "selftest: 24 tasks, 0 problems\n",
      "stderr": ""
    },
    {
      "cmd": ["/home/qinhaiyan/miniconda3/bin/python3", "serving_native/selftest.py"],
      "returncode": 0,
      "stdout": "serving_native selftest OK: 40 fixed workloads\n",
      "stderr": ""
    },
    {
      "cmd": ["/home/qinhaiyan/miniconda3/bin/python3", "-m", "unittest", "serving_native.test_contract_v2"],
      "returncode": 0,
      "stdout": "",
      "stderr": ".............................\n----------------------------------------------------------------------\nRan 29 tests in 0.074s\n\nOK\n"
    },
    {
      "cmd": ["/home/qinhaiyan/miniconda3/bin/python3", "testbench/bin/knowledge.py", "lint"],
      "returncode": 0,
      "stdout": "knowledge lint: 12 entries, 0 problems\n",
      "stderr": ""
    },
    {
      "cmd": ["/home/qinhaiyan/miniconda3/bin/python3", "testbench/bin/knowledge.py", "index", "--check"],
      "returncode": 0,
      "stdout": "index --check: 0 stale\n",
      "stderr": ""
    },
    {
      "cmd": ["/home/qinhaiyan/miniconda3/bin/python3", "testbench/bin/knowledge.py", "distill", "--check"],
      "returncode": 0,
      "stdout": "distill --check: up to date\n",
      "stderr": ""
    },
    {
      "cmd": ["git", "diff", "--check", "--", "AGENTS.md", "testbench/README.md", "testbench/VERIFY.md", "testbench/setup_env.sh", "testbench/bin", "testbench/harness", "testbench/knowledge", "testbench/tasks/glm52", "serving_native", ":(exclude)testbench/tasks/glm52/*/candidate.py"],
      "returncode": 0,
      "stdout": "",
      "stderr": ""
    },
    {
      "cmd": ["audit_result.py", "runs/glm52/*/*/result.json"],
      "returncode": 0,
      "strict": false,
      "audited": 0,
      "invalid": 0,
      "official": 0,
      "provisional": 0
    },
    {
      "cmd": ["verify_harness.py", "pointer-audit"],
      "returncode": 0,
      "strict": false,
      "index_rows": 0,
      "latest_files": 0,
      "stale_index": 0,
      "stale_latest": 0,
      "malformed": 1,
      "mismatched": 0,
      "problem_count": 1,
      "_problem": {
        "kind": "index",
        "path": "/home/qinhaiyan/glm52-v2-goal-runs/worktrees/00-serving-native-v2-contract/kernel-harness/runs/index.jsonl",
        "detail": "runs/index.jsonl is missing"
      }
    }
  ]
}
[exit 0]
```

The missing `runs/index.jsonl` is the verifier's documented non-strict pointer
advisory for this empty worktree; the verifier's top-level result is `ok: true`.
No historical result was cited.

## Standalone syntax, lint, and diff gates

```text
$ python3 -m py_compile serving_native/audit_result.py serving_native/contract_v2.py serving_native/launch.py serving_native/runner.py serving_native/selftest.py serving_native/test_contract_v2.py serving_native/workloads.py testbench/bin/verify_harness.py
[no output]
[exit 0]

$ ruff check serving_native/audit_result.py serving_native/contract_v2.py serving_native/runner.py serving_native/selftest.py serving_native/test_contract_v2.py serving_native/workloads.py testbench/bin/verify_harness.py
All checks passed!
[exit 0]

$ git diff --check -- serving_native testbench/bin/verify_harness.py
[no output]
[exit 0]
```

## P0 coverage disposition

- Canonical registry binding rejects unknown or self-consistent modified
  workloads.
- Candidate loader remains present in `sys.modules` for actual import
  provenance.
- The old candidate-controlled delegation flag has no authority; the invalid
  delegated pseudo-win returns a runner-owned false promotion gate.
- The runner-owned config API is positive-tested only on DeepEP normal mode and
  negative-tested on an O-projection workload.
- Graph capture IDs, round-robin sample bindings, raw graph handles, stream IDs,
  node count/type/kernel/forbidden derivations, and adapter/copy rejection are
  adversarially tested.
- Requested/completed series, repeats, raw samples, aggregate summaries,
  top-level calls, and every per-phase counter close exactly.
