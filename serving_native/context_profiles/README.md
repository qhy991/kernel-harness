# KV-context profiles

A profile is a measured serving distribution, not a convenient length sweep. Copy
the diagnostic example, replace its deployment facts and scenarios from a production
trace, set `production_evidence` to `true`, and keep the evidence source auditable.

Required axes are:

- decode: workload/batch bucket and every logical per-request KV length;
- incremental prefill: existing `prefix_tokens`, new `extend_tokens`, batch/ragged
  distribution, cache dtype/page size, selected backend, and execution scope;
- a positive weight for every scenario, summing to 1.0 independently within decode
  and incremental prefill;
- deployment: exact model, SGLang commit, GPU, cache dtype, page size, and topology.

The decode leaf matrix is executable:

```bash
.venv/bin/python serving_native/context_matrix.py \
  --profile serving_native/context_profiles/my-production-profile.json \
  --candidate /absolute/path/to/candidate.py
```

`production_evidence: false`, eager-only runs, warmup below 8, or repeat below 10
always produce `PROBE_ONLY_NO_VERDICT`. The matrix runner intentionally does not run
incremental prefill: SGLang's `ForwardBatch` dispatch and paged-prefix behavior must
be exercised at the real containing-region or end-to-end boundary. A decode matrix
win therefore remains `production_ready: false`. For a production-evidence profile,
the runner also rejects a runtime whose SGLang commit differs from the pinned value.
