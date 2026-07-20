# Kernel-Harness

Agent-ready **SGLang kernel-optimization tasks for GLM-5.2 on B200**.

**24 tasks** = 12 operators × 2 phases (prefill / decode), under
`testbench/tasks/glm52/`. Every operator is defined exactly once in
`testbench/harness/glm52_ops.py`; each task directory only names which problem it
is, and one command judges correctness, latency, speedup and roofline reward:

```bash
T=testbench/tasks/glm52/o_proj_decode

$T/run.sh --describe                        # what is this problem?
$T/run.sh --describe --json                 # ...machine-readable (== problem.json)
$T/run.sh                                   # the gate
$T/run.sh --candidate ~/kernels/mine.py     # test any kernel, without editing the task

# acceptance only (not the gate): swap the candidate into the 12-op layer budget
.venv/bin/python testbench/bin/accept_layer.py --M 32 --task o_proj_decode
```

Exit codes: `0` correct and faster · `1` correct, not faster · `2` incorrect ·
`3` infrastructure or contract error.

Start here: **[`AGENTS.md`](AGENTS.md)**. Worked Triton and CUDA `.cu` candidates:
[`testbench/docs/GLM52_CANDIDATES.md`](testbench/docs/GLM52_CANDIDATES.md).

## Retired

Kimi-K2.7 and MiniMax-M3, the `solution.py` + `definition.json` contract they used,
`evaluate.py` / `integrate.py`, and the older proxy benchmark catalogue have all moved
to [`legacy/`](legacy/README.md). They still run, but they are not the suite and not
an oracle for anything here.

## What makes a run trustworthy

- **One definition.** Inputs, reference, tolerances, masks, cost model and peaks live
  only in `glm52_ops.py`. A task restating any of them is rejected with exit 3.
- **The same bytes.** One frozen input dict feeds the reference and the candidate. The
  shared output buffer is NaN-poisoned between them, so a candidate that computes
  nothing cannot inherit the reference's answer.
- **Upstream correctness.** FlashMLA's three-layer check with DeepGEMM's `calc_diff`
  verbatim as the aggregate — not allclose, and not cosine, which is scale-blind.
- **Device time.** CUPTI cold-L2 device-kernel median. Wall clock would put half this
  op's "bandwidth utilisation" down to Python dispatch.
- **A win is per-shape.** At least one shape ahead and none behind; a candidate may
  fall back to the reference where it cannot win, exactly as SGLang does.
- **Every run is kept.** `runs/<model>/<task>/<run_id>/` holds `result.json`, the
  terminal log, the environment, and a byte-exact copy of the candidate that ran.
