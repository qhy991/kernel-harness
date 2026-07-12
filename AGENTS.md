# Kernel optimization agent guide

The agent-facing benchmark suite is `testbench/`. The root `benchmarks/` directory is
the older proxy-baseline catalogue and is not the optimization oracle.

## Environment

- Run on the target GPU node; production comparisons require the real SGLang kernels.
- Use only the repo-local `.venv`, created with `./testbench/setup_env.sh`.
- Set checkout paths in environment variables or `testbench/harness.env`; never add
  machine-specific paths to source files.
- Before testing, run `.venv/bin/python testbench/bin/check_env.py`.

## Optimization loop

1. Choose one directory under `testbench/tasks/<model>/<task>/`.
2. Read `task.json`, `definition.json`, `reference.py`, and the full workload sweep.
3. Edit only the task's `solution.py`.
4. Run a quick check:
   `.venv/bin/python testbench/bin/evaluate.py <task> --max-workloads 1 --no-baseline`.
5. Run the full gate:
   `.venv/bin/python testbench/bin/evaluate.py <task> --repeat 3`.
6. Treat exit 0 as a kernel-level candidate win only after every workload is correct
   and faster. Verify drop-in compatibility with `testbench/bin/integrate.py`.

Do not modify `reference.py`, workload tolerances, the evaluator, or timing code to
improve a score. Prefer memory-bound/fused families first; FP8 GEMM baselines use
highly tuned DeepGEMM and are intentionally difficult targets.
