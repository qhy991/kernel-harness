# Candidate smoke attempt

The first smoke invocation stopped before candidate import, JIT compilation, or
GPU timing because the repo-local Kernel-Harness venv does not include
`pytest`. The routing test was converted to `unittest`, and the next immutable
attempt is recorded separately.
