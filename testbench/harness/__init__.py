"""kernel-harness self-contained test runtime.

Owns the testing method: axis resolution, input building, CUPTI device-kernel timing,
correctness, and reward-hack defense. `driver.py` is the subprocess entrypoint used by
`evaluate.py`.
"""
