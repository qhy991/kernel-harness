"""Enable the source-integrated indexer wq_b graph-overlap policy for one call."""

from __future__ import annotations

import os
from contextlib import contextmanager


_POLICY = {
    "SGLANG_GLM52_OPT": "1",
    "SGLANG_GLM52_OPT_PROFILE": "serving_safe",
    "SGLANG_GLM52_OPT_OPS": "indexer_wq_overlap",
    "SGLANG_GLM52_OPT_M_BUCKETS": "indexer_wq_overlap:16|32",
}


@contextmanager
def _candidate_policy():
    previous = {key: os.environ.get(key) for key in _POLICY}
    try:
        os.environ.update(_POLICY)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run(inputs: dict, runtime):
    # Policy is needed only while Python captures the candidate graph. Replay
    # contains the selected fixed grid and executes no environment mutation.
    with _candidate_policy():
        return runtime.indexer_fused_prepare_store(inputs, candidate_wq=False)
