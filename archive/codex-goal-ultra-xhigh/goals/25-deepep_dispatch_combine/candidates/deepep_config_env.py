"""Immutable DeepEP Config candidate populated from one recorded environment value.

Use this for a search without rewriting the candidate file between trials::

    DEEPEP_CANDIDATE_CONFIG='{"num_sms":24,...}' \
      serving_native/run.sh ep4_deepep_normal_dispatch_prefill \
        --candidate serving_native/candidates/deepep_config_env.py

The runner validates the exact keys and persists both ``CONFIG`` and the
environment in each result.
"""

from __future__ import annotations

import json
import os


_RAW_CONFIG = os.environ.get("DEEPEP_CANDIDATE_CONFIG")
if not _RAW_CONFIG:
    raise RuntimeError("DEEPEP_CANDIDATE_CONFIG must contain one JSON object")
CONFIG = json.loads(_RAW_CONFIG)
if not isinstance(CONFIG, dict):
    raise TypeError("DEEPEP_CANDIDATE_CONFIG must decode to a JSON object")


def run(inputs, runtime):
    return runtime.reference(inputs, config=CONFIG)
