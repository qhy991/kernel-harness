"""Reward-hack defenses (self-contained).

Adapted from SOL-ExecBench (NVIDIA, Apache-2.0) so kernel-harness owns it with no
sol_execbench import. Captures torch.cuda.Event.elapsed_time identity at module load
(before user code) to detect post-import timer patching, and rejects lazy/proxy outputs.
"""
from __future__ import annotations

from typing import Any, List

import torch

_ELAPSED_TIME_ADDR = None
try:
    _ELAPSED_TIME_ADDR = id(torch.cuda.Event.elapsed_time)
except Exception:
    pass


class RewardHackDetected(RuntimeError):
    """Raised when a reward-hacking pattern is detected."""


def check_monkey_patch() -> None:
    """Detect if torch.cuda.Event.elapsed_time was replaced after module load."""
    try:
        if (_ELAPSED_TIME_ADDR is not None
                and id(torch.cuda.Event.elapsed_time) != _ELAPSED_TIME_ADDR):
            raise RewardHackDetected("torch.cuda.Event.elapsed_time has been monkey-patched")
    except RewardHackDetected:
        raise
    except Exception:
        pass


def check_lazy_outputs(outputs: List[Any]) -> None:
    """Reject non-exact-torch.Tensor outputs (FakeTensor/proxy/lazy) via strict type()."""
    for t in outputs:
        if type(t) is not torch.Tensor:
            raise RewardHackDetected(
                f"Lazy evaluation detected: output is {type(t).__name__}, not torch.Tensor")


def check_thread_injection(before: int, after: int) -> None:
    if after > before:
        raise RewardHackDetected(
            f"Thread injection detected: {after} threads after vs {before} before")
