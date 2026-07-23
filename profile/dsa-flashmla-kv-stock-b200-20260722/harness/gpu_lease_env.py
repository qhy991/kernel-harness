"""Fail closed unless one physical B200 was assigned by the flexible wrapper."""

from __future__ import annotations

import os


VALID_PHYSICAL_ORDINALS = {"0", "1", "2", "3"}
FLEXIBLE_WRAPPER = "/home/qinhaiyan/glm52-goal-runs/with_flexible_gpu.sh"


def require_flexible_gpu() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in VALID_PHYSICAL_ORDINALS:
        raise RuntimeError(
            "expected exactly one physical GPU ordinal assigned by "
            f"{FLEXIBLE_WRAPPER} -- <command>; got "
            f"CUDA_VISIBLE_DEVICES={visible!r}"
        )
    return int(visible)
