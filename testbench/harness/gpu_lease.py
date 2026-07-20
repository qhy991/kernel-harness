"""GPU selection + timing serialization, so the roofline measurement the whole
harness rests on is never polluted by a co-tenant on the same device.

Two independent, opt-outable services (stdlib-only; no torch import needed):

  pick_idle_gpu()          -> index of the least-busy visible GPU (nvidia-smi)
  gpu_timing_lock(device)  -> a flock held around the timed span so parallel
                              gate runs (campaign waves) serialize their timing

The lock is advisory and per-GPU (`$KH_LOCK_DIR/gpu<N>.lock`, default
/tmp/kernel-harness-locks). It is released when the fd closes, so a killed run
never leaves a stale lock. Everything degrades to a no-op on any error — the gate
must run even where nvidia-smi/flock are unavailable.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
from pathlib import Path

try:
    import fcntl  # POSIX only
    _HAVE_FLOCK = True
except Exception:  # pragma: no cover
    _HAVE_FLOCK = False


def lock_dir() -> Path:
    return Path(os.environ.get("KH_LOCK_DIR", "/tmp/kernel-harness-locks"))


def pick_idle_gpu(default: int = 0) -> int:
    """Index of the visible GPU with the lowest (utilization, memory-used). Falls
    back to `default` if nvidia-smi is missing or unparsable."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return default
        best = None
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            idx, util, mem = int(parts[0]), int(parts[1]), int(parts[2])
            score = (util, mem)
            if best is None or score < best[0]:
                best = (score, idx)
        return best[1] if best else default
    except Exception:
        return default


def device_index(device) -> int:
    s = str(device)
    if ":" in s:
        try:
            return int(s.rsplit(":", 1)[1])
        except ValueError:
            return 0
    return 0


@contextlib.contextmanager
def gpu_timing_lock(device, enabled: bool = True):
    """Hold an exclusive per-GPU flock for the duration of the block. No-op (still
    yields) when disabled or when flock/dir setup fails."""
    if not (enabled and _HAVE_FLOCK):
        yield None
        return
    idx = device_index(device)
    try:
        d = lock_dir()
        d.mkdir(parents=True, exist_ok=True)
        f = open(d / f"gpu{idx}.lock", "w")
    except Exception:
        yield None
        return
    try:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield f
    finally:
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()
        except Exception:
            pass
