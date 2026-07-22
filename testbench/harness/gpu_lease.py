"""GPU selection + timing serialization, so the roofline measurement the whole
harness rests on is never polluted by a co-tenant on the same device.

Two independent, opt-outable services (stdlib-only; no torch import needed):

  pick_idle_gpu()          -> index of the least-busy visible GPU (nvidia-smi)
  gpu_timing_lock(device)  -> a flock for an explicit device
  locked_idle_gpu()        -> lock-aware idle GPU selection for campaign workers

The lock is advisory and per-GPU (`$KH_LOCK_DIR/gpu<N>.lock`, default
/tmp/kernel-harness-locks). It is released when the fd closes, so a killed run
never leaves a stale lock. Everything degrades to a no-op on any error — the gate
must run even where nvidia-smi/flock are unavailable.
"""
from __future__ import annotations

import contextlib
import errno
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    import fcntl  # POSIX only
    _HAVE_FLOCK = True
except Exception:  # pragma: no cover
    _HAVE_FLOCK = False


def lock_dir() -> Path:
    return Path(os.environ.get("KH_LOCK_DIR", "/tmp/kernel-harness-locks"))


@dataclass
class GpuLease:
    index: int                  # logical CUDA device index for torch (cuda:<index>)
    physical_index: int | None = None  # nvidia-smi index used for advisory locking
    file: object | None = None


def _visible_physical_to_logical() -> dict[int, int] | None:
    """Map nvidia-smi physical indices to CUDA logical indices when CUDA_VISIBLE_DEVICES
    is a simple index list. Return None when all devices are visible or the value uses
    UUID/MIG syntax we cannot safely parse without importing CUDA."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return None
    ids = [p.strip() for p in cvd.split(",") if p.strip()]
    mapping = {}
    for logical, physical in enumerate(ids):
        try:
            mapping[int(physical)] = logical
        except ValueError:
            return None
    return mapping


def _gpu_scores(default: int = 0) -> list[tuple[tuple[int, int], int]]:
    """Visible GPU scores as ((utilization, memory-used), physical index)."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return [((0, 0), default)]
        scores = []
        visible = _visible_physical_to_logical()
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            idx, util, mem = int(parts[0]), int(parts[1]), int(parts[2])
            if visible is not None and idx not in visible:
                continue
            scores.append(((util, mem), idx))
        return sorted(scores) or [((0, 0), default)]
    except Exception:
        return [((0, 0), default)]


def pick_idle_gpu(default: int = 0) -> int:
    """Logical CUDA index of the visible GPU with the lowest (utilization, memory-used). Falls
    back to `default` if nvidia-smi is missing or unparsable."""
    physical = _gpu_scores(default)[0][1]
    mapping = _visible_physical_to_logical()
    return mapping.get(physical, physical) if mapping is not None else physical


def _lock_index(device) -> int:
    idx = device_index(device)
    mapping = _visible_physical_to_logical()
    if mapping is None:
        return idx
    inverse = {logical: physical for physical, logical in mapping.items()}
    return inverse.get(idx, idx)


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
    idx = _lock_index(device)
    f = _open_lock(idx)
    if f is None:
        yield None
        return
    try:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield GpuLease(index=device_index(device), physical_index=idx, file=f)
    finally:
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()
        except Exception:
            pass


def _open_lock(idx: int):
    try:
        d = lock_dir()
        d.mkdir(parents=True, exist_ok=True)
        return open(d / f"gpu{idx}.lock", "w")
    except Exception:
        return None


@contextlib.contextmanager
def locked_idle_gpu(default: int = 0, enabled: bool = True):
    """Pick an idle GPU and hold its lock.

    Selection and locking are one operation: try nonblocking locks in nvidia-smi
    score order, and if every candidate is already locked, block on the currently
    least-busy GPU. This avoids the common race where two campaign workers both
    observe the same idle device and start timing on it.
    """
    if not (enabled and _HAVE_FLOCK):
        logical = pick_idle_gpu(default)
        yield GpuLease(index=logical, physical_index=_lock_index(f"cuda:{logical}"), file=None)
        return

    scores = _gpu_scores(default)
    opened = []
    closed = set()
    chosen = None
    try:
        mapping = _visible_physical_to_logical()
        for _, physical_idx in scores:
            f = _open_lock(physical_idx)
            if f is None:
                continue
            opened.append(f)
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                logical_idx = mapping.get(physical_idx, physical_idx) if mapping is not None else physical_idx
                chosen = GpuLease(index=logical_idx, physical_index=physical_idx, file=f)
                break
            except OSError as exc:
                if getattr(exc, "errno", None) not in (errno.EACCES, errno.EAGAIN):
                    continue
                continue

        if chosen is None:
            physical_idx = scores[0][1]
            f = _open_lock(physical_idx)
            if f is None:
                logical = pick_idle_gpu(default)
                yield GpuLease(index=logical, physical_index=_lock_index(f"cuda:{logical}"), file=None)
                return
            opened.append(f)
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
            except OSError:
                logical = pick_idle_gpu(default)
                yield GpuLease(index=logical, physical_index=_lock_index(f"cuda:{logical}"), file=None)
                return
            logical_idx = mapping.get(physical_idx, physical_idx) if mapping is not None else physical_idx
            chosen = GpuLease(index=logical_idx, physical_index=physical_idx, file=f)

        for f in list(opened):
            if f is not chosen.file:
                try:
                    f.close()
                    closed.add(f)
                except Exception:
                    pass
        yield chosen
    finally:
        if chosen and chosen.file is not None:
            try:
                fcntl.flock(chosen.file, fcntl.LOCK_UN)
                chosen.file.close()
                closed.add(chosen.file)
            except Exception:
                pass
        for f in opened:
            if f not in closed and (chosen is None or f is not chosen.file):
                try:
                    f.close()
                except Exception:
                    pass
