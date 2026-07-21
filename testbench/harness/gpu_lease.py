"""GPU selection + timing serialization, so the roofline measurement the whole
harness rests on is never polluted by a co-tenant on the same device.

Two independent, opt-outable services (stdlib-only; no torch import needed):

  pick_idle_gpu()          -> index of the least-busy visible GPU
  gpu_timing_lock(device)  -> a flock held around the timed span so parallel
                              gate runs (campaign waves) serialize their timing

The lock is advisory and per-GPU (`$KH_LOCK_DIR/gpu<N>.lock`, default
/tmp/kernel-harness-locks). It is released when the fd closes, so a killed run
never leaves a stale lock. Everything degrades to a no-op on any error — the gate
must run even where nvidia-smi/rocm-smi/flock are unavailable.
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


def _platform() -> str:
    return os.environ.get("KERNEL_HARNESS_PLATFORM", "cuda").lower()


def _parse_nvidia_smi() -> list[tuple[int, int, int]]:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        return []
    rows = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return rows


def _parse_rocm_smi() -> list[tuple[int, int, int]]:
    """Best-effort idle ranking for AMD GPUs.

    Prefer ``rocm-smi --showuse --showmeminfo vram --csv`` when available; fall
    back to counting devices via ``rocm-smi -i``.
    """
    r = subprocess.run(
        ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--csv"],
        capture_output=True, text=True, timeout=10)
    rows: list[tuple[int, int, int]] = []
    if r.returncode == 0 and r.stdout.strip():
        lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
        # CSV formats vary by ROCm release; tolerate missing util/mem columns.
        for i, line in enumerate(lines[1:] if len(lines) > 1 else lines):
            parts = [p.strip() for p in line.split(",")]
            util = mem = 0
            nums = []
            for p in parts:
                p = p.replace("%", "").replace("MB", "").replace("GB", "").strip()
                try:
                    nums.append(float(p))
                except ValueError:
                    continue
            if len(nums) >= 2:
                util, mem = int(nums[0]), int(nums[1])
            elif len(nums) == 1:
                util = int(nums[0])
            rows.append((i, util, mem))
        if rows:
            return rows
    r2 = subprocess.run(["rocm-smi", "-i"], capture_output=True, text=True, timeout=10)
    if r2.returncode != 0:
        return []
    count = sum(1 for ln in r2.stdout.splitlines() if "GPU[" in ln or "DRM device" in ln)
    if count == 0:
        # Older output: one "GPU ID" style line per device.
        count = sum(1 for ln in r2.stdout.splitlines() if "GPU ID" in ln or "Device" in ln)
    return [(i, 0, 0) for i in range(max(count, 0))]


def pick_idle_gpu(default: int = 0) -> int:
    """Index of the visible GPU with the lowest (utilization, memory-used).

    Uses nvidia-smi on CUDA platforms and rocm-smi on ROCm platforms. Falls back
    to ``default`` if the query tool is missing or unparsable.
    """
    try:
        rows = _parse_rocm_smi() if _platform() == "rocm" else _parse_nvidia_smi()
        if not rows:
            return default
        best = min(rows, key=lambda t: (t[1], t[2], t[0]))
        return best[0]
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
