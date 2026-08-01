#!/usr/bin/env python3
"""Run one harness command with a process-group timeout and deterministic cleanup.

The child starts in its own session, so JIT compilers, torchrun workers, profilers,
and their descendants are terminated together. This avoids leaving GPU owners and
stale distributed workers behind when an experiment times out or is interrupted.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys


def _terminate_group(process: subprocess.Popen, grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=max(0.0, grace_seconds))
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run(command: list[str], *, timeout_seconds: float, kill_after_seconds: float) -> int:
    if not command:
        raise ValueError("supervised command is empty")
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return process.wait(timeout=None if timeout_seconds <= 0 else timeout_seconds)
    except subprocess.TimeoutExpired:
        print(
            f"HARNESS TIMEOUT after {timeout_seconds:g}s; terminating process group "
            f"pid={process.pid}: {' '.join(command)}",
            file=sys.stderr,
            flush=True,
        )
        _terminate_group(process, kill_after_seconds)
        return 3
    except KeyboardInterrupt:
        print(
            f"HARNESS INTERRUPTED; terminating process group pid={process.pid}",
            file=sys.stderr,
            flush=True,
        )
        _terminate_group(process, kill_after_seconds)
        return 130


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timeout", type=float, default=1800.0,
                        help="wall timeout in seconds; <=0 disables it")
    parser.add_argument("--kill-after", type=float, default=10.0,
                        help="SIGTERM grace before SIGKILL")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if args.kill_after < 0:
        parser.error("--kill-after must be >= 0")
    return run(
        command,
        timeout_seconds=args.timeout,
        kill_after_seconds=args.kill_after,
    )


if __name__ == "__main__":
    raise SystemExit(main())
