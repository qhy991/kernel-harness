#!/usr/bin/env python3
"""Read-only, best-effort bridge to the KernelWiki prior-art bank.

The harness's own knowledge base (testbench/knowledge) records what *our* sessions
tried; KernelWiki records what the *upstream world* has published (CUTLASS/SGLang/
vLLM/FlashInfer/PyTorch PRs, technique pages, blogs). This bridge lets one query
surface both. It shells KernelWiki's scripts/query.py and returns its text output,
degrading to None — never raising — when KernelWiki or its deps are absent, so the
harness always runs standalone. Stdlib-only, like the rest of testbench/bin.

    python3 testbench/bin/kwiki_bridge.py --op o_proj --bottleneck memory-bandwidth
    python3 testbench/bin/kwiki_bridge.py "fuse gate-up dual gemm on blackwell"
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Fallback locations tried when $KERNELWIKI_DIR is unset.
_DEFAULT_LOCATIONS = [
    Path.home() / "kernel-design-agents" / "skills" / "KernelWiki",
    Path("/home/qinhaiyan/kernel-design-agents/skills/KernelWiki"),
]

# internal bottleneck.kind (knowledge.py BOTTLENECKS) -> KernelWiki --symptom vocab.
# Best-effort: on a vocab miss the query is retried without --symptom.
_SYMPTOM = {
    "memory-bandwidth": "memory-bound",
    "compute": "compute-bound",
    "occupancy": "low-occupancy",
    "launch-overhead": "tail-effect",
    "kernel-count": "tail-effect",
    "synchronization": "pipeline-stall",
}


def find_wiki() -> Path | None:
    """Locate the KernelWiki clone, or None if not present."""
    env = os.environ.get("KERNELWIKI_DIR")
    candidates = ([Path(env)] if env else []) + _DEFAULT_LOCATIONS
    for c in candidates:
        try:
            if c and (c / "scripts" / "query.py").is_file():
                return c
        except OSError:
            continue
    return None


def query(nl: str | None = None, op: str | None = None, bottleneck: str | None = None,
          techniques=None, limit: int = 6, timeout: int = 30) -> str | None:
    """Return KernelWiki's matching prior-art (compact text), or None on any failure.

    Builds a natural-language query from op/bottleneck/techniques when `nl` is not
    given, and adds a best-effort --symptom filter mapped from the bottleneck kind.
    """
    wiki = find_wiki()
    if not wiki:
        return None
    if not nl:
        terms = []
        if op:
            terms.append(op.replace("_", " "))
        if bottleneck:
            terms.append(bottleneck.replace("-", " "))
        if techniques:
            terms += [t for t in list(techniques)[:3] if t]
        nl = " ".join(terms).strip() or "blackwell kernel optimization"
    base = [sys.executable, str(wiki / "scripts" / "query.py"), nl,
            "--limit", str(limit), "--compact"]
    sym = _SYMPTOM.get((bottleneck or "").lower())
    for cmd in ([base + ["--symptom", sym]] if sym else []) + [base]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout, cwd=str(wiki))
        except Exception:
            return None
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("nl", nargs="?", help="natural-language query")
    ap.add_argument("--op")
    ap.add_argument("--bottleneck")
    ap.add_argument("--technique", action="append", dest="techniques")
    ap.add_argument("--limit", type=int, default=6)
    args = ap.parse_args()
    wiki = find_wiki()
    if not wiki:
        print("KernelWiki not found (set $KERNELWIKI_DIR); prior-art bridge disabled.",
              file=sys.stderr)
        return 3
    out = query(nl=args.nl, op=args.op, bottleneck=args.bottleneck,
                techniques=args.techniques, limit=args.limit)
    if out is None:
        print("no KernelWiki matches (or query failed).", file=sys.stderr)
        return 1
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
