#!/usr/bin/env python3
"""Prove a re-measured variant's committed machine code is the measured code.

C1 was built, validated and timed before ``combine.cuh`` gained the STAGES
template parameter that C2 needs. Re-generating C1 from the final committed
header must not silently change what was measured, so this script accounts for
every instruction that differs between the retained pre-change SASS and the
freshly generated SASS, using the same rule as the identity audit: the only
tolerated difference is the ``__LINE__`` immediate handed to ``__assertfail`` on
the never-taken ``my_num_splits > MAX_SPLITS`` branch, and only when the two
immediates equal the device assert's line number in each header revision.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


HARNESS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_combine_audit", HARNESS / "audit_combine_binaries.py"
)
assert _spec is not None and _spec.loader is not None
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-sass", type=Path, required=True)
    parser.add_argument("--after-sass", type=Path, required=True)
    parser.add_argument("--before-assert-line", type=int, required=True)
    parser.add_argument(
        "--after-header",
        type=Path,
        required=True,
        help="current csrc/glm52_hotspot/combine.cuh",
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {output}")

    before = args.before_sass.read_text().splitlines()
    after = args.after_sass.read_text().splitlines()
    after_assert_line = _audit.assert_source_line(
        args.after_header.expanduser().resolve()
    )
    delta = _audit.explain_identity_delta(
        before, after, args.before_assert_line, after_assert_line
    )
    evidence = {
        "schema_version": 1,
        "label": args.label,
        "before_sass": {
            "path": str(args.before_sass.resolve()),
            "sha256": _audit.digest(args.before_sass.read_text()),
        },
        "after_sass": {
            "path": str(args.after_sass.resolve()),
            "sha256": _audit.digest(args.after_sass.read_text()),
        },
        "delta": delta,
        "measurements_bind_to_committed_source": True,
        "reason": (
            "the only differing instruction is the dead device-assert __LINE__ "
            "immediate, which moved because the assert moved within combine.cuh"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence["delta"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
