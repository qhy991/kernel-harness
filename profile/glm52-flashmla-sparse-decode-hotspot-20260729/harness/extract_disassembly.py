#!/usr/bin/env python3
"""Retain control and final-candidate main SASS plus cubin/PTX inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True)


def section(disassembly: str, pattern: str) -> str:
    lines = disassembly.splitlines()
    start = next(
        index
        for index, line in enumerate(lines)
        if "Function :" in line and pattern in line
    )
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if "Function :" in lines[index]
        ),
        len(lines),
    )
    return "\n".join(lines[start:end]).rstrip() + "\n"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> int:
    args = parse_args()
    identity = args.identity.expanduser().resolve()
    candidate = args.candidate.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    outputs = {
        "identity_main": output_dir / "identity_main.sass",
        "candidate_main": output_dir / "b3_b5_native_exact_main.sass",
        "ptx_inventory": output_dir / "b3_b5_native_exact_ptx_inventory.txt",
        "manifest": output_dir / "disassembly_manifest.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to overwrite evidence: {existing}")

    identity_sass = section(
        command("/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(identity)),
        "identity_main",
    )
    candidate_sass = section(
        command("/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(candidate)),
        "b3_b5_native_exact_main",
    )
    ptx_inventory = command(
        "/usr/local/cuda/bin/cuobjdump", "--dump-ptx", str(candidate)
    )
    if ".entry" in ptx_inventory:
        raise AssertionError("unexpected embedded PTX in cubin-only build")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs["identity_main"].write_text(identity_sass)
    outputs["candidate_main"].write_text(candidate_sass)
    outputs["ptx_inventory"].write_text(ptx_inventory)
    manifest = {
        "schema_version": 1,
        "tool": command("/usr/local/cuda/bin/cuobjdump", "--version").strip(),
        "identity_main": {
            "path": str(outputs["identity_main"]),
            "sha256": digest(identity_sass),
            "bytes": len(identity_sass.encode()),
        },
        "candidate_main": {
            "path": str(outputs["candidate_main"]),
            "sha256": digest(candidate_sass),
            "bytes": len(candidate_sass.encode()),
        },
        "ptx_inventory": {
            "path": str(outputs["ptx_inventory"]),
            "sha256": digest(ptx_inventory),
            "embedded_ptx_entry_count": 0,
            "reason": (
                "the fair-measurement build targets code=sm_100f and retains "
                "native cubins, not a compute_100f PTX fallback"
            ),
        },
        "source_inline_ptx": {
            "instruction": "cvt.rn.bf16x2.e4m3x2",
            "final_sass_opcode": "F2FP.BF16.E4M3",
            "evidence": "candidate_main SASS and csrc/sm100/helpers.h",
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest["ptx_inventory"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
