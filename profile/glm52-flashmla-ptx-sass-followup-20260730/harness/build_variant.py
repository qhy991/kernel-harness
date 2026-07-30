#!/usr/bin/env python3
"""Build one FlashMLA hotspot variant without holding a GPU.

The shared contract requires that builds do not reserve a GPU, so this script
refuses to run inside a lease. nvcc/ptxas need no CUDA context; the provider
passes an explicit -gencode for sm_100f rather than querying the device.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PROVIDER = (
    REPO_ROOT / "serving_native/candidates/flashmla_sparse_decode_provider.py"
).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if os.environ.get("GLM52_PHYSICAL_GPU"):
        raise RuntimeError("builds must not hold a GPU lease")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["GLM52_FLASHMLA_VARIANT"] = args.variant

    spec = importlib.util.spec_from_file_location(
        f"_glm52_build_{args.variant}", PROVIDER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    log_path = Path(os.environ["TORCH_EXTENSIONS_DIR"]) / f"build_{args.variant}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    spec.loader.exec_module(module)

    extension = Path(module._EXTENSION.__file__).resolve()
    build_dir = extension.parent
    ptxas_lines: list[str] = []
    for candidate in sorted(build_dir.glob("*.log")) + [
        build_dir / "build.ninja",
    ]:
        if candidate.is_file():
            continue
    # ptxas -v output is emitted during the ninja build; capture it by rerunning
    # ninja, which is a no-op when the target is already up to date.
    ninja = subprocess.run(
        ["ninja", "-v"],
        cwd=build_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    combined = ninja.stdout + ninja.stderr
    for line in combined.splitlines():
        if re.search(r"registers|spill|shared memory|stack frame", line):
            ptxas_lines.append(line.strip())

    evidence = {
        "schema_version": 1,
        "variant": args.variant,
        "provider_info": module.PROVIDER_INFO,
        "extension": {
            "path": str(extension),
            "sha256": sha256(extension),
            "size_bytes": extension.stat().st_size,
        },
        "build_dir": str(build_dir),
        "ninja_rebuild_was_noop": "ninja: no work to do" in combined,
        "ptxas_resource_lines": ptxas_lines,
        "cuda_context_created": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "variant": args.variant,
                "build_id": module.PROVIDER_INFO["build_id"],
                "module_name": module.PROVIDER_INFO["module_name"],
                "so_sha256": evidence["extension"]["sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
