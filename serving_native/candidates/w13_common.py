"""CPU-only artifact binding shared by W13 serving-native candidates."""

from __future__ import annotations

import json
import os
from pathlib import Path


def artifact_paths() -> tuple[str, ...]:
    manifest_text = os.environ.get("SGLANG_GLM52_W13_DECODE_MANIFEST", "").strip()
    if not manifest_text:
        raise RuntimeError("SGLANG_GLM52_W13_DECODE_MANIFEST is required")
    manifest = Path(manifest_text).expanduser().resolve()
    document = json.loads(manifest.read_text())
    provider_text = os.environ.get("SGLANG_GLM52_HOTSPOT_MODULE", "").strip()
    if not provider_text:
        raise RuntimeError("SGLANG_GLM52_HOTSPOT_MODULE is required")
    provider = Path(provider_text).expanduser().resolve()
    paths = [manifest, provider]
    for name in ("stock", "candidate"):
        record = document["variants"][name]
        paths.extend(
            (
                Path(record["package"]) / "__init__.py",
                Path(record["shared_object"]),
                Path(record["build_ninja"]),
            )
        )
        paths.extend(
            path
            for path in sorted(Path(record["jit_cache"]).rglob("*"))
            if path.is_file()
        )
    return tuple(str(path.resolve()) for path in paths)
