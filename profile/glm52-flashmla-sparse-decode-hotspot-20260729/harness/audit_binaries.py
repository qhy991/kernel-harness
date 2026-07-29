#!/usr/bin/env python3
"""Record generated-binary identity, resources, and SASS instruction evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


VARIANTS = (
    "identity",
    "b1_coord_ready",
    "b2_stagger_nope",
    "b3_native_fp8_bf16",
    "b4_evict_first",
    "b4_rope_early_warp",
    "b5_exact_no_extra",
    "b3_b5_native_exact",
)
SELECTED_OPCODES = (
    "BRA",
    "F2FP",
    "HADD2",
    "HMUL2",
    "IADD3",
    "IMAD",
    "LDG",
    "LDL",
    "LDS",
    "LOP3",
    "STL",
    "STS",
    "SYNCS",
    "UTCCP",
    "UTCHMMA",
    "UTMALDG",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extensions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True)


def function_section(sass: str, pattern: str) -> str:
    lines = sass.splitlines()
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
    return "\n".join(lines[start:end]) + "\n"


def parse_resources(resources: str, pattern: str) -> dict[str, int]:
    lines = resources.splitlines()
    function_index = next(
        index
        for index, line in enumerate(lines)
        if line.startswith(" Function ") and pattern in line
    )
    match = re.search(
        r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",
        lines[function_index + 1],
    )
    if match is None:
        raise AssertionError(f"resource line not found for {pattern}")
    return {
        "registers_per_thread": int(match.group(1)),
        "stack_bytes": int(match.group(2)),
        "static_shared_bytes": int(match.group(3)),
        "local_bytes": int(match.group(4)),
    }


def opcode_histogram(section: str) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for line in section.splitlines():
        match = re.match(r"\s*/\*[0-9a-f]+\*/\s+(?:@\S+\s+)?([A-Z][A-Z0-9_]*)", line)
        if match is None:
            continue
        opcode = match.group(1)
        histogram[opcode] = histogram.get(opcode, 0) + 1
    return histogram


def main() -> int:
    args = parse_args()
    extensions_dir = args.extensions_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {output}")

    entries: list[dict[str, object]] = []
    combine_hashes: set[str] = set()
    for variant in VARIANTS:
        matches = sorted(
            extensions_dir.glob(
                f"infini_kernel_glm52_flashmla_sparse_decode_{variant}_*/"
                f"infini_kernel_glm52_flashmla_sparse_decode_{variant}_*.so"
            )
        )
        if variant == "b5_exact_no_extra":
            matches = [path for path in matches if "ab8c9de505b43383" in path.name]
        if len(matches) != 1:
            raise AssertionError(f"{variant}: expected one survivor binary, got {matches}")
        path = matches[0].resolve()
        symbol_fragment = (
            "identity_main"
            if variant == "identity"
            else f"{variant}_main"
        )
        sass = run("/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(path))
        resources = run(
            "/usr/local/cuda/bin/cuobjdump",
            "--dump-resource-usage",
            str(path),
        )
        main_section = function_section(sass, symbol_fragment)
        required_symbol = (
            "infini_kernel_glm52_flashmla_sparse_decode_"
            f"{symbol_fragment}"
        )
        if required_symbol not in main_section:
            raise AssertionError(
                f"{variant}: required symbol prefix missing from generated SASS"
            )
        combine_start = sass.index(
            "Function : _ZN4smxx6decode28flash_fwd_mla_combine_kernel"
        )
        combine_section = sass[combine_start:]
        combine_hash = sha256_bytes(combine_section.encode())
        combine_hashes.add(combine_hash)
        histogram = opcode_histogram(main_section)
        main_resources = parse_resources(resources, symbol_fragment)
        entries.append(
            {
                "variant": variant,
                "role": "control" if variant == "identity" else "experimental",
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "main_symbol_contains": required_symbol,
                "main_resources": {
                    **main_resources,
                    "ptxas_allocated_barriers_from_build_console": 16,
                    "ptxas_spill_load_bytes_from_build_console": 0,
                    "ptxas_spill_store_bytes_from_build_console": 0,
                },
                "main_static_instruction_count": sum(histogram.values()),
                "selected_opcode_counts": {
                    opcode: histogram.get(opcode, 0)
                    for opcode in SELECTED_OPCODES
                },
                "main_has_no_local_or_spill_instructions": bool(
                    main_resources["stack_bytes"] == 0
                    and main_resources["local_bytes"] == 0
                    and histogram.get("LDL", 0) == 0
                    and histogram.get("STL", 0) == 0
                ),
                "combine_sass_sha256": combine_hash,
            }
        )

    if len(combine_hashes) != 1:
        raise AssertionError(f"combine SASS changed across variants: {combine_hashes}")
    evidence = {
        "schema_version": 1,
        "tool": run("/usr/local/cuda/bin/cuobjdump", "--version").strip(),
        "main_symbol_prefix_required": (
            "infini_kernel_glm52_flashmla_sparse_decode"
        ),
        "all_main_symbols_prefixed": True,
        "combine_sass_identical_across_all_variants": True,
        "combine_sass_sha256": next(iter(combine_hashes)),
        "entries": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "variants": len(entries),
                "combine_sass_identical": True,
                "instruction_counts": {
                    entry["variant"]: entry["main_static_instruction_count"]
                    for entry in entries
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
