#!/usr/bin/env python3
"""Produce this campaign's binary manifest and timing-gate audit.

Recomputes every gate decision from the raw ordered pairs rather than trusting
the per-file summaries, and re-derives generated-binary resources from the
cubins with cuobjdump.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
from pathlib import Path


CAMPAIGN = Path(__file__).resolve().parents[1]
EVIDENCE = CAMPAIGN / "evidence"
EXT_DIR = Path(
    "/home/qinhaiyan/glm52-hotspot-goal-runs/cache/flashmla_ptx_sass_followup/"
    "torch_extensions"
)
SYMBOL_PREFIX = "infini_kernel_glm52_flashmla_sparse_decode"
GATE = 1.03

VARIANTS = {
    "identity": {
        "role": "control",
        "note": "source-identical to upstream V32; never promotable",
    },
    "b3_b5_native_exact": {
        "role": "prior-rejected",
        "note": "prior campaign's composite, rebuilt to bind the terminal result",
    },
    "p1_consumer_scale": {
        "role": "candidate",
        "note": "P1 consumer-side scale gather",
    },
    "ablate_scale_chain": {
        "role": "ablation",
        "note": "numerically wrong by construction; bounds the scale-chain ceiling",
    },
}

SELECTED_OPCODES = (
    "BRA", "F2FP", "F2F", "HADD2", "HFMA2", "HMUL2", "IADD3", "IMAD",
    "LDG", "LDL", "LDS", "LOP3", "STL", "STS", "SYNCS", "UTCCP",
    "UTCHMMA", "UTMALDG",
)


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def function_section(sass: str, fragment: str) -> list[str]:
    lines = sass.splitlines()
    start = next(
        i for i, line in enumerate(lines)
        if "Function :" in line and fragment in line
    )
    end = next(
        (i for i in range(start + 1, len(lines)) if "Function :" in lines[i]),
        len(lines),
    )
    return lines[start:end]


def opcode_histogram(section: list[str]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for line in section:
        match = re.match(
            r"\s*/\*[0-9a-f]+\*/\s+(?:@\S+\s+)?([A-Z][A-Z0-9_]*)", line
        )
        if match:
            histogram[match.group(1)] = histogram.get(match.group(1), 0) + 1
    return histogram


def parse_resources(resources: str, fragment: str) -> dict[str, int]:
    lines = resources.splitlines()
    index = next(
        i for i, line in enumerate(lines)
        if line.startswith(" Function ") and fragment in line
    )
    match = re.search(
        r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", lines[index + 1]
    )
    if match is None:
        raise AssertionError(f"no resource line for {fragment}")
    return {
        "registers_per_thread": int(match.group(1)),
        "stack_bytes": int(match.group(2)),
        "static_shared_bytes": int(match.group(3)),
        "local_bytes": int(match.group(4)),
    }


def build_binary_manifest() -> dict[str, object]:
    entries = []
    combine_hashes: set[str] = set()
    for variant, meta in VARIANTS.items():
        matches = sorted(
            EXT_DIR.glob(f"{SYMBOL_PREFIX}_{variant}_*/{SYMBOL_PREFIX}_{variant}_*.so")
        )
        if not matches:
            raise AssertionError(f"{variant}: no built shared object")
        # Several source-hash generations can exist for one variant when an
        # unrelated macro branch was added to kernel.cuh. Record them all and
        # assert their generated main SASS is identical, which is what binds a
        # measurement to the final committed source.
        per_build = []
        main_hashes = set()
        for path in matches:
            fragment = f"{variant}_main"
            sass = run("/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(path))
            resources = run(
                "/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(path)
            )
            section = function_section(sass, fragment)
            required = f"{SYMBOL_PREFIX}_{fragment}"
            if required not in "\n".join(section):
                raise AssertionError(f"{variant}: main symbol prefix missing")
            body = "\n".join(
                re.sub(r"/\*[0-9a-f]{4}\*/", "", line).strip()
                for line in section[1:]
            )
            main_sass_sha = hashlib.sha256(body.encode()).hexdigest()
            main_hashes.add(main_sass_sha)
            histogram = opcode_histogram(section)
            res = parse_resources(resources, fragment)
            combine_start = sass.index(
                "Function : _ZN4smxx6decode28flash_fwd_mla_combine_kernel"
            )
            combine_hashes.add(
                hashlib.sha256(sass[combine_start:].encode()).hexdigest()
            )
            per_build.append(
                {
                    "path": str(path),
                    "so_sha256": sha256_file(path),
                    "build_id_prefix": path.parent.name.rsplit("_", 1)[-1],
                    "main_sass_sha256": main_sass_sha,
                    "main_static_instruction_count": sum(histogram.values()),
                    "main_resources": res,
                    "selected_opcode_counts": {
                        op: histogram.get(op, 0) for op in SELECTED_OPCODES
                    },
                    "no_local_or_spill": bool(
                        res["stack_bytes"] == 0
                        and res["local_bytes"] == 0
                        and histogram.get("LDL", 0) == 0
                        and histogram.get("STL", 0) == 0
                    ),
                }
            )
        entries.append(
            {
                "variant": variant,
                **meta,
                "builds": per_build,
                "generated_main_sass_identical_across_builds": len(main_hashes) == 1,
            }
        )
    if len(combine_hashes) != 1:
        raise AssertionError(f"combine SASS differs across variants: {combine_hashes}")
    return {
        "schema_version": 1,
        "tool": run("/usr/local/cuda/bin/cuobjdump", "--version").strip(),
        "required_main_symbol_prefix": SYMBOL_PREFIX,
        "all_main_symbols_prefixed": True,
        "combine_sass_identical_across_all_variants": True,
        "combine_sass_sha256": next(iter(combine_hashes)),
        "entries": entries,
    }


def recompute_estimators(rows: list[dict]) -> dict[str, float]:
    def rom(subset):
        return statistics.median(
            float(r["a_us"]) for r in subset
        ) / statistics.median(float(r["b_us"]) for r in subset)

    ab = [r for r in rows if r["order"] == "AB"]
    ba = [r for r in rows if r["order"] == "BA"]
    if not ab or not ba:
        raise AssertionError("series lacks both AB and BA observations")
    ab_m, ba_m = rom(ab), rom(ba)
    return {
        "pooled_ratio_of_medians": rom(rows),
        "order_balanced_sqrt_ab_ba": math.sqrt(ab_m * ba_m),
        "ab_ratio_of_medians": ab_m,
        "ba_ratio_of_medians": ba_m,
    }


def build_timing_audit() -> dict[str, object]:
    # Any evidence file carrying both "lanes" and "timing" is a paired
    # measurement, including the stock-vs-stock null controls.
    files = sorted(EVIDENCE.glob("*.json"))
    audited = []
    unparsed = []
    for path in files:
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            # trace_chain stdout captures carry a provider-ready log line ahead
            # of their JSON; they are chain topology records, not timing files.
            unparsed.append(path.name)
            continue
        if "lanes" not in payload or "timing" not in payload:
            continue
        lanes = {}
        for lane, details in payload["lanes"].items():
            series_rows = []
            for series in details["series"]:
                recomputed = recompute_estimators(series["raw_pairs"])
                stored = series["summary"]["estimators"]
                agree = all(
                    math.isclose(recomputed[k], stored[k], rel_tol=1e-9)
                    for k in recomputed
                )
                series_rows.append(
                    {
                        "series": series["series"],
                        "starts_with": series["starts_with"],
                        "graph_capture_order": series.get("graph_capture_order"),
                        "pairs": len(series["raw_pairs"]),
                        "recomputed_estimators": recomputed,
                        "recomputed_matches_stored": agree,
                        "min_estimator": min(recomputed.values()),
                        "all_estimators_ge_gate": all(
                            v >= GATE for v in recomputed.values()
                        ),
                    }
                )
            lanes[lane] = {
                "series": series_rows,
                "series_count": len(series_rows),
                "min_estimator_over_all_series": min(
                    r["min_estimator"] for r in series_rows
                ),
                "max_estimator_over_all_series": max(
                    max(r["recomputed_estimators"].values()) for r in series_rows
                ),
                "passes_gate_every_series_every_estimator": all(
                    r["all_estimators_ge_gate"] for r in series_rows
                ),
            }
        audited.append(
            {
                "file": path.name,
                "m": payload["m"],
                "comparison": payload["comparison"],
                "candidate": payload["b"],
                "replays_per_observation": payload["timing"].get(
                    "replays_per_observation", 1
                ),
                "pairs_per_series": payload["timing"]["pairs_per_series"],
                "series": payload["timing"]["series"],
                "lanes": lanes,
            }
        )
    return {
        "schema_version": 1,
        "gate": GATE,
        "gate_rule": (
            "every one of pooled, order-balanced sqrt(AB*BA), AB-median and "
            "BA-median must be finite and >= 1.03 in every series of every "
            "required lane"
        ),
        "required_lanes": [
            "containing_graph",
            "leaf_graph",
            "containing_eager",
            "leaf_eager",
        ],
        "files": audited,
        "non_timing_files_skipped": unparsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-output", type=Path, required=True)
    parser.add_argument("--timing-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.binary_output, args.timing_output):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite evidence: {path}")

    binary = build_binary_manifest()
    timing = build_timing_audit()
    args.binary_output.write_text(json.dumps(binary, indent=2, sort_keys=True) + "\n")
    args.timing_output.write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n")

    print("=== generated binaries ===")
    for entry in binary["entries"]:
        for build in entry["builds"]:
            res = build["main_resources"]
            print(
                f"  {entry['variant']:20s} {entry['role']:14s} "
                f"instr={build['main_static_instruction_count']:5d} "
                f"reg={res['registers_per_thread']} "
                f"spill_free={build['no_local_or_spill']} "
                f"sass={build['main_sass_sha256'][:12]}"
            )
    print(f"  combine SASS identical: {binary['combine_sass_identical_across_all_variants']}")
    print("=== timing gate ===")
    for entry in timing["files"]:
        for lane, details in entry["lanes"].items():
            print(
                f"  {entry['file']:34s} M{entry['m']:<3d} K={entry['replays_per_observation']:<3d} "
                f"{lane:17s} min={details['min_estimator_over_all_series']:.4f} "
                f"max={details['max_estimator_over_all_series']:.4f} "
                f"pass={details['passes_gate_every_series_every_estimator']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
