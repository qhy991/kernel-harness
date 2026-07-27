#!/usr/bin/env python3
"""Map each DeepGEMM JIT artifact to its generated template configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET_PREFIX = "kernel.sm100_m_grouped_fp8_fp4_gemm_masked_1d1d."
TEMPLATE_FIELDS = (
    "major_a",
    "major_b",
    "gran_k_a",
    "gran_k_b",
    "k_alignment",
    "shape_m",
    "shape_n",
    "shape_k",
    "block_m",
    "block_n",
    "block_k",
    "num_groups",
    "swizzle_a",
    "swizzle_b",
    "swizzle_cd",
    "stages",
    "non_epilogue_threads",
    "epilogue_threads",
    "num_multicast",
    "multicast_on_a",
    "num_sms",
    "swap_ab",
    "ensure_zero_padding",
    "gemm_type",
    "with_accumulation",
    "a_dtype",
    "b_dtype",
    "cd_dtype",
    "epilogue",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_template(path: Path) -> dict[str, str]:
    text = path.read_text()
    match = re.search(
        r"sm100_fp8_fp4_gemm_1d1d_impl<(?P<args>.*?)>\s*\);",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f"template instantiation missing: {path}")
    values = [value.strip() for value in match.group("args").split(",")]
    if len(values) != len(TEMPLATE_FIELDS):
        raise RuntimeError(
            f"unexpected template arity in {path}: {len(values)} != {len(TEMPLATE_FIELDS)}"
        )
    return dict(zip(TEMPLATE_FIELDS, values))


def inventory(
    run_dir: Path,
    *,
    expected_alignment: int,
    campaign_id: str,
    require_asm: bool,
) -> dict:
    cache_root = run_dir / "cache/cache"
    if not cache_root.is_dir():
        raise FileNotFoundError(f"JIT cache missing: {cache_root}")
    kernels = []
    for kernel_dir in sorted(cache_root.glob(f"{TARGET_PREFIX}*")):
        source = kernel_dir / "kernel.cu"
        if not source.is_file():
            raise FileNotFoundError(f"generated source missing: {source}")
        artifacts = {}
        for artifact in sorted(kernel_dir.iterdir()):
            if artifact.is_file():
                artifacts[artifact.name] = {
                    "bytes": artifact.stat().st_size,
                    "sha256": sha256(artifact),
                }
        config = parse_template(source)
        required_config = {
            # The masked API defaults compiled_dims="nk"; logical M=1024 is
            # dynamic in the generated template and is validated by the
            # config metadata/benchmark ABI instead.
            "shape_m": "0",
            "shape_n": "6144",
            "shape_k": "2048",
            "num_groups": "32",
            "gemm_type": "GemmType::MGroupedMasked",
        }
        observed_config = {key: config[key] for key in required_config}
        if observed_config != required_config:
            raise RuntimeError(
                f"non-production masked config in {source}: "
                f"{observed_config} != {required_config}"
            )
        required_artifacts = {"kernel.cu", "kernel.cubin"}
        if require_asm:
            required_artifacts.update(("kernel.ptx", "kernel.sass"))
        missing_artifacts = sorted(required_artifacts - set(artifacts))
        empty_artifacts = sorted(
            name for name in required_artifacts if artifacts.get(name, {}).get("bytes", 0) <= 0
        )
        if missing_artifacts or empty_artifacts:
            raise RuntimeError(
                f"incomplete JIT artifacts in {kernel_dir}: "
                f"missing={missing_artifacts} empty={empty_artifacts}"
            )
        kernels.append(
            {
                "cache_key": kernel_dir.name.removeprefix(TARGET_PREFIX),
                "directory": str(kernel_dir.relative_to(ROOT)),
                "config": config,
                "artifacts": artifacts,
            }
        )
    if not kernels:
        raise RuntimeError(f"no target DeepGEMM kernels in {cache_root}")
    matching = [
        kernel
        for kernel in kernels
        if int(kernel["config"]["block_m"]) == expected_alignment
    ]
    if not matching:
        raise RuntimeError(
            f"no masked JIT kernel with candidate block_m={expected_alignment} "
            f"in {cache_root}"
        )
    return {
        "schema_version": 2,
        "campaign_id": campaign_id,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "expected_alignment": expected_alignment,
        "candidate_cache_keys": [kernel["cache_key"] for kernel in matching],
        "kernels": kernels,
    }


def write_outputs(run_dir: Path, payload: dict) -> None:
    analysis = run_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    json_path = analysis / "jit_inventory.json"
    temporary = json_path.with_name(f".{json_path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, json_path)
    lines = [
        "# DeepGEMM JIT inventory",
        "",
        "| Cache key | GEMM type | BM | BN | BK | Stages | SMs | Cubin SHA256 | PTX | SASS |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for kernel in payload["kernels"]:
        config = kernel["config"]
        artifacts = kernel["artifacts"]
        cubin = artifacts.get("kernel.cubin", {}).get("sha256", "missing")
        lines.append(
            "| {cache_key} | {gemm_type} | {block_m} | {block_n} | {block_k} | "
            "{stages} | {num_sms} | {cubin} | {ptx} | {sass} |".format(
                cache_key=kernel["cache_key"],
                cubin=cubin,
                ptx="yes" if "kernel.ptx" in artifacts else "no",
                sass="yes" if "kernel.sass" in artifacts else "no",
                **config,
            )
        )
    markdown_path = analysis / "jit_inventory.md"
    temporary = markdown_path.with_name(f".{markdown_path.name}.tmp.{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, markdown_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--expected-alignment", type=int, choices=(16, 32, 64, 96, 128), required=True
    )
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--require-asm", action="store_true")
    args = parser.parse_args()
    for raw in args.run_dirs:
        run_dir = raw.resolve()
        if ROOT not in run_dir.parents:
            raise RuntimeError(f"run directory outside isolated worktree: {run_dir}")
        payload = inventory(
            run_dir,
            expected_alignment=args.expected_alignment,
            campaign_id=args.campaign_id,
            require_asm=args.require_asm,
        )
        write_outputs(run_dir, payload)
        print(f"{run_dir}: {len(payload['kernels'])} target kernels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
