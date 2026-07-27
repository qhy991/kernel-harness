#!/usr/bin/env python3
"""Validate this goal's durable evidence and write CPU-only final metadata."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).resolve().parent
PROFILE = ROOT / "profile" / "indexer-score-decode-20260723T113910Z"
SGLANG = ROOT.parent / "sglang"
KNOWLEDGE = (
    ROOT
    / "testbench"
    / "knowledge"
    / "entries"
    / "glm52-production--indexer-score-decode--b200--20260723a.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(path: Path) -> int:
    count = 0
    for line in path.read_text().splitlines():
        if not line:
            continue
        expected, raw_path = line.split("  ", 1)
        artifact = Path(raw_path)
        if not artifact.is_absolute():
            artifact = ROOT / artifact
        if not artifact.is_file():
            raise RuntimeError(f"manifest artifact is missing: {artifact}")
        actual = sha256(artifact)
        if actual != expected:
            raise RuntimeError(
                f"manifest mismatch: {artifact}: {actual} != {expected}"
            )
        count += 1
    return count


def git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    run_paths = {
        "score": EVIDENCE / "runs" / "20260723T113910Z",
        "region": EVIDENCE / "region_runs" / "20260723T120153Z",
        "tp4": EVIDENCE / "tp4_runs" / "20260723T121417Z",
        "superseded": EVIDENCE / "runs" / "20260723T112717Z",
    }
    expected_status = {
        "score": "failures=0",
        "region": "failures=0",
        "tp4": "failures=0",
        "superseded": "failures=1",
    }
    manifest_counts = {}
    for name, run_path in run_paths.items():
        status = (run_path / "status.txt").read_text()
        if expected_status[name] not in status:
            raise RuntimeError(f"unexpected {name} status: {status!r}")
        manifest_counts[name] = verify_manifest(
            run_path / "artifact_manifest.sha256"
        )

    for json_path in (
        EVIDENCE / "backend_validation.json",
        run_paths["score"] / "paired_summary.json",
        run_paths["region"] / "paired_summary.json",
        run_paths["tp4"] / "paired_summary.json",
    ):
        json.loads(json_path.read_text())

    if git("status", "--porcelain", cwd=SGLANG):
        raise RuntimeError("SGLang worktree is not clean")
    sglang_head = git("rev-parse", "HEAD", cwd=SGLANG)
    if sglang_head != "f93f8867b4bc124c9809c9110ec7361ed11b6b4a":
        raise RuntimeError(f"unexpected SGLang HEAD: {sglang_head}")

    model_dir = Path("/mnt/OS-oKqEXySb/models/GLM-5.2-NVFP4")
    hf_snapshot = Path(
        "/home/qinhaiyan/.cache/huggingface/hub/"
        "models--nvidia--GLM-5.2-NVFP4/snapshots/"
        "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa"
    )
    model_files = (
        sorted(str(path.relative_to(model_dir)) for path in model_dir.rglob("*")
               if path.is_file())
        if model_dir.is_dir()
        else []
    )
    hf_entries = (
        sorted(
            [
                {
                "name": str(path.relative_to(hf_snapshot)),
                "size": path.stat().st_size,
                "resolved": str(path.resolve()),
                }
                for path in hf_snapshot.rglob("*")
                if path.is_file()
            ],
            key=lambda item: item["name"],
        )
        if hf_snapshot.is_dir()
        else []
    )
    module_specs = {}
    for module in ("deep_ep", "modelopt"):
        spec = importlib.util.find_spec(module)
        module_specs[module] = None if spec is None else spec.origin

    blocker = {
        "schema_version": 1,
        "required_topology": "TP8/DP8/EP8",
        "available_gpu_count_evidence": (
            "tp4_runs/20260723T121417Z/gpu_identity_start.txt records four B200s"
        ),
        "configured_model_dir": str(model_dir),
        "configured_model_files": model_files,
        "hf_snapshot": str(hf_snapshot),
        "hf_snapshot_files": hf_entries,
        "repo_venv_module_specs": module_specs,
        "disposition": "external validation blocked; stock fallback active",
    }
    validation = {
        "schema_version": 1,
        "manifest_counts": manifest_counts,
        "json_parse": "pass",
        "sglang_head": sglang_head,
        "sglang_worktree": "clean",
        "authoritative_campaigns": {
            "score": str(run_paths["score"].relative_to(ROOT)),
            "region": str(run_paths["region"].relative_to(ROOT)),
            "tp4_diagnostic": str(run_paths["tp4"].relative_to(ROOT)),
        },
        "superseded_campaign": str(
            run_paths["superseded"].relative_to(ROOT)
        ),
        "outcome": "no replacement",
    }

    if args.write:
        (EVIDENCE / "external_environment_probe.json").write_text(
            json.dumps(blocker, indent=2, sort_keys=True) + "\n"
        )
        (EVIDENCE / "final_validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n"
        )
        roots = (EVIDENCE, PROFILE)
        files = []
        for root in roots:
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.name != "final_artifact_manifest.sha256"
            )
        if KNOWLEDGE.is_file():
            files.append(KNOWLEDGE)
        manifest_lines = [
            f"{sha256(path)}  {path.relative_to(ROOT)}"
            for path in sorted(set(files))
        ]
        (EVIDENCE / "final_artifact_manifest.sha256").write_text(
            "\n".join(manifest_lines) + "\n"
        )
    print(json.dumps({"blocker": blocker, "validation": validation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
