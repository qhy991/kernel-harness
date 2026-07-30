#!/usr/bin/env python3
"""CPU-only environment, source, ABI, and negative-evidence audit.

Runs before any source change and without initializing CUDA, as required by the
follow-up plan's "Begin with CPU-only environment, source, ABI, and
negative-evidence audits" instruction and by the shared GPU-scheduling rule
that source analysis must not hold a GPU.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


WORKTREES = {
    "kernel_harness": Path(
        "/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/"
        "flashmla-sparse-decode/kernel-harness"
    ),
    "sglang": Path(
        "/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/"
        "flashmla-sparse-decode/sglang"
    ),
    "flashmla": Path(
        "/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/"
        "flashmla-sparse-decode/flashmla"
    ),
}

# Round-3 plan "Worktrees (reuse; do not create new clones)" mandates continuing
# at or after these commits. The baseline is P1, not stock.
REQUIRED_BASES = {
    "kernel_harness": "c6a5802",
    "sglang": "d7fe89a71",
    "flashmla": "b5af443",
}

TASK_CACHE_PREFIX = (
    "/home/qinhaiyan/glm52-hotspot-goal-runs/cache/flashmla_ptx_graph_only_r3"
)

# Round-2 is the immediately prior campaign whose ledger and negative evidence
# this round inherits; round-1 is reachable through it.
PRIOR_CAMPAIGN = (
    WORKTREES["kernel_harness"]
    / "profile/glm52-flashmla-ptx-sass-followup-20260730"
)

# This goal's own evidence directory is untracked until its first commit. It is
# new output, not a pre-existing edit, so it is enumerated explicitly instead of
# being folded into a blanket "ignore dirt" allowance.
CAMPAIGN_DIR_ENTRY = "?? profile/glm52-flashmla-graph-only-r3-20260730/"

# Plan section 0.2 frozen contract, transcribed for machine comparison against
# the SGLang guard and the harness workload fixtures.
FROZEN_ABI = {
    16: {
        "q_shape": [16, 1, 64, 576],
        "kv_shape": [2049, 64, 1, 656],
        "indices_shape": [16, 1, 2048],
        "cache_seqlens_shape": [16],
        "tile_scheduler_metadata_shape": [148, 8],
        "num_splits_shape": [17],
        "block_table_shape": [16, 0],
        "out_shape": [16, 1, 64, 512],
        "lse_shape": [16, 64, 1],
    },
    32: {
        "q_shape": [32, 1, 64, 576],
        "kv_shape": [4097, 64, 1, 656],
        "indices_shape": [32, 1, 2048],
        "cache_seqlens_shape": [32],
        "tile_scheduler_metadata_shape": [148, 8],
        "num_splits_shape": [33],
        "block_table_shape": [32, 0],
        "out_shape": [32, 1, 64, 512],
        "lse_shape": [32, 64, 1],
    },
}
FROZEN_SCALARS = {
    "head_dim_v": 512,
    "softmax_scale": 0.0625,
    "page_size": 64,
    "topk": 2048,
    "kv_bytes_per_token": 656,
    "num_sm_parts": 148,
    "is_fp8_kvcache": True,
    "causal": False,
}
SYMBOL_PREFIX = "infini_kernel_glm52_flashmla_sparse_decode"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_worktrees() -> dict[str, object]:
    result: dict[str, object] = {}
    for name, path in WORKTREES.items():
        head = run("git", "-C", str(path), "rev-parse", "HEAD")
        dirty = run("git", "-C", str(path), "status", "--porcelain=v1").splitlines()
        required = REQUIRED_BASES[name]
        # "at or after" is an ancestry test, not string equality.
        contains = (
            subprocess.run(
                ["git", "-C", str(path), "merge-base", "--is-ancestor", required, head],
                check=False,
            ).returncode
            == 0
        )
        unexpected = [entry for entry in dirty if entry != CAMPAIGN_DIR_ENTRY]
        result[name] = {
            "path": str(path),
            "head": head,
            "branch": run("git", "-C", str(path), "branch", "--show-current"),
            "dirty_entries": dirty,
            "clean": not dirty,
            "unexpected_dirty_entries": unexpected,
            "only_dirt_is_this_campaign_output_dir": not unexpected,
            "required_base": required,
            "head_is_at_or_after_required_base": contains,
        }
    return result


def audit_sglang_abi_guard() -> dict[str, object]:
    """Statically confirm the fail-closed page-count and scale gates."""
    guard = (
        WORKTREES["sglang"] / "python/sglang/srt/layers/glm52_opt/dispatch.py"
    )
    text = guard.read_text()
    tree = ast.parse(text)
    # SGLang d7fe89a71 hoisted the per-bucket shape/stride literals out of the
    # guard into an lru_cached contracts helper, so the audit reads both bodies.
    # The guard consumes those literals through contracts[...] and still fails
    # closed on each of them; only their textual location changed.
    wanted = ("_flashmla_hotspot_abi_matches", "_flashmla_expected_contracts")
    bodies: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            bodies[node.name] = ast.get_source_segment(text, node) or ""
    missing = [name for name in wanted if name not in bodies]
    if missing:
        raise AssertionError(f"missing fail-closed guard functions: {missing}")
    guard_body = bodies["_flashmla_hotspot_abi_matches"]
    body = "\n".join(bodies[name] for name in wanted)
    checks = {
        # The page count must still gate the guard's own return, not merely
        # appear in the precomputed table.
        "guard_returns_on_page_count": "expected_num_pages is not None" in guard_body
        and 'contracts["kv_shape"]' in guard_body,
        "rejects_non_decode": "get_forward_mode() is not ForwardMode.DECODE" in text,
        "exact_page_counts": "{16: 2049, 32: 4097}" in body,
        "exact_softmax_scale": "float(softmax_scale) == 0.0625" in guard_body,
        "requires_fp8_kvcache": "is_fp8_kvcache is True" in guard_body,
        "fp8_e4m3_kv_dtype": "torch.float8_e4m3fn" in guard_body,
        "kv_contiguous_zero_offset": (
            "k_cache.is_contiguous()" in guard_body
            and "k_cache.storage_offset() == 0" in guard_body
        ),
        "metadata_shape_148_8": '"tile_meta_shape": (148, 8)' in body,
        "single_device": "k_cache.device == device" in guard_body,
    }
    return {
        "path": str(guard.relative_to(WORKTREES["sglang"])),
        "sha256": sha256_file(guard),
        "checks": checks,
        "all_present": all(checks.values()),
    }


def audit_sglang_graph_only() -> dict[str, object]:
    """Confirm the round-3 graph-only selection contract statically.

    Round-3 promotion requires that eager decode never reaches the provider, so
    the eager containing lane is a stock fallback rather than a 1.03x gate. That
    only holds if the spec is ``graph_only``, the env default is on, and the
    dispatch returns before the ABI guard when the stream is not capturing.
    """
    root = WORKTREES["sglang"] / "python/sglang/srt/layers/glm52_opt"
    dispatch = root / "dispatch.py"
    config_py = root / "config.py"
    registry = root / "registry.py"
    dispatch_text = dispatch.read_text()
    config_text = config_py.read_text()
    registry_text = registry.read_text()

    tree = ast.parse(dispatch_text)
    target = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "try_dispatch_flashmla_sparse_decode"
        ):
            target = node
            break
    if target is None:
        raise AssertionError("try_dispatch_flashmla_sparse_decode is absent")
    body = ast.get_source_segment(dispatch_text, target) or ""

    # The graph-only early return must precede the ABI guard call, otherwise an
    # eager step still pays the guard cost this round's policy assumes is gone.
    graph_gate_pos = body.find("_is_cuda_graph_capturing()")
    abi_guard_pos = body.find("_flashmla_hotspot_abi_matches(")
    checks = {
        "spec_declares_graph_only": "graph_only: bool = False" in registry_text
        and "graph_only=True," in registry_text,
        "dispatch_checks_capture": "_is_cuda_graph_capturing()" in body,
        "dispatch_honors_spec_graph_only": "spec.graph_only" in body,
        "dispatch_honors_env": "config.flashmla_graph_only_enabled()" in body,
        "graph_gate_precedes_abi_guard": (
            graph_gate_pos != -1
            and abi_guard_pos != -1
            and graph_gate_pos < abi_guard_pos
        ),
        "env_default_is_on": 'os.environ.get("SGLANG_GLM52_FLASHMLA_GRAPH_ONLY", "1")'
        in config_text,
        "env_can_be_disabled_for_diagnostics": 'raw in ("0", "false", "no", "off")'
        in config_text,
        "capture_probe_uses_torch": "torch.cuda.is_current_stream_capturing()"
        in dispatch_text,
    }
    return {
        "dispatch_sha256": sha256_file(dispatch),
        "config_sha256": sha256_file(config_py),
        "registry_sha256": sha256_file(registry),
        "checks": checks,
        "all_present": all(checks.values()),
        "expected_eager_behavior": "no provider hit; stock fallback",
    }


def audit_flashmla_variants() -> dict[str, object]:
    hotspot = WORKTREES["flashmla"] / "csrc/glm52_hotspot"
    variants = {}
    for path in sorted(hotspot.glob("v32_*.cu")):
        text = path.read_text()
        macros = sorted(
            line.split()[1]
            for line in text.splitlines()
            if line.startswith("#define GLM52_")
        )
        renames = [
            line
            for line in text.splitlines()
            if "flash_fwd_splitkv_mla_fp8_sparse_kernel" in line
            and line.startswith("#define")
        ]
        variants[path.name] = {
            "sha256": sha256_file(path),
            "goal_macros": macros,
            "renames_global_symbol": bool(renames),
            "symbol_prefix_ok": SYMBOL_PREFIX in text,
        }
    kernel = WORKTREES["flashmla"] / "csrc/sm100/decode/head64/kernel.cuh"
    config = WORKTREES["flashmla"] / "csrc/sm100/decode/head64/config.h"
    kernel_text = kernel.read_text()
    return {
        "variant_sources": variants,
        "all_variants_prefixed": all(
            entry["symbol_prefix_ok"] and entry["renames_global_symbol"]
            for entry in variants.values()
        ),
        "kernel_cuh_sha256": sha256_file(kernel),
        "config_h_sha256": sha256_file(config),
        "macro_gates_in_kernel": sorted(
            {
                token
                for token in kernel_text.replace("(", " ").replace(")", " ").split()
                if token.startswith("GLM52_")
            }
        ),
        # The upstream default instantiations must remain macro-free so that
        # goal-local variants cannot change the production kernel.
        "default_instantiations": sorted(
            path.name
            for path in (
                WORKTREES["flashmla"] / "csrc/sm100/decode/head64/instantiations"
            ).glob("*")
        ),
    }


def audit_prior_negative_evidence() -> dict[str, object]:
    ledger_path = PRIOR_CAMPAIGN / "evidence/attempt_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    closed = {}
    for attempt in ledger["attempts"]:
        closed[attempt["id"]] = {
            "variant": attempt["variant"],
            "outcome": attempt["outcome"],
            "hypothesis": attempt["hypothesis"],
            "why_rejected": attempt.get("why"),
            "generated_binary": attempt.get("generated_binary"),
        }
    audit = json.loads((PRIOR_CAMPAIGN / "evidence/timing_gate_audit.json").read_text())
    binaries = json.loads(
        (PRIOR_CAMPAIGN / "evidence/binary_manifest.json").read_text()
    )
    return {
        "ledger_path": str(ledger_path),
        "ledger_sha256": sha256_file(ledger_path),
        "terminal_disposition": ledger["terminal_disposition"],
        "closed_attempts": closed,
        "material_candidates_used": ledger["material_candidate_budget"]["used"],
        "evidence_gated_non_attempts": ledger["evidence_gated_non_attempts"],
        "timing_gate_audit_sha256": sha256_file(
            PRIOR_CAMPAIGN / "evidence/timing_gate_audit.json"
        ),
        "timing_gate_audit_keys": sorted(audit.keys()),
        "binary_manifest_sha256": sha256_file(
            PRIOR_CAMPAIGN / "evidence/binary_manifest.json"
        ),
        "binary_manifest_keys": sorted(binaries.keys()),
    }


def audit_cpu_toolchain() -> dict[str, object]:
    cache_env = {
        name: os.environ.get(name)
        for name in (
            "CUDA_CACHE_PATH",
            "TORCH_EXTENSIONS_DIR",
            "TRITON_CACHE_DIR",
            "XDG_CACHE_HOME",
            "DG_JIT_CACHE_DIR",
            "SGLANG_DG_CACHE_DIR",
            "GLM52_TASK_BUILD_DIR",
            "MAX_JOBS",
            "NVCC_THREADS",
            "CMAKE_BUILD_PARALLEL_LEVEL",
        )
    }
    path_like = {
        name: value
        for name, value in cache_env.items()
        if value and ("DIR" in name or "PATH" in name or name == "XDG_CACHE_HOME")
    }
    free = shutil.disk_usage("/")
    return {
        "nvcc": run("/usr/local/cuda/bin/nvcc", "--version").splitlines()[-1],
        "nvcc_release": [
            line for line in run("/usr/local/cuda/bin/nvcc", "--version").splitlines()
            if "release" in line
        ],
        "gxx": run("g++", "--version").splitlines()[0],
        "ncu": [
            line for line in run("ncu", "--version").splitlines() if "Version" in line
        ],
        "nsys": [
            line for line in run("nsys", "--version").splitlines() if line.strip()
        ][:1],
        "cache_environment": cache_env,
        "all_cache_paths_task_local": all(
            value.startswith(TASK_CACHE_PREFIX) for value in path_like.values()
        ),
        "task_cache_prefix": TASK_CACHE_PREFIX,
        "root_free_gib": free.free / (1024**3),
        "stop_expansion_below_gib": 8,
        "disk_threshold_satisfied": free.free >= 8 * 1024**3,
    }


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {output}")
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ[
        "CUDA_VISIBLE_DEVICES"
    ] not in ("",):
        raise RuntimeError(
            "this audit is CPU-only; do not run it inside a GPU lease"
        )

    evidence = {
        "schema_version": 1,
        "stage": "P0_cpu_only_audit",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_initialized": False,
        "worktrees": audit_worktrees(),
        "sglang_abi_guard": audit_sglang_abi_guard(),
        "sglang_graph_only": audit_sglang_graph_only(),
        "flashmla_source": audit_flashmla_variants(),
        "frozen_abi_from_plan_section_0_2": {
            "per_bucket": {str(k): v for k, v in FROZEN_ABI.items()},
            "scalars": FROZEN_SCALARS,
            "candidate_symbol_prefix": SYMBOL_PREFIX,
        },
        "prior_negative_evidence": audit_prior_negative_evidence(),
        "cpu_toolchain": audit_cpu_toolchain(),
    }
    gates = {
        "no_unexpected_worktree_edits": all(
            entry["only_dirt_is_this_campaign_output_dir"]
            for entry in evidence["worktrees"].values()
        ),
        "source_worktrees_clean": all(
            evidence["worktrees"][name]["clean"] for name in ("sglang", "flashmla")
        ),
        "all_worktrees_at_or_after_required_base": all(
            entry["head_is_at_or_after_required_base"]
            for entry in evidence["worktrees"].values()
        ),
        "sglang_guard_complete": evidence["sglang_abi_guard"]["all_present"],
        "sglang_graph_only_contract_complete": evidence["sglang_graph_only"][
            "all_present"
        ],
        "all_variants_prefixed": evidence["flashmla_source"]["all_variants_prefixed"],
        "cache_paths_task_local": evidence["cpu_toolchain"][
            "all_cache_paths_task_local"
        ],
        "disk_threshold_satisfied": evidence["cpu_toolchain"][
            "disk_threshold_satisfied"
        ],
        "prior_disposition_is_no_replacement": (
            evidence["prior_negative_evidence"]["terminal_disposition"]
            == "no-replacement"
        ),
    }
    evidence["gates"] = gates
    evidence["all_gates_pass"] = all(gates.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "all_gates_pass": evidence["all_gates_pass"]},
                     indent=2, sort_keys=True))
    return 0 if evidence["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
