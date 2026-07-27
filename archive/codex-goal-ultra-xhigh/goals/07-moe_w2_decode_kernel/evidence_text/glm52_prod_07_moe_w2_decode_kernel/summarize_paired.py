#!/usr/bin/env python3
"""Validate and summarize the pinned production-W2 paired campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import secrets
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[1]
SGLANG_ROOT = Path(
    "/home/qinhaiyan/glm52-goal-runs/07-moe_w2_decode_kernel/sglang"
).resolve()
DEEP_GEMM_ROOT = (SGLANG_ROOT / "build/deep-gemm-stock-0.1.4.post1").resolve()
BENCHMARK_HARNESS = (
    ROOT / "profile/moe-w2-packed-baseline/harness/benchmark.py"
).resolve()
CONFIG_HARNESS = (
    ROOT / "profile/moe-w2-packed-baseline/harness/profile_once.py"
).resolve()
GRAPH_HARNESS = (
    ROOT / "profile/moe-w2-packed-baseline/harness/graph_replay.py"
).resolve()
EDGE_HARNESS = (
    ROOT / "profile/moe-w2-packed-baseline/harness/edge_mask_correctness.py"
).resolve()
INVENTORY_HARNESS = (EVIDENCE / "inventory_jit_cache.py").resolve()
ALIGNMENT_DRIVER = (ROOT / "profile/run_alignment_portfolio.sh").resolve()
PROFILE_DRIVER = (
    ROOT / "profile/moe-w2-packed-baseline/harness/run_profiles.sh"
).resolve()
GRAPH_DRIVER = (ROOT / "profile/run_graph_checks.sh").resolve()
DEFAULT_MANIFEST = EVIDENCE / "alignment_campaign_manifest.json"
PINNED = {
    "deep_gemm_version": "0.1.4.post1",
    "deep_gemm_python_sha256": "b33e89deacdce241f01f5070d321918f5e5480e3e6d3af569678d4192db4f2a7",
    "deep_gemm_extension_sha256": "cd8beab174071777c972c5948af7706ae2cfb5d2adcdbb7e6fbea253ce3a81bf",
    "deep_gemm_device_source_sha256": "9c1e70677ede6ba09ab98e629482da7874182f8227907382efe0a81658da5a37",
}
MASK_M16 = [2, 3, 6, 3, 1, 5, 3, 7, 5, 6, 8, 5, 4, 3, 2, 1, 2, 4, 7, 5, 5, 1, 3, 9, 5, 5, 4, 2, 2, 4, 4, 2]
MASK_M32 = [4, 5, 9, 7, 5, 9, 11, 9, 8, 11, 14, 6, 9, 5, 8, 5, 6, 9, 13, 7, 7, 5, 8, 14, 11, 7, 8, 7, 7, 9, 8, 5]
WORKLOAD_SPECS = {
    "moe_w2_grouped_decode_m16": (16, 4, 128, MASK_M16),
    "moe_w2_grouped_decode_m32": (32, 8, 256, MASK_M32),
    "moe_w2_grouped_decode_m16_current_source_m5": (16, 5, 128, MASK_M16),
    "moe_w2_grouped_decode_m32_current_source_m9": (32, 9, 256, MASK_M32),
}
TENSOR_ABI = {
    "activation_fp8": ([32, 1024, 2048], [2097152, 2048, 1], "torch.float8_e4m3fn", True),
    "activation_scale": ([32, 1024, 4], [4096, 1, 1024], "torch.int32", False),
    "weight_fp8": ([32, 6144, 2048], [12582912, 2048, 1], "torch.float8_e4m3fn", True),
    "weight_scale": ([32, 6144, 4], [24576, 1, 6144], "torch.int32", False),
    "out": ([32, 1024, 6144], [6291456, 6144, 1], "torch.bfloat16", True),
    "masked_m": ([32], [1], "torch.int32", True),
}
ALIGNMENT_RUN = re.compile(r"moe-w2-(?:packed-baseline|alignment(?:16|32|64|96|128))$")
SMS_RUN = re.compile(r"moe-w2-alignment(?:16|32|64|96|128)-sms\d+$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def _manifest_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign": "alignment",
        "measurement_contract": {
            "sessions": 3,
            "warmup": 5,
            "repeat": 30,
            "alignments": [128, 32, 64, 96, 16],
            "workloads": list(WORKLOAD_SPECS),
        },
        "pinned": PINNED,
        "roots": {
            "kernel_harness": str(ROOT),
            "sglang": str(SGLANG_ROOT),
            "deep_gemm": str(DEEP_GEMM_ROOT),
        },
        "git_heads": {
            "kernel_harness": _git_head(ROOT),
            "sglang": _git_head(SGLANG_ROOT),
        },
        "harness_sha256": {
            "benchmark": _sha256(BENCHMARK_HARNESS),
            "config_probe": _sha256(CONFIG_HARNESS),
            "graph": _sha256(GRAPH_HARNESS),
            "edge": _sha256(EDGE_HARNESS),
            "summarizer": _sha256(Path(__file__).resolve()),
            "inventory": _sha256(INVENTORY_HARNESS),
            "alignment_driver": _sha256(ALIGNMENT_DRIVER),
            "profile_driver": _sha256(PROFILE_DRIVER),
            "graph_driver": _sha256(GRAPH_DRIVER),
        },
    }


def _init_or_load_manifest(
    path: Path, *, create: bool, require_measurement_head: bool
) -> dict[str, Any]:
    expected = _manifest_contract()
    if path.exists():
        manifest = json.loads(path.read_text())
        for key, value in expected.items():
            if key == "git_heads":
                continue
            if manifest.get(key) != value:
                raise RuntimeError(
                    f"campaign manifest drift for {key}: {manifest.get(key)!r} != {value!r}"
                )
        manifest_heads = manifest.get("git_heads", {})
        if manifest_heads.get("sglang") != expected["git_heads"]["sglang"]:
            raise RuntimeError(
                "campaign manifest drift for SGLang HEAD: "
                f"{manifest_heads.get('sglang')!r} != "
                f"{expected['git_heads']['sglang']!r}"
            )
        if (
            require_measurement_head
            and manifest_heads.get("kernel_harness")
            != expected["git_heads"]["kernel_harness"]
        ):
            raise RuntimeError(
                "current Kernel-Harness HEAD differs from the measurement HEAD; "
                "new/resumed GPU artifacts are forbidden. Post-collection audit may "
                "omit --require-measurement-head because artifact HEADs and exact "
                "harness hashes remain validated."
            )
        campaign_id = manifest.get("campaign_id")
        if not isinstance(campaign_id, str) or not campaign_id:
            raise RuntimeError("campaign manifest has no valid campaign_id")
        return manifest
    if not create:
        raise FileNotFoundError(f"campaign manifest is missing: {path}")
    manifest = {
        **expected,
        "campaign_id": f"glm52-w2-alignment-{secrets.token_hex(12)}",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write(path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _check_params(data: dict[str, Any], workload: str, where: str) -> None:
    decode_m, expected_m, assignments, mask = WORKLOAD_SPECS[workload]
    params = data["params"]
    required = {
        "decode_m": decode_m,
        "experts_per_rank": 32,
        "expert_slab": 1024,
        "expected_m": expected_m,
        "valid_assignments": assignments,
        "group_size": 128,
        "topk": 8,
        "k": 2048,
        "n": 6144,
    }
    observed = {key: params.get(key) for key in required}
    _expect(observed == required, f"{where}: workload parameters drifted: {observed}")
    _expect(data["masked_m"] == mask, f"{where}: masked_m drifted")


def _check_tensor_abi(data: dict[str, Any], where: str) -> None:
    observed = data["tensor_abi"]
    _expect(set(observed) == set(TENSOR_ABI), f"{where}: tensor ABI keys drifted")
    for key, (shape, stride, dtype, contiguous) in TENSOR_ABI.items():
        item = observed[key]
        _expect(item["shape"] == shape, f"{where}: {key} shape drifted")
        _expect(item["stride"] == stride, f"{where}: {key} stride drifted")
        _expect(item["dtype"] == dtype, f"{where}: {key} dtype drifted")
        _expect(item["device"] == "cuda:0", f"{where}: {key} device drifted")
        _expect(item["contiguous"] is contiguous, f"{where}: {key} contiguity drifted")


def _check_correctness(item: dict[str, Any], active_rows: int, where: str) -> None:
    _expect(item["finite"] is True, f"{where}: non-finite output")
    _expect(
        item["allclose_rtol_2e-2_atol_2e-2"] is True,
        f"{where}: allclose failed",
    )
    _expect(item["active_rows"] == active_rows, f"{where}: active row count drifted")
    contract = item["return_contract"]
    _expect(contract.get("matches") is True, f"{where}: return contract mismatch")
    _expect(
        contract["reference_type"] == contract["candidate_type"]
        and contract["reference_is_out"] == contract["candidate_is_out"],
        f"{where}: stock/candidate return contract differs",
    )


def _run_name(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).parts[1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"artifact is outside a profile run: {path}") from exc


def _expected_run_name(alignment: int, num_sms: int | None) -> str:
    base = "moe-w2-packed-baseline" if alignment == 128 else f"moe-w2-alignment{alignment}"
    return base if num_sms is None else f"moe-w2-alignment{alignment}-sms{num_sms}"


def _check_common_provenance(
    *,
    manifest: dict[str, Any],
    harness_sha: str,
    environment: dict[str, Any],
    run_name: str,
    where: str,
    harness_kind: str = "benchmark",
) -> None:
    _expect(
        harness_sha == manifest["harness_sha256"][harness_kind],
        f"{where}: {harness_kind} harness SHA drifted",
    )
    _expect(
        environment["deep_gemm_distribution_version"] == PINNED["deep_gemm_version"],
        f"{where}: DeepGEMM version drifted",
    )
    _expect(
        environment["deep_gemm_python_sha256"] == PINNED["deep_gemm_python_sha256"],
        f"{where}: DeepGEMM Python SHA drifted",
    )
    _expect(
        environment["deep_gemm_extension_sha256"] == PINNED["deep_gemm_extension_sha256"],
        f"{where}: DeepGEMM extension SHA drifted",
    )
    _expect(
        environment["deep_gemm_device_source_sha256"]
        == PINNED["deep_gemm_device_source_sha256"],
        f"{where}: DeepGEMM device-source SHA drifted",
    )
    _expect(
        Path(environment["deep_gemm_import"]).resolve().is_relative_to(DEEP_GEMM_ROOT),
        f"{where}: DeepGEMM import is outside pinned overlay",
    )
    _expect(
        Path(environment["sglang_import"]).resolve().is_relative_to(SGLANG_ROOT),
        f"{where}: SGLang import is outside isolated worktree",
    )
    _expect(environment["cuda_visible_devices"] == "0,1,2,3", f"{where}: lock guard missing")
    contract_env = environment["contract_environment"]
    _expect(contract_env["SGLANG_GLM52_OPT"] == "0", f"{where}: replacement was enabled")
    _expect(contract_env["SGLANG_DEEPGEMM_PDL"] == "true", f"{where}: PDL policy env drifted")
    for key in (
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE",
        "SGLANG_JIT_DEEPGEMM_FAST_WARMUP",
        "SGL_DG_USE_NVRTC",
        "DG_JIT_USE_NVRTC",
        "SGLANG_DEEPGEMM_SANITY_CHECK",
    ):
        _expect(contract_env[key] == "0", f"{where}: {key} must remain disabled")
    for key in (
        "DG_JIT_WITH_LINEINFO",
        "DG_JIT_PTXAS_VERBOSE",
        "DG_JIT_DUMP_ASM",
        "DG_PRINT_CONFIGS",
        "DG_USE_NVIDIA_TOOLS",
    ):
        _expect(contract_env[key] == "1", f"{where}: {key} evidence setting drifted")
    _expect(contract_env["SGLANG_ROOT"] == str(SGLANG_ROOT), f"{where}: SGLang root drifted")
    _expect(contract_env["DEEP_GEMM_ROOT"] == str(DEEP_GEMM_ROOT), f"{where}: overlay root drifted")
    _expect(environment["active_gpu"]["index"] == "0", f"{where}: wrong physical GPU")
    _expect(
        environment["kernel_harness_git"]["head"]
        == manifest["git_heads"]["kernel_harness"],
        f"{where}: Kernel-Harness HEAD drifted",
    )
    _expect(
        environment["sglang_git"]["head"] == manifest["git_heads"]["sglang"],
        f"{where}: SGLang HEAD drifted",
    )
    expected_cache = (ROOT / "profile" / run_name / "cache").resolve()
    _expect(
        Path(environment["dg_jit_cache_dir"]).resolve() == expected_cache
        and Path(environment["sglang_dg_cache_dir"]).resolve() == expected_cache,
        f"{where}: JIT cache is not isolated to {expected_cache}",
    )


def validate_result(
    path: Path,
    manifest: dict[str, Any],
    *,
    expected_workload: str | None = None,
    expected_alignment: int | None = None,
    expected_num_sms: int | None = None,
) -> tuple[dict[str, Any], list[float], list[float], list[float]]:
    data = json.loads(path.read_text())
    where = str(path)
    _expect(data.get("schema_version") == 2, f"{where}: wrong schema")
    _expect(data.get("benchmark") == "glm52_production_moe_w2_grouped_decode", f"{where}: wrong benchmark")
    _expect(data.get("campaign_id") == manifest["campaign_id"], f"{where}: mixed campaign")
    workload = data["workload"]
    _expect(workload in WORKLOAD_SPECS, f"{where}: unknown workload {workload}")
    if expected_workload is not None:
        _expect(workload == expected_workload, f"{where}: wrong workload")
    knobs = data["candidate_knobs"]
    alignment, num_sms = knobs["alignment"], knobs["num_sms"]
    if expected_alignment is not None:
        _expect(alignment == expected_alignment, f"{where}: wrong alignment")
    _expect(num_sms == expected_num_sms, f"{where}: wrong num_sms {num_sms}")
    run_name = _run_name(path)
    _expect(run_name == _expected_run_name(alignment, num_sms), f"{where}: run/config mismatch")
    _check_params(data, workload, where)
    _check_tensor_abi(data, where)
    _, expected_m, assignments, mask = WORKLOAD_SPECS[workload]
    _expect(data["masked_m_sum"] == assignments, f"{where}: mask sum drifted")
    _expect(data["masked_m_max"] == max(mask), f"{where}: mask max drifted")
    call = data["production_call_contract"]
    _expect(call["expected_m"] == expected_m, f"{where}: expected_m drifted")
    _expect(call["overlap_args"] is None and call["recipe_a"] is None and call["recipe_b"] is None, f"{where}: leaf call contract drifted")
    _expect(data["stock_knobs"]["alignment"] == 128, f"{where}: stock alignment drifted")
    _expect(data["stock_knobs"]["pdl"] is True, f"{where}: PDL was disabled")
    pdl_policy = data["stock_knobs"]["pdl_policy"]
    _expect(
        pdl_policy["requested"] is True
        and pdl_policy["active_during_setup_and_measurement"] is True
        and pdl_policy["active_before_restore"] is True
        and pdl_policy["restored"] is True,
        f"{where}: production PDL policy was not active and restored",
    )
    measurement = data["measurement_contract"]
    _expect(measurement == {
        "sessions": 3,
        "warmup": 5,
        "repeat": 30,
        "alternating_order": True,
        "gate": "median(reference_ms / candidate_ms) >= 1.03",
    }, f"{where}: measurement contract drifted")
    _check_correctness(data["correctness_before_timing"], assignments, f"{where}: pre")
    _check_correctness(data["correctness_after_timing_fresh_inputs"], assignments, f"{where}: fresh")
    _expect(data["fresh_input_storage_distinct"] is True, f"{where}: fresh storage was reused")

    sessions = data["sessions"]
    _expect(len(sessions) == 3, f"{where}: expected three sessions")
    references: list[float] = []
    candidates: list[float] = []
    ratios: list[float] = []
    for index, session in enumerate(sessions):
        _expect(session["index"] == index, f"{where}: session index drifted")
        ref = [float(value) for value in session["reference_ms"]]
        cand = [float(value) for value in session["candidate_ms"]]
        stored = [float(value) for value in session["paired_speedup"]]
        _expect(len(ref) == len(cand) == len(stored) == 30, f"{where}: session {index} is not 30 pairs")
        _expect(all(math.isfinite(value) and value > 0 for value in ref + cand + stored), f"{where}: invalid latency")
        computed = [r / c for r, c in zip(ref, cand)]
        _expect(all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(stored, computed)), f"{where}: stored ratios do not match raw pairs")
        gate = statistics.median(computed) >= 1.03
        _expect(session["passes_3pct_paired_median_gate"] is gate, f"{where}: session gate mismatch")
        _check_correctness(session["correctness_after_timing"], assignments, f"{where}: session {index}")
        references.extend(ref)
        candidates.extend(cand)
        ratios.extend(computed)
    aggregate_gate = statistics.median(ratios) >= 1.03
    _expect(data["aggregate_passes_3pct_gate"] is aggregate_gate, f"{where}: aggregate gate mismatch")
    _expect(math.isclose(data["aggregate_paired_speedup"]["median_ms"], statistics.median(ratios), rel_tol=1e-12), f"{where}: aggregate median mismatch")
    log_path = data.get("log_path")
    _expect(isinstance(log_path, str) and Path(log_path).is_file(), f"{where}: benchmark log is missing")
    _expect(Path(log_path).resolve().is_relative_to(ROOT / "profile" / run_name / "reports"), f"{where}: benchmark log escaped run")
    _check_common_provenance(
        manifest=manifest,
        harness_sha=data["harness_sha256"],
        environment=data["environment"],
        run_name=run_name,
        where=where,
    )
    return data, references, candidates, ratios


def validate_config_metadata(
    path: Path,
    manifest: dict[str, Any],
    *,
    workload: str,
    alignment: int,
    num_sms: int | None,
) -> dict[str, Any]:
    data = json.loads(path.read_text())
    where = str(path)
    _expect(data.get("schema_version") == 2, f"{where}: wrong config schema")
    _expect(data.get("probe") == "glm52_production_moe_w2_candidate_first_config", f"{where}: wrong config probe")
    _expect(data.get("campaign_id") == manifest["campaign_id"], f"{where}: mixed campaign")
    _expect(data["harness_sha256"] == manifest["harness_sha256"]["config_probe"], f"{where}: config harness drifted")
    _expect(data["workload"] == workload, f"{where}: wrong workload")
    _expect(data["alignment"] == alignment, f"{where}: wrong alignment")
    expected_sms = data["stock_num_sms"] if num_sms is None else num_sms
    _expect(data["num_sms"] == expected_sms, f"{where}: wrong SM allocation")
    _expect(data["stock_alignment"] == 128, f"{where}: stock alignment drifted")
    _expect(data["pdl"] is True, f"{where}: PDL disabled")
    pdl_policy = data["pdl_policy"]
    _expect(
        pdl_policy["requested"] is True
        and pdl_policy["active_during_setup_and_measurement"] is True
        and pdl_policy["active_before_restore"] is True
        and pdl_policy["restored"] is True,
        f"{where}: production PDL policy was not active and restored",
    )
    _check_params(data, workload, where)
    _check_tensor_abi(data, where)
    _expect(data["cuda_visible_devices"] == "0,1,2,3", f"{where}: lock guard missing")
    contract_env = data["contract_environment"]
    required_environment = {
        "SGLANG_ROOT": str(SGLANG_ROOT),
        "DEEP_GEMM_ROOT": str(DEEP_GEMM_ROOT),
        "SGLANG_GLM52_OPT": "0",
        "SGLANG_DEEPGEMM_PDL": "true",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
        "SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "0",
        "SGL_DG_USE_NVRTC": "0",
        "DG_JIT_USE_NVRTC": "0",
        "SGLANG_DEEPGEMM_SANITY_CHECK": "0",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "DG_JIT_WITH_LINEINFO": "1",
        "DG_JIT_PTXAS_VERBOSE": "1",
        "DG_JIT_DUMP_ASM": "1",
        "DG_PRINT_CONFIGS": "1",
        "DG_USE_NVIDIA_TOOLS": "1",
    }
    _expect(
        contract_env == required_environment,
        f"{where}: config-probe environment drifted: {contract_env}",
    )
    _expect(data["active_gpu"]["index"] == "0", f"{where}: wrong GPU")
    _expect(data["kernel_harness_git"]["head"] == manifest["git_heads"]["kernel_harness"], f"{where}: Kernel-Harness HEAD drifted")
    _expect(data["sglang_git"]["head"] == manifest["git_heads"]["sglang"], f"{where}: SGLang HEAD drifted")
    provenance = data["provenance"]
    _expect(provenance["deep_gemm_distribution_version"] == PINNED["deep_gemm_version"], f"{where}: DeepGEMM version drifted")
    _expect(provenance["sha256"] == {
        "python": PINNED["deep_gemm_python_sha256"],
        "extension": PINNED["deep_gemm_extension_sha256"],
        "device_source": PINNED["deep_gemm_device_source_sha256"],
    }, f"{where}: DeepGEMM hashes drifted")
    run_name = _expected_run_name(alignment, num_sms)
    expected_cache = (ROOT / "profile" / run_name / "cache").resolve()
    _expect(Path(data["dg_jit_cache_dir"]).resolve() == expected_cache, f"{where}: wrong JIT cache")
    _expect(Path(data["sglang_dg_cache_dir"]).resolve() == expected_cache, f"{where}: wrong SGLang cache")
    log_path = Path(data["config_log_path"]).resolve()
    _expect(log_path.is_file(), f"{where}: config log missing: {log_path}")
    _expect(log_path.is_relative_to(ROOT / "profile" / run_name / "reports"), f"{where}: config log escaped run")
    text = log_path.read_text(errors="replace")
    expected_m = WORKLOAD_SPECS[workload][1]
    pattern = re.compile(
        rf"GemmDesc\(gemm_type=2,.*?m=1024, n=6144, k=2048, num_groups=32,.*?"
        rf"num_sms={expected_sms},.*?expected_m={expected_m}, expected_n=6144, "
        rf"expected_k=2048, expected_num_groups=32\): GemmConfig\(layout=Layout\("
        rf"swap_ab=\d+, block_m={alignment},",
    )
    _expect(pattern.search(text) is not None, f"{where}: selected masked config line missing")
    return data


def validate_graph_result(
    path: Path,
    manifest: dict[str, Any],
    *,
    workload: str,
    alignment: int,
    num_sms: int | None,
) -> dict[str, Any]:
    data = json.loads(path.read_text())
    where = str(path)
    _expect(data.get("schema_version") == 2, f"{where}: wrong graph schema")
    _expect(
        data.get("check") == "glm52_production_moe_w2_graph_replay",
        f"{where}: wrong graph check identity",
    )
    _expect(data.get("campaign_id") == manifest["campaign_id"], f"{where}: mixed campaign")
    _expect(data.get("workload") == workload, f"{where}: wrong workload")
    _check_params(data, workload, where)
    _check_tensor_abi(data, where)

    knobs = data["candidate_knobs"]
    _expect(knobs["alignment"] == alignment, f"{where}: wrong alignment")
    _expect(knobs["num_sms"] == num_sms, f"{where}: wrong requested SM allocation")
    expected_active_sms = data["stock_knobs"]["num_sms"] if num_sms is None else num_sms
    _expect(
        knobs["active_num_sms"] == expected_active_sms,
        f"{where}: wrong active SM allocation",
    )
    run_name = _run_name(path)
    _expect(
        run_name == _expected_run_name(alignment, num_sms),
        f"{where}: graph run/config mismatch",
    )
    _expect(data["stock_knobs"]["alignment"] == 128, f"{where}: stock alignment drifted")
    _expect(data["stock_knobs"]["pdl"] is True, f"{where}: PDL disabled")
    pdl_policy = data["stock_knobs"]["pdl_policy"]
    _expect(
        pdl_policy["requested"] is True
        and pdl_policy["active_during_setup_and_measurement"] is True
        and pdl_policy["active_before_restore"] is True
        and pdl_policy["restored"] is True,
        f"{where}: production PDL policy was not active and restored",
    )

    assignments = WORKLOAD_SPECS[workload][2]
    _check_correctness(
        data["stock_candidate_correctness"],
        assignments,
        f"{where}: stock/candidate",
    )
    _expect(data["return_contracts"]["matches"] is True, f"{where}: graph return contract changed")
    _expect(data["capturing_during_launch"] is True, f"{where}: launch was not captured")
    _expect(data["deterministic"] is True, f"{where}: graph replay was non-deterministic")
    _expect(data["finite_active_output"] is True, f"{where}: graph output was non-finite")
    _expect(data["eager_graph_allclose"] is True, f"{where}: eager/graph mismatch")
    elapsed = [float(value) for value in data["elapsed_ms"]]
    _expect(data["replays"] == 30 and len(elapsed) == 30, f"{where}: expected 30 replays")
    _expect(
        all(math.isfinite(value) and value > 0 for value in elapsed),
        f"{where}: invalid graph latency",
    )
    ordered = sorted(elapsed)
    _expect(
        math.isclose(data["median_ms"], statistics.median(elapsed), rel_tol=1e-12),
        f"{where}: graph median mismatch",
    )
    _expect(
        math.isclose(data["p10_ms"], ordered[int(0.10 * len(ordered))], rel_tol=1e-12)
        and math.isclose(data["p90_ms"], ordered[int(0.90 * len(ordered))], rel_tol=1e-12),
        f"{where}: graph percentile mismatch",
    )
    log_path = data.get("log_path")
    _expect(isinstance(log_path, str) and Path(log_path).is_file(), f"{where}: graph log missing")
    _expect(
        Path(log_path).resolve().is_relative_to(ROOT / "profile" / run_name / "reports"),
        f"{where}: graph log escaped run",
    )
    _check_common_provenance(
        manifest=manifest,
        harness_sha=data["harness_sha256"],
        environment=data["environment"],
        run_name=run_name,
        where=where,
        harness_kind="graph",
    )
    return data


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def _summarize(args: argparse.Namespace, manifest: dict[str, Any]) -> int:
    rows = []
    provenance: dict[str, set[str]] = {
        "campaign_id": set(),
        "harness_sha256": set(),
        "deep_gemm_extension_sha256": set(),
        "deep_gemm_device_source_sha256": set(),
        "kernel_harness_head": set(),
        "sglang_head": set(),
        "active_gpu_uuid": set(),
    }
    for path in sorted(ROOT.glob("profile/moe-w2-*/analysis/microbench/*.json")):
        run_name = _run_name(path)
        if args.campaign == "alignment" and not ALIGNMENT_RUN.fullmatch(run_name):
            continue
        if args.campaign == "sms" and not SMS_RUN.fullmatch(run_name):
            continue
        raw = json.loads(path.read_text())
        num_sms = raw.get("candidate_knobs", {}).get("num_sms")
        data, reference, candidate, ratios = validate_result(
            path, manifest, expected_num_sms=num_sms
        )
        knobs = data["candidate_knobs"]
        env = data["environment"]
        provenance["campaign_id"].add(data["campaign_id"])
        provenance["harness_sha256"].add(data["harness_sha256"])
        provenance["deep_gemm_extension_sha256"].add(env["deep_gemm_extension_sha256"])
        provenance["deep_gemm_device_source_sha256"].add(env["deep_gemm_device_source_sha256"])
        provenance["kernel_harness_head"].add(env["kernel_harness_git"]["head"])
        provenance["sglang_head"].add(env["sglang_git"]["head"])
        provenance["active_gpu_uuid"].add(env["active_gpu"]["uuid"])
        rows.append({
            "artifact": str(path.relative_to(ROOT)),
            "run_name": run_name,
            "workload": data["workload"],
            "alignment": knobs["alignment"],
            "num_sms": knobs["num_sms"],
            "pairs": len(ratios),
            "reference_p50_ms": statistics.median(reference),
            "candidate_p50_ms": statistics.median(candidate),
            "paired_p10": _percentile(ratios, 0.10),
            "paired_p50": statistics.median(ratios),
            "paired_p90": _percentile(ratios, 0.90),
            "passes_3pct": statistics.median(ratios) >= 1.03,
            "correct_before": True,
            "correct_fresh": True,
        })
    _expect(bool(rows), f"no {args.campaign} paired artifacts found")
    if not args.allow_incomplete:
        if args.campaign == "alignment":
            expected = {(workload, alignment, None) for workload in WORKLOAD_SPECS for alignment in (16, 32, 64, 96, 128)}
        else:
            alignments = {row["alignment"] for row in rows}
            _expect(len(alignments) == 1, "SMS campaign mixed alignments")
            alignment = next(iter(alignments))
            expected = {(workload, alignment, sms) for workload in WORKLOAD_SPECS for sms in (116, 128, 136, 140, 144, 148)}
        observed = {(row["workload"], row["alignment"], row["num_sms"]) for row in rows}
        _expect(observed == expected and len(rows) == len(expected), f"incomplete {args.campaign} campaign: missing={sorted(expected-observed)} extra={sorted(observed-expected)}")
    inconsistent = {key: sorted(values) for key, values in provenance.items() if len(values) != 1}
    _expect(not inconsistent, f"mixed provenance in paired campaign: {inconsistent}")

    prefix = args.output_prefix or f"paired_{args.campaign}_summary"
    summary = {
        "schema_version": 3,
        "campaign": args.campaign,
        "manifest": str(args.manifest.resolve()),
        "provenance": {key: next(iter(values)) for key, values in provenance.items()},
        "rows": rows,
    }
    _atomic_write(EVIDENCE / f"{prefix}.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    columns = list(rows[0])
    csv_path = EVIDENCE / f"{prefix}.csv"
    temporary_csv = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    with temporary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_csv, csv_path)
    lines = [
        "# Paired production-W2 summary",
        "",
        "| Workload | Alignment | SMs | Pairs | Ref p50 (ms) | Cand p50 (ms) | Paired p10 | Paired p50 | Paired p90 | 3% gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {workload} | {alignment} | {num_sms} | {pairs} | "
            "{reference_p50_ms:.6f} | {candidate_p50_ms:.6f} | "
            "{paired_p10:.6f}x | {paired_p50:.6f}x | {paired_p90:.6f}x | "
            "{passes_3pct} |".format(**row)
        )
    _atomic_write(EVIDENCE / f"{prefix}.md", "\n".join(lines) + "\n")
    print(json.dumps({"artifacts": len(rows), "campaign": args.campaign, "output_prefix": prefix}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=("alignment", "sms", "all"), default="alignment")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--init-manifest", action="store_true")
    parser.add_argument(
        "--require-measurement-head",
        action="store_true",
        help="fail unless current Kernel-Harness HEAD is the manifest measurement HEAD",
    )
    parser.add_argument("--print-campaign-id", action="store_true")
    parser.add_argument("--validate-artifact", type=Path)
    parser.add_argument("--validate-config-metadata", type=Path)
    parser.add_argument("--validate-graph-artifact", type=Path)
    parser.add_argument("--expected-workload", choices=tuple(WORKLOAD_SPECS))
    parser.add_argument("--expected-alignment", type=int, choices=(16, 32, 64, 96, 128))
    parser.add_argument("--expected-num-sms", type=int)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output-prefix")
    args = parser.parse_args()
    manifest = _init_or_load_manifest(
        args.manifest,
        create=args.init_manifest,
        require_measurement_head=args.require_measurement_head,
    )
    if args.init_manifest or args.print_campaign_id:
        print(manifest["campaign_id"])
        return 0
    if args.validate_artifact is not None:
        _expect(args.expected_workload is not None and args.expected_alignment is not None, "artifact validation requires expected workload/alignment")
        validate_result(
            args.validate_artifact,
            manifest,
            expected_workload=args.expected_workload,
            expected_alignment=args.expected_alignment,
            expected_num_sms=args.expected_num_sms,
        )
        print(f"VALID {args.validate_artifact}")
        return 0
    if args.validate_config_metadata is not None:
        _expect(args.expected_workload is not None and args.expected_alignment is not None, "config validation requires expected workload/alignment")
        validate_config_metadata(
            args.validate_config_metadata,
            manifest,
            workload=args.expected_workload,
            alignment=args.expected_alignment,
            num_sms=args.expected_num_sms,
        )
        print(f"VALID {args.validate_config_metadata}")
        return 0
    if args.validate_graph_artifact is not None:
        _expect(
            args.expected_workload is not None and args.expected_alignment is not None,
            "graph validation requires expected workload/alignment",
        )
        validate_graph_result(
            args.validate_graph_artifact,
            manifest,
            workload=args.expected_workload,
            alignment=args.expected_alignment,
            num_sms=args.expected_num_sms,
        )
        print(f"VALID {args.validate_graph_artifact}")
        return 0
    if args.campaign == "all":
        raise RuntimeError("summarize alignment and SMS campaigns separately")
    return _summarize(args, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
