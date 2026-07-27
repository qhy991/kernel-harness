#!/usr/bin/env python3
"""Validate the complete graph and edge-correctness evidence for one W2 leaf.

This is a post-collection, CPU-only audit.  It deliberately does not require
the current Kernel-Harness HEAD to equal the measurement HEAD: the campaign
manifest, artifact-recorded heads, and exact harness hashes are the authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SGLANG_ROOT = Path(
    "/home/qinhaiyan/glm52-goal-runs/07-moe_w2_decode_kernel/sglang"
).resolve()
DEEP_GEMM_ROOT = (SGLANG_ROOT / "build/deep-gemm-stock-0.1.4.post1").resolve()
EDGE_HARNESS = (
    ROOT / "profile/moe-w2-packed-baseline/harness/edge_mask_correctness.py"
).resolve()
EDGE_CACHE = (ROOT / "profile/moe-w2-edge-mask-correctness/cache").resolve()

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import summarize_paired as campaign  # noqa: E402


WORKLOADS = (
    "moe_w2_grouped_decode_m16",
    "moe_w2_grouped_decode_m32",
    "moe_w2_grouped_decode_m16_current_source_m5",
    "moe_w2_grouped_decode_m32_current_source_m9",
)
WORKLOAD_SUFFIX = {
    "moe_w2_grouped_decode_m16": "m16",
    "moe_w2_grouped_decode_m32": "m32",
    "moe_w2_grouped_decode_m16_current_source_m5": "m16_current_source_m5",
    "moe_w2_grouped_decode_m32_current_source_m9": "m32_current_source_m9",
}
EXPECTED_M = {
    "moe_w2_grouped_decode_m16": 4,
    "moe_w2_grouped_decode_m32": 8,
    "moe_w2_grouped_decode_m16_current_source_m5": 5,
    "moe_w2_grouped_decode_m32_current_source_m9": 9,
}
TENSOR_ABI = {
    "activation_fp8": ([32, 1024, 2048], [2097152, 2048, 1], "torch.float8_e4m3fn", True),
    "activation_scale": ([32, 1024, 4], [4096, 1, 1024], "torch.int32", False),
    "weight_fp8": ([32, 6144, 2048], [12582912, 2048, 1], "torch.float8_e4m3fn", True),
    "weight_scale": ([32, 6144, 4], [24576, 1, 6144], "torch.int32", False),
    "out": ([32, 1024, 6144], [6291456, 6144, 1], "torch.bfloat16", True),
    "masked_m": ([32], [1], "torch.int32", True),
}
CASE_COUNTS = {
    "front_loaded_boundaries": [31, 32, 33, 127, 1024] + [0] * 27,
    "scattered_boundaries_with_empty_ends": [
        0,
        1024,
        0,
        0,
        0,
        0,
        0,
        127,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        33,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        32,
        0,
        0,
        0,
        0,
        0,
        0,
        31,
        0,
    ],
    "small_tile_boundaries_15_16_17": [
        0,
        0,
        15,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        16,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        17,
        0,
        0,
        0,
        0,
    ],
}
REQUIRED_COUNTS = [0, 15, 16, 17, 31, 32, 33, 127, 1024]
INDEPENDENT_KEYS = {
    "activation_fp8",
    "activation_scale",
    "weight_fp8",
    "weight_scale",
    "out",
    "masked_m",
}
CONTENT_KEYS = {
    "activation_fp8",
    "activation_scale",
    "weight_fp8",
    "weight_scale",
    "masked_m",
}


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    expect(path.is_file() and not path.is_symlink(), f"not a regular artifact: {path}")
    data = json.loads(path.read_text())
    expect(isinstance(data, dict), f"{path}: expected a JSON object")
    return data


def atomic_publish_or_recheck(path: Path, text: str) -> str:
    """Create a deterministic summary once, or verify the existing bytes."""
    if path.exists() or path.is_symlink():
        expect(path.is_file() and not path.is_symlink(), f"invalid summary path: {path}")
        expect(path.read_text() == text, f"existing summary does not match audit: {path}")
        return "rechecked"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.tmp.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return "created"


def run_name(alignment: int) -> str:
    return "moe-w2-packed-baseline" if alignment == 128 else f"moe-w2-alignment{alignment}"


def graph_paths(alignment: int) -> dict[str, Path]:
    graph_root = ROOT / "profile" / run_name(alignment) / "analysis/graph"
    expected = {
        workload: graph_root / f"{WORKLOAD_SUFFIX[workload]}_alignment{alignment}.json"
        for workload in WORKLOADS
    }
    expect(graph_root.is_dir(), f"graph evidence directory is missing: {graph_root}")
    observed = {path.resolve() for path in graph_root.glob("*.json")}
    expected_set = {path.resolve() for path in expected.values()}
    expect(
        observed == expected_set,
        f"graph JSON set drifted: missing={sorted(map(str, expected_set-observed))}, "
        f"extra={sorted(map(str, observed-expected_set))}",
    )
    for path in expected.values():
        expect(path.is_file() and not path.is_symlink(), f"not a regular graph artifact: {path}")
    return expected


def parse_edge_mapping(value: str) -> tuple[str, Path]:
    workload, separator, raw_path = value.partition("=")
    if not separator or workload not in WORKLOADS or not raw_path:
        raise argparse.ArgumentTypeError(
            "--edge must be one of the four WORKLOAD=/absolute/or/relative.json mappings"
        )
    return workload, Path(raw_path).expanduser().resolve()


def edge_paths(args: argparse.Namespace) -> dict[str, Path]:
    if args.edge_root is not None:
        root = args.edge_root.expanduser().resolve()
        expect(root.is_dir(), f"edge evidence root is missing: {root}")
        expected = {
            workload: root / f"{workload}_alignment{args.alignment}.json"
            for workload in WORKLOADS
        }
        observed = {path.resolve() for path in root.glob("*.json")}
        expected_set = {path.resolve() for path in expected.values()}
        expect(
            observed == expected_set,
            f"edge JSON set drifted: missing={sorted(map(str, expected_set-observed))}, "
            f"extra={sorted(map(str, observed-expected_set))}",
        )
        return expected

    mappings = args.edge or []
    expect(len(mappings) == 4, "exactly four --edge WORKLOAD=PATH mappings are required")
    result: dict[str, Path] = {}
    for workload, path in mappings:
        expect(workload not in result, f"duplicate --edge mapping for {workload}")
        result[workload] = path
    expect(set(result) == set(WORKLOADS), "--edge mappings do not cover the exact workload set")
    expect(len({path for path in result.values()}) == 4, "edge mappings reuse an artifact path")
    return result


def check_tensor_abi(data: dict[str, Any], where: str) -> None:
    abi = data.get("tensor_abi")
    expect(isinstance(abi, dict) and set(abi) == set(TENSOR_ABI), f"{where}: ABI keys drifted")
    for key, (shape, stride, dtype, contiguous) in TENSOR_ABI.items():
        item = abi[key]
        expected = {
            "shape": shape,
            "stride": stride,
            "dtype": dtype,
            "device": "cuda:0",
            "contiguous": contiguous,
        }
        expect(item == expected, f"{where}: {key} packed ABI drifted: {item}")


def check_edge_provenance(
    data: dict[str, Any], manifest: dict[str, Any], expected_cache: Path, where: str
) -> tuple[str, tuple[tuple[str, str, str], ...]]:
    expect(data.get("harness_path") == str(EDGE_HARNESS), f"{where}: wrong edge harness path")
    expect(
        data.get("harness_sha256") == manifest["harness_sha256"]["edge"],
        f"{where}: edge harness SHA drifted",
    )
    optional_campaign = data.get("campaign_id")
    expect(
        optional_campaign in (None, manifest["campaign_id"]),
        f"{where}: edge artifact belongs to another campaign",
    )

    provenance = data.get("provenance", {})
    expect(provenance.get("sglang_root") == str(SGLANG_ROOT), f"{where}: SGLang root drifted")
    expect(
        Path(provenance.get("sglang_import", "")).resolve().is_relative_to(SGLANG_ROOT),
        f"{where}: SGLang import escaped the isolated worktree",
    )
    expect(provenance.get("deep_gemm_root") == str(DEEP_GEMM_ROOT), f"{where}: DeepGEMM root drifted")
    expect(
        Path(provenance.get("deep_gemm_import", "")).resolve().is_relative_to(DEEP_GEMM_ROOT),
        f"{where}: DeepGEMM import escaped the pinned overlay",
    )
    expect(
        provenance.get("deep_gemm_distribution_version")
        == campaign.PINNED["deep_gemm_version"],
        f"{where}: DeepGEMM version drifted",
    )
    expect(
        provenance.get("sha256")
        == {
            "python": campaign.PINNED["deep_gemm_python_sha256"],
            "extension": campaign.PINNED["deep_gemm_extension_sha256"],
            "device_source": campaign.PINNED["deep_gemm_device_source_sha256"],
        },
        f"{where}: DeepGEMM hashes drifted",
    )
    expect(
        Path(provenance.get("jit_cache", "")).resolve() == expected_cache,
        f"{where}: edge JIT cache drifted",
    )
    visible_names = provenance.get("visible_device_names")
    expect(
        isinstance(visible_names, list)
        and len(visible_names) == 4
        and all("B200" in name for name in visible_names),
        f"{where}: edge run did not expose four B200s",
    )

    environment = data.get("environment", {})
    required_environment = {
        "cuda_visible_devices": "0,1,2,3",
        "sglang_deepgemm_pdl": "1",
        "sglang_jit_deepgemm_precompile": "0",
        "sglang_jit_deepgemm_fast_warmup": "0",
        "sgl_dg_use_nvrtc": "0",
        "dg_jit_use_nvrtc": "0",
        "sglang_deepgemm_sanity_check": "0",
    }
    for key, expected in required_environment.items():
        expect(environment.get(key) == expected, f"{where}: {key} policy drifted")
    expect(
        environment.get("kernel_harness_git", {}).get("head")
        == manifest["git_heads"]["kernel_harness"],
        f"{where}: Kernel-Harness measurement HEAD drifted",
    )
    expect(
        environment.get("sglang_git", {}).get("head") == manifest["git_heads"]["sglang"],
        f"{where}: SGLang measurement HEAD drifted",
    )
    gpu = environment.get("gpu")
    expect(isinstance(gpu, list) and len(gpu) == 4, f"{where}: expected four GPU records")
    identity = []
    for index, item in enumerate(gpu):
        expect(item.get("index") == str(index), f"{where}: GPU index ordering drifted")
        expect(item.get("uuid") and "B200" in item.get("name", ""), f"{where}: invalid GPU identity")
        identity.append((item["index"], item["uuid"], item["name"]))
    return gpu[0]["uuid"], tuple(identity)


def check_pdl(data: dict[str, Any], where: str) -> None:
    stock = data.get("stock_config", {})
    expect(stock.get("alignment") == 128 and stock.get("pdl") is True, f"{where}: stock policy drifted")
    candidate = data.get("candidate_config", {})
    expect(
        candidate.get("num_sms") == stock.get("num_sms"),
        f"{where}: unselected SM reservation was used",
    )
    policy = data.get("deep_gemm_pdl_policy", {})
    expect(
        policy.get("applicable") is True
        and policy.get("requested") is True
        and policy.get("active_during_setup_and_measurement") is True
        and policy.get("active_before_restore") is True
        and policy.get("setter_called_after_cuda_device_assignment") is True
        and policy.get("restored") is True
        and policy.get("restored_value") == policy.get("original"),
        f"{where}: production PDL was not active and restored",
    )
    expect(
        Path(policy.get("module_path", "")).resolve().is_relative_to(DEEP_GEMM_ROOT),
        f"{where}: PDL module escaped pinned DeepGEMM",
    )


def check_case(case: dict[str, Any], name: str, expected_m: int, where: str) -> int:
    counts = CASE_COUNTS[name]
    expect(case.get("name") == name, f"{where}: case order/name drifted")
    expect(case.get("masked_m") == counts, f"{where}: boundary placement drifted")
    expect(case.get("masked_m_sum") == sum(counts), f"{where}: mask sum drifted")
    expect(case.get("masked_m_max") == max(counts), f"{where}: mask max drifted")
    expect(
        case.get("empty_experts") == [index for index, count in enumerate(counts) if count == 0],
        f"{where}: empty-expert set drifted",
    )
    expect(case.get("expected_m") == expected_m, f"{where}: expected_m drifted")
    independent = case.get("independent_storage", {})
    equal_contents = case.get("identical_input_contents", {})
    expect(set(independent) == INDEPENDENT_KEYS and all(independent.values()), f"{where}: storage alias")
    expect(set(equal_contents) == CONTENT_KEYS and all(equal_contents.values()), f"{where}: inputs differ")
    expect(
        case.get("nan_poisoned_active_before_launch")
        == {"reference": True, "candidate": True},
        f"{where}: active output was not poisoned",
    )
    expect(case.get("masked_m_unmodified") is True, f"{where}: masked_m was modified")
    active_rows = sum(counts)
    expect(case.get("active_rows_compared") == active_rows, f"{where}: active row count drifted")
    expect(
        case.get("active_values_compared") == active_rows * 6144,
        f"{where}: active value count drifted",
    )
    expect(
        case.get("finite_active_output") == {"reference": True, "candidate": True},
        f"{where}: non-finite active output",
    )
    expect(case.get("allclose_rtol_2e_2_atol_2e_2") is True, f"{where}: allclose failed")
    for key in ("max_abs", "max_rel"):
        value = case.get(key)
        expect(isinstance(value, (int, float)) and not isinstance(value, bool), f"{where}: missing {key}")
        expect(math.isfinite(float(value)) and float(value) >= 0, f"{where}: invalid {key}")
    contract = case.get("return_contract", {})
    expect(contract.get("matches") is True, f"{where}: return contract mismatch")
    expect(
        contract.get("reference") == contract.get("candidate")
        and set(contract.get("reference", {})) == {"type", "is_out", "is_none"},
        f"{where}: stock/candidate return semantics differ",
    )
    stream = case.get("stream")
    # cudaStream_t == 0 is the valid legacy default stream.  The edge harness
    # separately proves that the current stream is unchanged across both calls.
    expect(isinstance(stream, int) and not isinstance(stream, bool) and stream >= 0, f"{where}: invalid stream")
    expect(case.get("passed") is True, f"{where}: case did not pass")
    return stream


def validate_edge(
    path: Path, workload: str, alignment: int, manifest: dict[str, Any]
) -> tuple[dict[str, Any], str, tuple[tuple[str, str, str], ...]]:
    data = read_object(path)
    where = str(path)
    expect(data.get("schema_version") == 1, f"{where}: wrong edge schema")
    expect(
        data.get("check") == "glm52_production_moe_w2_edge_mask_correctness",
        f"{where}: wrong check identity",
    )
    expect(data.get("status") == "PASS" and data.get("passes_all_cases") is True, f"{where}: edge check failed")
    expect(data.get("workload") == workload, f"{where}: wrong workload")
    params = data.get("params", {})
    expected_params = {
        "experts_per_rank": 32,
        "expert_slab": 1024,
        "expected_m": EXPECTED_M[workload],
        "group_size": 128,
        "k": 2048,
        "n": 6144,
        "topk": 8,
    }
    expect(
        {key: params.get(key) for key in expected_params} == expected_params,
        f"{where}: production geometry drifted",
    )
    expect(data.get("candidate_config", {}).get("alignment") == alignment, f"{where}: wrong alignment")
    expect(data.get("required_mask_counts") == REQUIRED_COUNTS, f"{where}: required boundaries drifted")
    expect(data.get("correctness_scope") == "active rows only", f"{where}: correctness scope drifted")
    check_tensor_abi(data, where)
    check_pdl(data, where)
    active_uuid, gpu_identity = check_edge_provenance(data, manifest, EDGE_CACHE, where)

    cases = data.get("cases")
    expect(isinstance(cases, list) and len(cases) == 3, f"{where}: expected exactly three cases")
    streams = [
        check_case(case, name, EXPECTED_M[workload], f"{where}: {name}")
        for case, name in zip(cases, CASE_COUNTS)
    ]
    expect(len(set(streams)) == 1, f"{where}: current stream changed between edge cases")
    return data, active_uuid, gpu_identity


def artifact_record(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "campaign_id": data.get("campaign_id"),
        "kernel_harness_head": data["environment"]["kernel_harness_git"]["head"],
        "sglang_head": data["environment"]["sglang_git"]["head"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--alignment", type=int, choices=(16, 32, 64, 96, 128), required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--edge", action="append", type=parse_edge_mapping)
    source.add_argument(
        "--edge-root",
        type=Path,
        help=(
            "directory containing exactly four "
            "<workload>_alignment<alignment>.json files"
        ),
    )
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    manifest = campaign._init_or_load_manifest(
        manifest_path,
        create=False,
        require_measurement_head=False,
    )
    expect(manifest.get("campaign") == "alignment", "not an alignment campaign manifest")

    graphs = graph_paths(args.alignment)
    edges = edge_paths(args)
    graph_data: dict[str, dict[str, Any]] = {}
    edge_data: dict[str, dict[str, Any]] = {}
    active_uuids: set[str] = set()
    gpu_inventories: set[tuple[tuple[str, str, str], ...]] = set()

    for workload in WORKLOADS:
        graph = campaign.validate_graph_result(
            graphs[workload],
            manifest,
            workload=workload,
            alignment=args.alignment,
            num_sms=None,
        )
        graph_data[workload] = graph
        graph_gpu = graph["environment"]["active_gpu"]
        expect(graph_gpu.get("uuid"), f"{graphs[workload]}: active GPU UUID is missing")
        active_uuids.add(graph_gpu["uuid"])

        edge, edge_uuid, gpu_inventory = validate_edge(
            edges[workload], workload, args.alignment, manifest
        )
        edge_data[workload] = edge
        active_uuids.add(edge_uuid)
        gpu_inventories.add(gpu_inventory)

    expect(len(active_uuids) == 1, f"leaf evidence spans active GPUs: {sorted(active_uuids)}")
    expect(len(gpu_inventories) == 1, "edge evidence spans different four-GPU inventories")

    summary = {
        "schema_version": 1,
        "check": "glm52_production_moe_w2_leaf_evidence",
        "status": "PASS",
        "evidence_scope": "single_gpu_leaf_graph_and_edge_correctness_not_tp8_acceptance",
        "campaign_id": manifest["campaign_id"],
        "measurement_heads": manifest["git_heads"],
        "alignment": args.alignment,
        "num_sms": None,
        "run_name": run_name(args.alignment),
        "active_gpu_uuid": next(iter(active_uuids)),
        "edge_gpu_inventory": [
            {"index": index, "uuid": uuid, "name": name}
            for index, uuid, name in next(iter(gpu_inventories))
        ],
        "artifact_counts": {"graph": 4, "edge": 4, "edge_cases": 12},
        "validator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "graph": {
            workload: {
                **artifact_record(graphs[workload], graph_data[workload]),
                "replays": graph_data[workload]["replays"],
                "median_ms": graph_data[workload]["median_ms"],
                "p10_ms": graph_data[workload]["p10_ms"],
                "p90_ms": graph_data[workload]["p90_ms"],
            }
            for workload in WORKLOADS
        },
        "edge": {
            workload: {
                **artifact_record(edges[workload], edge_data[workload]),
                "campaign_binding": (
                    "campaign_id"
                    if edge_data[workload].get("campaign_id") is not None
                    else "manifest_measurement_heads_and_edge_harness_sha256"
                ),
                "case_names": list(CASE_COUNTS),
                "required_mask_counts": REQUIRED_COUNTS,
            }
            for workload in WORKLOADS
        },
    }
    output = manifest_path.parent / "leaf_validation_summary.json"
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    disposition = atomic_publish_or_recheck(output, rendered)
    print(f"VALID {output} ({disposition})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
