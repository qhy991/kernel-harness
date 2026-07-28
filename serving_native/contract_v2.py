"""Schema-v2 helpers for the serving-native benchmark contract.

This module is deliberately safe to import in GPU-free structural tests.  CUDA
bindings and torch profiler APIs are imported only by the functions that need
them after the scheduler wrapper has admitted a GPU command.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
MIN_REQUIRED_SERIES = 3
PERFORMANCE_THRESHOLD = 1.03
FORBIDDEN_GRAPH_NODE_MARKERS = (
    "MEMCPY",
    "MEMSET",
    "MEM_ALLOC",
    "MEM_FREE",
    "HOST",
)
FORBIDDEN_GRAPH_KERNEL_MARKERS = (
    "adapter",
    "aten::copy",
    "_to_copy",
    "contiguous",
    "layout_convert",
    "reformat",
)
CACHE_ENV_VARS = (
    "DG_JIT_CACHE_DIR",
    "SGLANG_DG_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_artifact(role: str, path: Path, *, kind: str | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{role} artifact not found: {resolved}")
    return {
        "role": role,
        "kind": kind or ("shared_object" if ".so" in resolved.name else "python"),
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def git_repository(path: Path) -> dict[str, Any]:
    resolved = path.resolve()

    def run(*args: str) -> str:
        proc = subprocess.run(
            ["git", "-C", str(resolved), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode:
            raise RuntimeError(
                f"git {' '.join(args)} failed for {resolved}: {proc.stderr.strip()}"
            )
        return proc.stdout.strip()

    status = run("status", "--porcelain=v1").splitlines()
    return {
        "path": str(resolved),
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status": status,
    }


def _module_file(module: Any) -> str | None:
    value = getattr(module, "__file__", None)
    if not value:
        return None
    try:
        resolved = Path(value).resolve()
    except (OSError, RuntimeError):
        return None
    # Torch exposes pseudo-modules such as torch.classes and torch.ops with
    # synthetic cwd-relative __file__ values.  They are not imported artifacts
    # and cannot be bound to a file hash or verified later.
    return str(resolved) if resolved.is_file() else None


def module_path_snapshot() -> dict[str, str]:
    paths: dict[str, str] = {}
    for name, module in tuple(sys.modules.items()):
        path = _module_file(module)
        if path:
            paths[name] = path
    return paths


def loaded_shared_objects() -> list[str]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return []
    paths: set[str] = set()
    for line in maps.read_text(errors="replace").splitlines():
        fields = line.split()
        if not fields:
            continue
        raw = fields[-1]
        if not raw.startswith("/") or ".so" not in Path(raw).name:
            continue
        path = Path(raw)
        if path.exists():
            paths.add(str(path.resolve()))
    return sorted(paths)


def _cache_roots() -> list[Path]:
    roots: list[Path] = []
    for variable in CACHE_ENV_VARS:
        raw = os.environ.get(variable)
        if not raw:
            continue
        path = Path(raw).expanduser().resolve()
        if path.is_dir() and path not in roots:
            roots.append(path)
    return roots


def _tree_state(root: Path) -> dict[str, tuple[int, int]]:
    state: dict[str, tuple[int, int]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        state[str(path.relative_to(root))] = (stat.st_size, stat.st_mtime_ns)
    return state


def runtime_state_snapshot(
    candidate_artifacts: Iterable[Path] = (),
) -> dict[str, Any]:
    artifact_state: dict[str, tuple[int, int, str]] = {}
    for path in candidate_artifacts:
        resolved = path.expanduser().resolve()
        if resolved.is_file():
            stat = resolved.stat()
            artifact_state[str(resolved)] = (
                stat.st_size,
                stat.st_mtime_ns,
                sha256_file(resolved),
            )
    return {
        "modules": module_path_snapshot(),
        "shared_objects": loaded_shared_objects(),
        "caches": {str(root): _tree_state(root) for root in _cache_roots()},
        "candidate_artifacts": artifact_state,
    }


def runtime_state_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    before_modules = before.get("modules", {})
    after_modules = after.get("modules", {})
    new_imports = [
        {"module": name, "path": path}
        for name, path in sorted(after_modules.items())
        if name not in before_modules
    ]
    before_sos = set(before.get("shared_objects", []))
    new_sos = sorted(set(after.get("shared_objects", [])) - before_sos)
    cache_changes: list[dict[str, Any]] = []
    before_caches = before.get("caches", {})
    for root, current in sorted(after.get("caches", {}).items()):
        previous = before_caches.get(root, {})
        for relative, state in sorted(current.items()):
            if previous.get(relative) != state:
                cache_changes.append(
                    {
                        "root": root,
                        "path": relative,
                        "before": previous.get(relative),
                        "after": state,
                    }
                )
        for relative, state in sorted(previous.items()):
            if relative not in current:
                cache_changes.append(
                    {
                        "root": root,
                        "path": relative,
                        "before": state,
                        "after": None,
                    }
                )
    artifact_changes: list[dict[str, Any]] = []
    before_artifacts = before.get("candidate_artifacts", {})
    after_artifacts = after.get("candidate_artifacts", {})
    for path in sorted(set(before_artifacts) | set(after_artifacts)):
        if before_artifacts.get(path) != after_artifacts.get(path):
            artifact_changes.append(
                {
                    "path": path,
                    "before": before_artifacts.get(path),
                    "after": after_artifacts.get(path),
                }
            )
    clean = not (new_imports or new_sos or cache_changes or artifact_changes)
    return {
        "phase": phase,
        "clean": clean,
        "new_imports": new_imports,
        "new_shared_objects": new_sos,
        "cache_changes": cache_changes,
        "candidate_artifact_changes": artifact_changes,
    }


def collect_import_provenance(
    candidate_module_name: str,
    *,
    relevant_roots: tuple[str, ...] = (
        "torch",
        "sglang",
        "deep_gemm",
        "serving_native",
    ),
) -> dict[str, Any]:
    python_modules: list[dict[str, str]] = []
    for name, module in sorted(sys.modules.items()):
        path = _module_file(module)
        if not path:
            continue
        if (
            name == candidate_module_name
            or name in relevant_roots
            or any(name.startswith(f"{root}.") for root in relevant_roots)
        ):
            python_modules.append(
                {
                    "module": name,
                    "path": path,
                    "kind": "shared_object" if ".so" in Path(path).name else "python",
                }
            )
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "modules": python_modules,
        "shared_objects": loaded_shared_objects(),
    }


def _nvidia_smi_rows() -> list[dict[str, Any]]:
    query = (
        "index,uuid,name,driver_version,"
        "clocks.current.sm,clocks.current.memory,pstate"
    )
    proc = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode:
        raise RuntimeError(f"nvidia-smi query failed: {proc.stderr.strip()}")
    rows: list[dict[str, Any]] = []
    for values in csv.reader(proc.stdout.splitlines(), skipinitialspace=True):
        if len(values) != 7:
            continue
        rows.append(
            {
                "physical_index": int(values[0]),
                "uuid": values[1].strip(),
                "name": values[2].strip(),
                "driver_version": values[3].strip(),
                "sm_clock_mhz": int(values[4]),
                "memory_clock_mhz": int(values[5]),
                "pstate": values[6].strip(),
            }
        )
    return rows


def _visible_physical_index(logical_index: int) -> int | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return logical_index
    entries = [item.strip() for item in visible.split(",") if item.strip()]
    if logical_index >= len(entries) or not entries[logical_index].isdigit():
        return None
    return int(entries[logical_index])


def collect_hardware_provenance(torch: Any, device: Any) -> dict[str, Any]:
    logical_index = int(device.index or 0)
    properties = torch.cuda.get_device_properties(device)
    physical_index = _visible_physical_index(logical_index)
    rows = _nvidia_smi_rows()
    row = next(
        (item for item in rows if item["physical_index"] == physical_index),
        None,
    )
    if row is None:
        raise RuntimeError(
            f"cannot resolve logical CUDA device {logical_index} to nvidia-smi"
        )
    property_uuid = getattr(properties, "uuid", None)
    normalized_property_uuid = None
    if property_uuid is not None:
        normalized_property_uuid = str(property_uuid)
        if not normalized_property_uuid.startswith("GPU-"):
            normalized_property_uuid = f"GPU-{normalized_property_uuid}"
    if (
        normalized_property_uuid is not None
        and normalized_property_uuid != row["uuid"]
    ):
        raise RuntimeError(
            "CUDA/nvidia-smi UUID mismatch: "
            f"{normalized_property_uuid} != {row['uuid']}"
        )
    return {
        "logical_index": logical_index,
        **row,
        "compute_capability": [
            int(properties.major),
            int(properties.minor),
        ],
        "multi_processor_count": int(properties.multi_processor_count),
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": str(torch.version.cuda),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def clock_sample() -> dict[str, Any]:
    visible_index = _visible_physical_index(0)
    row = next(
        (
            item
            for item in _nvidia_smi_rows()
            if item["physical_index"] == visible_index
        ),
        None,
    )
    if row is None:
        raise RuntimeError("cannot sample clocks for logical CUDA device 0")
    return {
        "captured_utc": utc_now(),
        "physical_index": row["physical_index"],
        "uuid": row["uuid"],
        "sm_clock_mhz": row["sm_clock_mhz"],
        "memory_clock_mhz": row["memory_clock_mhz"],
        "pstate": row["pstate"],
    }


def _cuda_value(result: tuple[Any, ...], operation: str) -> Any:
    error = result[0]
    if int(error) != 0:
        raise RuntimeError(f"{operation} failed with CUDA error {int(error)}")
    values = result[1:]
    if len(values) == 1:
        return values[0]
    return values


def _kernel_name(cuda_driver: Any, params: Any) -> str:
    for handle_name, getter_name in (
        ("kern", "cuKernelGetName"),
        ("func", "cuFuncGetName"),
    ):
        handle = getattr(params, handle_name, None)
        getter = getattr(cuda_driver, getter_name, None)
        if handle is None or getter is None or int(handle) == 0:
            continue
        result = getter(handle)
        if int(result[0]) == 0:
            value = result[1]
            return (
                value.decode("utf-8", "replace")
                if isinstance(value, bytes)
                else str(value)
            )
    return f"func:{int(getattr(params, 'func', 0))}"


def inspect_cuda_graph(raw_graph: int) -> dict[str, Any]:
    """Return CUDA graph node types and kernel identities without replaying it."""
    from cuda.bindings import driver as cuda_driver

    _, node_count = _cuda_value(
        cuda_driver.cuGraphGetNodes(raw_graph, 0),
        "cuGraphGetNodes(count)",
    )
    nodes, _ = _cuda_value(
        cuda_driver.cuGraphGetNodes(raw_graph, node_count),
        "cuGraphGetNodes(nodes)",
    )
    rendered: list[dict[str, Any]] = []
    for index, node in enumerate(nodes):
        node_type = _cuda_value(
            cuda_driver.cuGraphNodeGetType(node),
            "cuGraphNodeGetType",
        )
        item: dict[str, Any] = {
            "index": index,
            "type": node_type.name,
        }
        if "KERNEL" in node_type.name:
            params = _cuda_value(
                cuda_driver.cuGraphKernelNodeGetParams(node),
                "cuGraphKernelNodeGetParams",
            )
            item.update(
                {
                    "kernel": _kernel_name(cuda_driver, params),
                    "grid": [
                        int(params.gridDimX),
                        int(params.gridDimY),
                        int(params.gridDimZ),
                    ],
                    "block": [
                        int(params.blockDimX),
                        int(params.blockDimY),
                        int(params.blockDimZ),
                    ],
                    "shared_memory_bytes": int(params.sharedMemBytes),
                }
            )
        rendered.append(item)
    forbidden = graph_forbidden_nodes(rendered)
    return {
        "node_count": len(rendered),
        "nodes": rendered,
        "node_type_counts": graph_node_type_counts(rendered),
        "kernel_identities": graph_kernel_identities(rendered),
        "forbidden_nodes": forbidden,
    }


def graph_node_type_counts(nodes: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in nodes:
        node_type = str(node.get("type", ""))
        counts[node_type] = counts.get(node_type, 0) + 1
    return counts


def graph_kernel_identities(nodes: Iterable[dict[str, Any]]) -> list[str]:
    return [
        str(node["kernel"])
        for node in nodes
        if isinstance(node.get("kernel"), str) and bool(node["kernel"])
    ]


def graph_forbidden_nodes(
    nodes: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    forbidden: list[dict[str, Any]] = []
    for node in nodes:
        node_type = str(node.get("type", ""))
        kernel = str(node.get("kernel", "")).casefold()
        if any(marker in node_type for marker in FORBIDDEN_GRAPH_NODE_MARKERS) or any(
            marker in kernel for marker in FORBIDDEN_GRAPH_KERNEL_MARKERS
        ):
            forbidden.append(node)
    return forbidden


def profile_cuda_callable(torch: Any, fn: Callable[[], Any]) -> dict[str, Any]:
    """Capture one untimed CUDA execution and return its raw kernel events."""
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(activities=activities) as profiler:
        fn()
        torch.cuda.synchronize()
    events: list[dict[str, Any]] = []
    for event in profiler.events():
        device_type = str(getattr(event, "device_type", ""))
        if "cuda" not in device_type.lower():
            continue
        duration = getattr(event, "device_time_total", None)
        if duration is None:
            duration = getattr(event, "self_device_time_total", 0.0)
        events.append(
            {
                "name": str(event.name),
                "duration_us": float(duration or 0.0),
                "device_type": device_type,
            }
        )
    identities = sorted({event["name"] for event in events})
    if not identities:
        raise RuntimeError("CUDA profiler captured no device kernel identities")
    return {
        "captured": True,
        "events": events,
        "kernel_identities": identities,
    }


def nearest_rank(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(quantile * len(ordered) + 0.999999999) - 1),
    )
    return ordered[index]


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("latency summary requires at least one sample")
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "p95_ms": nearest_rank(values, 0.95),
    }
