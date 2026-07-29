#!/usr/bin/env python3
"""Build and attest the two exact GLM-5.2 MoE W2 decode identities.

This tool is deliberately CPU-only. It asks the task DeepGEMM DSO to construct
the actual generated sources, compiles them with nvcc, audits the final binary,
and pre-populates the exact runtime JIT cache. It never queries or initializes a
CUDA device.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


MIN_FREE_BYTES = 8 * 1024**3
EXPECTED_HINTS = (4, 5, 8, 9)
EXPECTED_SYMBOL = "infini_kernel_glm52_moe_w2_decode_bm16_auto"
ROLE_NAMES = {
    "stock": "sm100_m_grouped_fp8_fp4_gemm_masked_1d1d",
    "candidate": EXPECTED_SYMBOL,
}
PTXAS_FLAGS = (
    "-std=c++20 --diag-suppress=39,161,174,177,186,940 "
    "--ptxas-options=--register-usage-level=10 "
    "--ptxas-options=--verbose,--warn-on-local-memory-usage"
)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value)
    else:
        path.write_bytes(value)


def _check_free(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"free disk {free} is below the required {MIN_FREE_BYTES} bytes at {path}"
        )


def _run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    stdout_path: Path | None = None,
) -> str:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if stdout_path is not None:
        _write(stdout_path, completed.stdout)
    if completed.returncode != 0:
        rendered = " ".join(shlex.quote(part) for part in argv)
        raise RuntimeError(
            f"command failed with exit {completed.returncode}: {rendered}\n"
            f"{completed.stdout}"
        )
    return completed.stdout


def _git_sha(root: Path) -> str:
    return _run(["git", "rev-parse", "HEAD"], cwd=root).strip()


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        if not path.is_file():
            continue
        digest.update(str(relative).encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _copy_package(source: Path, packages_root: Path) -> tuple[Path, str]:
    package_hash = _tree_hash(source)
    parent = packages_root / package_hash
    destination = parent / "deep_gemm"
    if destination.exists():
        if _tree_hash(destination) != package_hash:
            raise RuntimeError(f"content-addressed package drift: {destination}")
        return parent, package_hash

    temporary_parent = Path(
        tempfile.mkdtemp(prefix=".package.", dir=packages_root)
    )
    temporary = temporary_parent / "deep_gemm"
    shutil.copytree(
        source,
        temporary,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "_C_build"),
    )
    if _tree_hash(temporary) != package_hash:
        raise RuntimeError("package tree changed during content-addressed copy")
    try:
        os.rename(temporary_parent, parent)
    except FileExistsError:
        shutil.rmtree(temporary_parent)
        if not destination.is_dir() or _tree_hash(destination) != package_hash:
            raise
    return parent, package_hash


def _fnv1a(data: bytes, seed: int) -> int:
    value = seed
    for byte in data:
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


def _splitmix64(value: int) -> int:
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (value ^ (value >> 31)) & 0xFFFFFFFFFFFFFFFF


def _deepgemm_digest(value: str) -> str:
    data = value.encode()
    state_0 = _fnv1a(data, 0xC6A4A7935BD1E995)
    state_1 = _fnv1a(data, 0x9E3779B97F4A7C15)
    return f"{_splitmix64(state_0):016x}{_splitmix64(state_1):016x}"


def _compiler_identity(nvcc: Path, package: Path) -> tuple[str, str]:
    version_output = _run([str(nvcc), "--version"])
    match = re.search(r"release (\d+)\.(\d+)", version_output)
    if match is None:
        raise RuntimeError("could not parse nvcc release")
    signature = f"NVCC{match.group(1)}.{match.group(2)}"
    flags = (
        f"{PTXAS_FLAGS} -I{package / 'include'} --gpu-architecture=sm_100f "
        "--compiler-options=-fPIC,-O3,-fconcepts,-Wno-deprecated-declarations,-Wno-abi "
        "-O3 --expt-relaxed-constexpr --expt-extended-lambda"
    )
    return signature, flags


def _extract_entry_symbols(symbols: str) -> list[str]:
    result = []
    for line in symbols.splitlines():
        if line.startswith("STT_FUNC") and "STO_ENTRY" in line:
            result.append(line.rsplit(" ", 1)[-1])
    return result


def _resource_record(resource: str, symbol: str) -> dict[str, int]:
    match = re.search(
        rf"Function {re.escape(symbol)}:\s*\n\s*"
        r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",
        resource,
    )
    if match is None:
        raise RuntimeError(f"could not parse resource usage for {symbol}")
    return {
        "registers": int(match.group(1)),
        "stack_bytes": int(match.group(2)),
        "static_shared_bytes": int(match.group(3)),
        "local_bytes": int(match.group(4)),
    }


def _instruction_record(sass: str) -> dict[str, Any]:
    counts: collections.Counter[str] = collections.Counter(
        re.findall(r"/\*[0-9a-fA-F]+\*/\s+([A-Z][A-Z0-9_.]+)", sass)
    )

    def total(prefixes: tuple[str, ...]) -> int:
        return sum(
            count
            for mnemonic, count in counts.items()
            if mnemonic.startswith(prefixes)
        )

    selected = {
        mnemonic: count
        for mnemonic, count in sorted(counts.items())
        if mnemonic.startswith(
            (
                "UTC",
                "UTMA",
                "UCGABAR",
                "BAR",
                "DEPBAR",
                "MEMBAR",
            )
        )
    }
    return {
        "selected_mnemonics": selected,
        "tmem_tcgen05_lowered": total(("UTC",)),
        "tma_load": total(("UTMALDG",)),
        "tma_store": total(("UTMASTG",)),
        "tma_control": total(("UTMAC",)),
        "mbarrier_cluster": total(("UCGABAR", "UTCBAR")),
        "barrier_and_fence": total(("BAR", "DEPBAR", "MEMBAR")),
        "one_sm_instructions": sum(
            count for mnemonic, count in counts.items() if ".1CTA" in mnemonic
        ),
        "two_sm_instructions": sum(
            count for mnemonic, count in counts.items() if ".2CTA" in mnemonic
        ),
    }


def _assert_config(role: str, configs: dict[int, dict[str, Any]]) -> None:
    expected_block_m = 128 if role == "stock" else 16
    expected_stages = 8 if role == "stock" else 12
    for hint, config in configs.items():
        expected = {
            "candidate": role == "candidate",
            "expected_m": hint,
            "compiled_dims": "nk",
            "num_groups": 32,
            "m": 1024,
            "n": 6144,
            "k": 2048,
            "block_m": expected_block_m,
            "block_n": 128,
            "block_k": 128,
            "cluster_m": 1,
            "cluster_n": 2,
            "num_stages": expected_stages,
            "num_sms": 148,
            "num_threads": 256,
            "num_non_epilogue_threads": 128,
            "num_epilogue_threads": 128,
            "swizzle_a_mode": 128,
            "swizzle_b_mode": 128,
            "swizzle_cd_mode": 128,
            "pdl_required": True,
            "recipe": [1, 1, 128],
            "disable_ue8m0_cast": True,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise RuntimeError(
                    f"{role} expected-M {hint} config mismatch for {key}: "
                    f"{config.get(key)!r} != {value!r}"
                )


def _install_jit_cache(
    cache_root: Path,
    *,
    name: str,
    digest: str,
    artifacts: dict[str, Path],
) -> Path:
    parent = cache_root / "cache"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f"kernel.{name}.{digest}"
    required = ("kernel.cu", "kernel.cubin", "kernel.ptx", "kernel.sass")
    if destination.exists():
        for filename in required:
            if _sha256(destination / filename) != _sha256(artifacts[filename]):
                raise RuntimeError(f"existing JIT cache entry drift: {destination}")
        return destination

    temporary = Path(tempfile.mkdtemp(prefix=".prepopulate.", dir=parent))
    for filename in required:
        shutil.copy2(artifacts[filename], temporary / filename)
    os.rename(temporary, destination)
    return destination


def _write_role_bundle(
    bundles_root: Path,
    role: str,
    manifest: dict[str, Any],
    artifacts: dict[str, Path],
) -> tuple[Path, str]:
    manifest_bytes = _json_bytes(manifest)
    manifest_digest = _sha256_bytes(manifest_bytes)
    destination = bundles_root / role / manifest_digest
    if destination.exists():
        if _sha256(destination / "manifest.json") != manifest_digest:
            raise RuntimeError(f"bundle manifest drift: {destination}")
        return destination, manifest_digest

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{role}.", dir=destination.parent)
    )
    for filename, source in artifacts.items():
        shutil.copy2(source, temporary / filename)
    _write(temporary / "manifest.json", manifest_bytes)
    _write(
        temporary / "manifest.json.sha256",
        f"{manifest_digest}  manifest.json\n",
    )
    os.rename(temporary, destination)
    return destination, manifest_digest


def _compile_role(
    *,
    role: str,
    source: str,
    configs: dict[int, dict[str, Any]],
    package: Path,
    dso: Path,
    prebuild_root: Path,
    bundles_root: Path,
    jit_cache_root: Path,
    nvcc: Path,
    cuobjdump: Path,
    signature: str,
    flags: str,
    shared: dict[str, Any],
) -> tuple[Path, str, dict[str, Any]]:
    _check_free(prebuild_root)
    source_sha = _sha256_bytes(source.encode())
    work = prebuild_root / role / source_sha[:20]
    work.mkdir(parents=True, exist_ok=True)
    source_path = work / "kernel.cu"
    cubin_path = work / "kernel.cubin"
    ptx_path = work / "kernel.ptx"
    sass_path = work / "kernel.sass"
    symbols_path = work / "symbols.txt"
    resource_path = work / "resource.txt"
    ptxas_path = work / "ptxas.log"
    commands_path = work / "build_commands.json"
    _write(source_path, source)

    split_flags = shlex.split(flags)
    cubin_command = [
        str(nvcc),
        str(source_path),
        "-cubin",
        "-o",
        str(cubin_path),
        *split_flags,
    ]
    ptx_command = [
        str(nvcc),
        str(source_path),
        "-ptx",
        "-o",
        str(ptx_path),
        *split_flags,
    ]
    ptxas_output = _run(cubin_command, cwd=work)
    _write(ptxas_path, ptxas_output)
    _run(ptx_command, cwd=work)
    sass = _run([str(cuobjdump), "--dump-sass", str(cubin_path)])
    _write(sass_path, sass)
    symbols = _run([str(cuobjdump), "-symbols", str(cubin_path)])
    _write(symbols_path, symbols)
    resource = _run(
        [str(cuobjdump), "--dump-resource-usage", str(cubin_path)]
    )
    _write(resource_path, resource)

    commands = {
        "cubin": cubin_command,
        "ptx": ptx_command,
        "sass": [str(cuobjdump), "--dump-sass", str(cubin_path)],
        "symbols": [str(cuobjdump), "-symbols", str(cubin_path)],
        "resource": [
            str(cuobjdump),
            "--dump-resource-usage",
            str(cubin_path),
        ],
    }
    _write(commands_path, _json_bytes(commands))

    entries = _extract_entry_symbols(symbols)
    if len(entries) != 1:
        raise RuntimeError(f"{role} must contain exactly one entry, got {entries}")
    symbol = entries[0]
    if role == "candidate" and symbol != EXPECTED_SYMBOL:
        raise RuntimeError(f"candidate symbol mismatch: {symbol}")
    if role == "candidate" and not symbol.startswith(
        "infini_kernel_glm52_moe_w2_decode"
    ):
        raise RuntimeError(f"candidate symbol lacks required prefix: {symbol}")

    config = configs[EXPECTED_HINTS[0]]
    resources = _resource_record(resource, symbol)
    instructions = _instruction_record(sass)
    if resources["stack_bytes"] or resources["local_bytes"]:
        raise RuntimeError(f"{role} has stack/local memory: {resources}")
    spill_match = re.search(
        r"(\d+) bytes spill stores, (\d+) bytes spill loads", ptxas_output
    )
    if spill_match is None or spill_match.groups() != ("0", "0"):
        raise RuntimeError(f"{role} spill audit failed:\n{ptxas_output}")
    if instructions["two_sm_instructions"] == 0:
        raise RuntimeError(f"{role} contains no 2-CTA instruction evidence")
    if instructions["tma_load"] == 0 or instructions["tma_store"] == 0:
        raise RuntimeError(f"{role} contains no TMA load/store evidence")

    name = ROLE_NAMES[role]
    kernel_signature = f"{name}$${signature}$${flags}$${source}"
    jit_digest = _deepgemm_digest(kernel_signature)
    artifacts = {
        "kernel.cu": source_path,
        "kernel.cubin": cubin_path,
        "kernel.ptx": ptx_path,
        "kernel.sass": sass_path,
        "symbols.txt": symbols_path,
        "resource.txt": resource_path,
        "ptxas.log": ptxas_path,
        "build_commands.json": commands_path,
    }
    jit_path = _install_jit_cache(
        jit_cache_root,
        name=name,
        digest=jit_digest,
        artifacts=artifacts,
    )
    artifact_hashes = {
        filename: _sha256(path) for filename, path in artifacts.items()
    }
    source_macros = {
        key: value
        for key, value in re.findall(
            r"^#define DEEP_GEMM_GLM52_MOE_W2_DECODE_([A-Z0-9_]+)\s+(.+)$",
            source,
            flags=re.MULTILINE,
        )
    }
    if role == "candidate":
        if source_macros.get("BLOCK_M") != "16":
            raise RuntimeError("candidate generated source does not encode BM16")
        if source_macros.get("NUM_STAGES") != "12":
            raise RuntimeError(
                "candidate generated source does not encode actual auto-stage 12"
            )
    else:
        template = re.search(
            r"\b0,\s*6144,\s*\n?\s*2048,\s*128,\s*128,\s*\n?\s*128,\s*32,",
            source,
        )
        if template is None:
            raise RuntimeError("stock generated source does not encode BM128")

    manifest = {
        "schema": "glm52-moe-w2-decode-binary-bundle-v1",
        "role": role,
        "identity_count": 1,
        "kernel_name": name,
        "entry_symbol": symbol,
        "jit_digest": jit_digest,
        "jit_cache_directory": str(jit_path),
        "source_sha256": source_sha,
        "source_include_hash": source.splitlines()[0].removeprefix(
            "// Includes' hash value: "
        ),
        "artifact_sha256": artifact_hashes,
        "generated_source_macros": source_macros,
        "config_by_expected_m": {
            str(hint): configs[hint] for hint in EXPECTED_HINTS
        },
        "launch": {
            "grid": [148, 1, 1],
            "block": [config["num_threads"], 1, 1],
            "cluster": [config["cluster_n"], 1, 1],
            "dynamic_shared_bytes": config["smem_size"],
            "pdl": True,
            "sm_budget": config["num_sms"],
            "topology": "two_sm_cluster",
        },
        "resources": {
            **resources,
            "dynamic_shared_bytes": config["smem_size"],
            "spill_store_bytes": 0,
            "spill_load_bytes": 0,
        },
        "instructions": instructions,
        "compiler": shared["compiler"],
        "bases": shared["bases"],
        "dso": shared["dso"],
        "package": shared["package"],
        "dispatch_keys": [
            {
                "op": "moe_w2",
                "phase": "decode",
                "forward_m": 16 if hint in (4, 5) else 32,
                "expected_m": hint,
                "jit_key": jit_digest,
            }
            for hint in EXPECTED_HINTS
        ],
        "abi": {
            "groups": 32,
            "slab_m": 1024,
            "k": 2048,
            "n": 6144,
            "a": {
                "dtype": "float8_e4m3fn",
                "shape": [32, 1024, 2048],
                "contiguous": True,
                "storage_offset": 0,
            },
            "a_scale": {
                "dtype": "int32",
                "shape": [32, 1024, 4],
                "stride": [4096, 1, 1024],
                "storage_offset": 0,
            },
            "b": {
                "dtype": "float8_e4m3fn",
                "shape": [32, 6144, 2048],
                "contiguous": True,
                "storage_offset": 0,
            },
            "b_scale": {
                "dtype": "int32",
                "shape": [32, 6144, 4],
                "stride": [24576, 1, 6144],
                "storage_offset": 0,
            },
            "out": {
                "dtype": "bfloat16",
                "shape": [32, 1024, 6144],
                "contiguous": True,
                "caller_owned": True,
                "storage_offset": 0,
            },
            "mask": {
                "dtype": "int32",
                "shape": [32],
                "device_resident": True,
                "host_read": False,
                "storage_offset": 0,
            },
            "recipe": [1, 1, 128],
            "disable_ue8m0_cast": True,
            "compiled_dims": "nk",
            "return": None,
        },
    }
    bundle, digest = _write_role_bundle(
        bundles_root, role, manifest, artifacts
    )
    return bundle, digest, manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepgemm-root", type=Path, required=True)
    parser.add_argument("--sglang-root", type=Path, required=True)
    parser.add_argument("--harness-root", type=Path, required=True)
    parser.add_argument("--staged-package", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jit-cache-root", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, default=Path("/usr/local/cuda/bin/nvcc"))
    parser.add_argument(
        "--cuobjdump",
        type=Path,
        default=Path("/usr/local/cuda/bin/cuobjdump"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError(
            "CPU bundle build requires CUDA_VISIBLE_DEVICES='' or '-1'"
        )
    if os.environ.get("DG_JIT_CPP_STANDARD", "20") != "20":
        raise RuntimeError("bundle identity requires DG_JIT_CPP_STANDARD=20")
    os.environ["DG_JIT_PTXAS_VERBOSE"] = "1"
    os.environ["DG_JIT_PTXAS_CHECK"] = "1"
    os.environ["DG_JIT_DUMP_PTX"] = "1"
    os.environ["DG_JIT_DUMP_SASS"] = "1"

    roots = (
        args.deepgemm_root.resolve(),
        args.sglang_root.resolve(),
        args.harness_root.resolve(),
        args.staged_package.resolve(),
        args.output_root.resolve(),
        args.jit_cache_root.resolve(),
    )
    (
        deepgemm_root,
        sglang_root,
        harness_root,
        staged_package,
        output_root,
        jit_cache_root,
    ) = roots
    _check_free(output_root)
    packages_root = output_root / "packages"
    packages_root.mkdir(parents=True, exist_ok=True)
    package_parent, package_hash = _copy_package(
        staged_package, packages_root
    )
    package = package_parent / "deep_gemm"
    dso = package / "_C.so"
    if not dso.is_file():
        raise RuntimeError(f"staged candidate DSO is missing: {dso}")

    sys.path.insert(0, str(package_parent))
    deep_gemm = importlib.import_module("deep_gemm")
    torch = importlib.import_module("torch")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU source audit unexpectedly initialized CUDA")
    if Path(deep_gemm.__file__).resolve().parent != package.resolve():
        raise RuntimeError(
            f"wrong DeepGEMM package imported: {deep_gemm.__file__}"
        )

    generated: dict[str, str] = {}
    configs_by_role: dict[str, dict[int, dict[str, Any]]] = {}
    for role, candidate in (("stock", False), ("candidate", True)):
        sources = {
            hint: deep_gemm.glm52_moe_w2_decode_generated_source(
                candidate, hint
            )
            for hint in EXPECTED_HINTS
        }
        hashes = {_sha256_bytes(value.encode()) for value in sources.values()}
        if len(hashes) != 1:
            raise RuntimeError(
                f"{role} expected-M hints generated multiple JIT identities"
            )
        generated[role] = sources[EXPECTED_HINTS[0]]
        configs = {
            hint: json.loads(
                deep_gemm.glm52_moe_w2_decode_config_json(candidate, hint)
            )
            for hint in EXPECTED_HINTS
        }
        _assert_config(role, configs)
        configs_by_role[role] = configs

    nvcc = args.nvcc.resolve()
    cuobjdump = args.cuobjdump.resolve()
    signature, flags = _compiler_identity(nvcc, package)
    bases = {
        "deepgemm": _git_sha(deepgemm_root),
        "cutlass": _git_sha(deepgemm_root / "third-party/cutlass"),
        "fmt": _git_sha(deepgemm_root / "third-party/fmt"),
        "sglang": _git_sha(sglang_root),
        "kernel_harness": _git_sha(harness_root),
    }
    if bases["deepgemm"] != "edcf77b276965de8f03cdc47c23f01b08bf7c7ab":
        raise RuntimeError(f"unexpected DeepGEMM base: {bases['deepgemm']}")
    if bases["sglang"] != "83d313104d089bcd2af26b28453ff880f1e6a80b":
        raise RuntimeError(f"unexpected SGLang base: {bases['sglang']}")

    compiler_record = {
        "kind": "nvcc",
        "signature": signature,
        "path": str(nvcc),
        "sha256": _sha256(nvcc),
        "flags": flags,
        "flags_sha256": _sha256_bytes(flags.encode()),
        "target": "sm_100f",
    }
    shared = {
        "compiler": compiler_record,
        "bases": bases,
        "dso": {
            "path": str(dso),
            "sha256": _sha256(dso),
            "build_mode": "task-local staged package, CPU-only TVM-FFI build",
            "normalized_replay_command": [
                "env",
                f"PATH={harness_root / '.venv/bin'}:$PATH",
                "TVM_FFI_CUDA_ARCH_LIST=100",
                f"SGL_DEEP_GEMM_BUILD_DIR={staged_package.parent}",
                f"SGL_DEEP_GEMM_DIST_DIR={staged_package.parents[2] / 'dist'}",
                "SGL_DEEP_GEMM_SKIP_WHEEL=1",
                "bash",
                str(deepgemm_root / "build_sgl_deep_gemm.sh"),
            ],
        },
        "package": {
            "path": str(package),
            "tree_sha256": package_hash,
            "content_address": package_parent.name,
        },
    }
    shared["dso"]["normalized_replay_command_sha256"] = _sha256_bytes(
        _json_bytes(shared["dso"]["normalized_replay_command"])
    )

    prebuild_root = output_root / "cpu_prebuild"
    bundles_root = output_root / "bundles"
    results: dict[str, tuple[Path, str, dict[str, Any]]] = {}
    for role in ("stock", "candidate"):
        results[role] = _compile_role(
            role=role,
            source=generated[role],
            configs=configs_by_role[role],
            package=package,
            dso=dso,
            prebuild_root=prebuild_root,
            bundles_root=bundles_root,
            jit_cache_root=jit_cache_root,
            nvcc=nvcc,
            cuobjdump=cuobjdump,
            signature=signature,
            flags=flags,
            shared=shared,
        )

    stock_bundle, stock_digest, stock_manifest = results["stock"]
    candidate_bundle, candidate_digest, candidate_manifest = results[
        "candidate"
    ]
    for key in ("compiler", "bases", "dso", "package"):
        if stock_manifest[key] != candidate_manifest[key]:
            raise RuntimeError(f"stock/candidate shared identity mismatch: {key}")
    if stock_manifest["source_sha256"] == candidate_manifest["source_sha256"]:
        raise RuntimeError("stock and candidate sources unexpectedly match")
    if stock_manifest["entry_symbol"] == candidate_manifest["entry_symbol"]:
        raise RuntimeError("stock and candidate entry symbols unexpectedly match")

    ready = {
        "schema": "glm52-moe-w2-decode-ready-v1",
        "cpu_only_identity_gate": True,
        "cuda_initialized": False,
        "material_identities": 2,
        "stock_bundle": str(stock_bundle),
        "stock_manifest_sha256": stock_digest,
        "candidate_bundle": str(candidate_bundle),
        "candidate_manifest_sha256": candidate_digest,
        "runtime_package_parent": str(package_parent),
        "runtime_package_tree_sha256": package_hash,
        "runtime_dso_sha256": _sha256(dso),
        "jit_cache_root": str(jit_cache_root),
        "same_source_control": {
            "deepgemm": bases["deepgemm"],
            "cutlass": bases["cutlass"],
            "fmt": bases["fmt"],
            "compiler_signature": signature,
            "compiler_flags_sha256": compiler_record["flags_sha256"],
            "dso_sha256": _sha256(dso),
        },
        "candidate_symbol_prefix": "infini_kernel_glm52_moe_w2_decode",
        "candidate_symbol": EXPECTED_SYMBOL,
        "automatic_stage_selected": 12,
        "stock_stage": 8,
        "stock_block_m": 128,
        "candidate_block_m": 16,
        "expected_m_hints": list(EXPECTED_HINTS),
        "pdl_required": True,
        "sm_budget": 148,
        "production_default": False,
    }
    ready_bytes = _json_bytes(ready)
    ready_digest = _sha256_bytes(ready_bytes)
    ready_dir = output_root / "ready" / ready_digest
    if ready_dir.exists():
        if _sha256(ready_dir / "READY.json") != ready_digest:
            raise RuntimeError(f"READY drift: {ready_dir}")
    else:
        ready_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=".ready.", dir=ready_dir.parent)
        )
        _write(temporary / "READY.json", ready_bytes)
        _write(
            temporary / "READY.json.sha256",
            f"{ready_digest}  READY.json\n",
        )
        os.rename(temporary, ready_dir)

    print(
        json.dumps(
            {
                "ready": str(ready_dir / "READY.json"),
                "ready_sha256": ready_digest,
                "stock_bundle": str(stock_bundle),
                "candidate_bundle": str(candidate_bundle),
                "runtime_package_parent": str(package_parent),
                "jit_cache_root": str(jit_cache_root),
                "free_bytes": shutil.disk_usage(output_root).free,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
