"""Leased-GPU owner for independent stock and API-v1 W13 runtimes."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


BASE_COMMIT = "731e7c7a97d269e4b9f482ea18d0e709a948f293"
CANDIDATE_COMMIT = "87e0359edbb461181d3bba218442132007b9a738"
CANDIDATE_DIFF_SHA256 = (
    "465c8373c0a37970225a0e93267b6c399431b23e22cf35b4511db2308df98092"
)
STOCK_TREE_SHA256 = (
    "917592ab68ea0608c9be33208c2c609bc7f20bd9b1603f32743dd0d1ae03d0ed"
)
CANDIDATE_TREE_SHA256 = (
    "d682daa65b8ba0ac3846d766910b8c751e0568fe62087084271bb354e46c49e4"
)
VARIANT_CONFIGS = {
    "bm16_2sm": (16, 128, 128, 12, 2),
    "bm16_1sm": (16, 128, 128, 11, 1),
}
EXPECTED_M_VALUES = (4, 5, 8, 9)
REQUIRED_PDL = True
REQUIRED_NUM_SMS = 148
REQUIRED_TC_UTIL = 100

_A_SHAPE = (32, 1024, 6144)
_A_STRIDE = (6291456, 6144, 1)
_AS_SHAPE = (32, 1024, 12)
_AS_STRIDE = (12288, 1, 1024)
_B_SHAPE = (32, 4096, 6144)
_B_STRIDE = (25165824, 6144, 1)
_BS_SHAPE = (32, 4096, 12)
_BS_STRIDE = (49152, 1, 4096)
_OUT_SHAPE = (32, 1024, 4096)
_OUT_STRIDE = (4194304, 4096, 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _load_package(package: Path, module_name: str) -> ModuleType:
    init_py = package / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        module_name,
        init_py,
        submodule_search_locations=[str(package)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load W13 package from {init_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    module.__path__ = [str(package)]  # type: ignore[attr-defined]
    module.__package__ = module_name
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _validate_manifest(
    torch: Any,
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 3:
        raise RuntimeError("W13 build manifest schema mismatch")
    source = manifest.get("source", {})
    expected_source = {
        "base_commit": BASE_COMMIT,
        "candidate_commit": CANDIDATE_COMMIT,
        "candidate_diff_sha256": CANDIDATE_DIFF_SHA256,
        "stock_source_tree_sha256": STOCK_TREE_SHA256,
        "candidate_source_tree_sha256": CANDIDATE_TREE_SHA256,
    }
    actual_source = {key: source.get(key) for key in expected_source}
    if actual_source != expected_source:
        raise RuntimeError(
            f"W13 manifest source mismatch: {actual_source} != {expected_source}"
        )
    build = manifest.get("build", {})
    if (
        build.get("torch") != torch.__version__
        or build.get("torch_cuda") != torch.version.cuda
        or build.get("cuda_arch") != "10.0a"
        or build.get("jit_compiler") != "nvcc"
        or build.get("stock_candidate_command_identical") is not True
        or build.get("elf_symbol_binding") != "Bsymbolic"
        or build.get("elf_symbol_visibility") != "hidden"
    ):
        raise RuntimeError("W13 build/runtime contract mismatch")
    records: dict[str, dict[str, Any]] = {}
    for variant, expected_commit, expected_tree in (
        ("stock", BASE_COMMIT, STOCK_TREE_SHA256),
        ("candidate", CANDIDATE_COMMIT, CANDIDATE_TREE_SHA256),
    ):
        record = manifest.get("variants", {}).get(variant)
        if (
            not isinstance(record, dict)
            or record.get("commit") != expected_commit
            or record.get("source_tree_sha256") != expected_tree
            or record.get("normalized_build_plan_sha256")
            != build.get("normalized_build_plan_sha256")
        ):
            raise RuntimeError(f"W13 {variant} build record mismatch")
        package = Path(str(record.get("package", ""))).resolve()
        shared_object = Path(str(record.get("shared_object", ""))).resolve()
        build_ninja = Path(str(record.get("build_ninja", ""))).resolve()
        jit_cache = Path(str(record.get("jit_cache", ""))).resolve()
        if (
            package / "_C.so" != shared_object
            or not shared_object.is_file()
            or _sha256(shared_object) != record.get("shared_object_sha256")
            or not (package / "__init__.py").is_file()
            or _sha256(package / "__init__.py") != record.get("package_init_sha256")
            or not build_ninja.is_file()
            or _sha256(build_ninja) != record.get("build_ninja_sha256")
            or not jit_cache.is_dir()
        ):
            raise RuntimeError(f"W13 {variant} artifact identity mismatch")
        records[variant] = record
    return manifest, records["stock"], records["candidate"]


def _read_runtime_state(module: ModuleType) -> dict[str, Any]:
    return {
        "pdl": bool(module.get_pdl()),
        "num_sms": int(module.get_num_sms()),
        "tc_util": int(module.get_tc_util()),
    }


def _set_runtime_state(module: ModuleType, label: str) -> dict[str, Any]:
    module.set_pdl(REQUIRED_PDL)
    module.set_num_sms(REQUIRED_NUM_SMS)
    module.set_tc_util(REQUIRED_TC_UTIL)
    actual = _read_runtime_state(module)
    expected = {
        "pdl": REQUIRED_PDL,
        "num_sms": REQUIRED_NUM_SMS,
        "tc_util": REQUIRED_TC_UTIL,
    }
    if actual != expected:
        raise RuntimeError(f"W13 {label} state mismatch: {actual} != {expected}")
    return actual


def _prove_runtime_state_independence(
    stock: ModuleType,
    candidate: ModuleType,
) -> dict[str, Any]:
    required = {
        "pdl": REQUIRED_PDL,
        "num_sms": REQUIRED_NUM_SMS,
        "tc_util": REQUIRED_TC_UTIL,
    }
    mutations = {
        "pdl": ("set_pdl", False),
        "num_sms": ("set_num_sms", REQUIRED_NUM_SMS - 1),
        "tc_util": ("set_tc_util", REQUIRED_TC_UTIL - 1),
    }
    proof: dict[str, Any] = {}
    for field, (setter, mutation) in mutations.items():
        field_proof: dict[str, Any] = {}
        for mutated_name, mutated, other_name, other in (
            ("stock", stock, "candidate", candidate),
            ("candidate", candidate, "stock", stock),
        ):
            getattr(mutated, setter)(mutation)
            mutated_value = _read_runtime_state(mutated)[field]
            other_value = _read_runtime_state(other)[field]
            if mutated_value != mutation or other_value != required[field]:
                raise RuntimeError(
                    "W13 runtime globals alias: "
                    f"{field} {mutated_name}={mutated_value} "
                    f"{other_name}={other_value}"
                )
            getattr(mutated, setter)(required[field])
            field_proof[f"mutate_{mutated_name}"] = {
                "mutated_value": mutated_value,
                f"{other_name}_unchanged": other_value,
                "restored": _read_runtime_state(mutated)[field],
            }
        proof[field] = field_proof
    return proof


def _allocate_warm_inputs(torch: Any, device: Any) -> dict[str, Any]:
    def empty_strided(shape, stride, dtype):
        value = torch.empty_strided(shape, stride, device=device, dtype=dtype)
        value.zero_()
        return value

    return {
        "a": empty_strided(_A_SHAPE, _A_STRIDE, torch.float8_e4m3fn),
        "a_scale": empty_strided(_AS_SHAPE, _AS_STRIDE, torch.int32),
        "b": empty_strided(_B_SHAPE, _B_STRIDE, torch.float8_e4m3fn),
        "b_scale": empty_strided(_BS_SHAPE, _BS_STRIDE, torch.int32),
        "out": empty_strided(_OUT_SHAPE, _OUT_STRIDE, torch.bfloat16),
        "masked_m": torch.zeros((32,), device=device, dtype=torch.int32),
    }


def _launch_stock(module: ModuleType, tensors: dict[str, Any], expected_m: int) -> None:
    tensors["masked_m"].fill_(expected_m)
    tensors["out"].fill_(float("nan"))
    returned = module.fp8_m_grouped_gemm_nt_masked(
        (tensors["a"], tensors["a_scale"]),
        (tensors["b"], tensors["b_scale"]),
        tensors["out"],
        tensors["masked_m"],
        expected_m,
        compiled_dims="nk",
        disable_ue8m0_cast=True,
    )
    if returned is not None:
        raise RuntimeError("same-source stock changed the None return contract")


def _module_identity(
    record: dict[str, Any],
    cache_snapshot: dict[str, str],
) -> dict[str, Any]:
    return {
        "package": str(Path(record["package"]).resolve()),
        "package_init_sha256": record["package_init_sha256"],
        "shared_object": str(Path(record["shared_object"]).resolve()),
        "shared_object_sha256": record["shared_object_sha256"],
        "jit_cache": str(Path(record["jit_cache"]).resolve()),
        "jit_artifacts": cache_snapshot,
    }


class W13Runtime:
    """Bind exact stock and one current SGLang API-v1 candidate provider."""

    def __init__(
        self,
        torch_module: Any,
        device: Any,
        *,
        manifest_path: Path,
        variant: str,
    ) -> None:
        self.torch = torch_module
        self.device = device
        self.variant = variant
        self.config = VARIANT_CONFIGS.get(variant)
        if self.config is None:
            raise ValueError(f"unsupported W13 candidate variant: {variant}")
        gpu_id = int(device.index or 0)
        if int(torch_module.cuda.current_device()) != gpu_id:
            raise RuntimeError(
                "W13 harness initialized before assigned GPU became current"
            )
        if torch_module.cuda.get_device_capability(gpu_id) != (10, 0):
            raise RuntimeError("W13 harness requires an sm_100 B200")

        self.manifest_path = manifest_path.expanduser().resolve()
        manifest, stock_record, candidate_record = _validate_manifest(
            torch_module, self.manifest_path
        )
        stock_cache = Path(stock_record["jit_cache"]).resolve()
        candidate_cache = Path(candidate_record["jit_cache"]).resolve()
        if stock_cache == candidate_cache:
            raise RuntimeError("W13 stock and candidate cache roots alias")

        provider_path_text = os.environ.get(
            "SGLANG_GLM52_HOTSPOT_MODULE", ""
        ).strip()
        if not provider_path_text:
            raise RuntimeError("W13 benchmark requires SGLANG_GLM52_HOTSPOT_MODULE")
        provider_path = Path(provider_path_text).expanduser().resolve()
        expected_provider_name = {
            "bm16_2sm": "provider_bm16_2sm.py",
            "bm16_1sm": "provider_bm16_1sm.py",
        }[variant]
        if not provider_path.is_file() or provider_path.name != expected_provider_name:
            raise RuntimeError(
                f"W13 provider {provider_path} does not match {variant}"
            )

        saved = {
            name: os.environ.get(name)
            for name in (
                "DG_JIT_CACHE_DIR",
                "SGLANG_DG_CACHE_DIR",
                "DG_JIT_USE_NVRTC",
                "SGL_DG_USE_NVRTC",
                "DG_JIT_DUMP_PTX",
                "DG_JIT_DUMP_SASS",
                "DG_JIT_PTXAS_VERBOSE",
                "DG_JIT_PTXAS_CHECK",
                "SGLANG_GLM52_OPT",
                "SGLANG_GLM52_OPT_PROFILE",
                "SGLANG_GLM52_OPT_OPS",
                "SGLANG_GLM52_OPT_M_BUCKETS",
                "SGLANG_GLM52_HOTSPOT_MODULE",
            )
        }
        tensors = None
        try:
            os.environ.update(
                {
                    "DG_JIT_CACHE_DIR": str(stock_cache),
                    "SGLANG_DG_CACHE_DIR": str(stock_cache),
                    "DG_JIT_USE_NVRTC": "0",
                    "SGL_DG_USE_NVRTC": "0",
                    "DG_JIT_DUMP_PTX": "1",
                    "DG_JIT_DUMP_SASS": "1",
                    "DG_JIT_PTXAS_VERBOSE": "1",
                    "DG_JIT_PTXAS_CHECK": "0",
                }
            )
            self.stock = _load_package(
                Path(stock_record["package"]).resolve(),
                f"deep_gemm_w13_stock_harness_{os.getpid()}",
            )
            stock_state = _set_runtime_state(self.stock, "same-source stock")
            tensors = _allocate_warm_inputs(torch_module, device)
            for expected_m in EXPECTED_M_VALUES:
                _launch_stock(self.stock, tensors, expected_m)
            torch_module.cuda.synchronize(device)
            stock_snapshot = _snapshot(stock_cache)
            if not stock_snapshot:
                raise RuntimeError("same-source stock cache is empty")

            compile_utils = importlib.import_module(
                "sglang.srt.layers.deep_gemm_wrapper.compile_utils"
            )
            compile_utils._ENABLE_JIT_DEEPGEMM_PRECOMPILE = False

            # Stock W2 in the containing region remains the installed module.
            import deep_gemm

            installed_state = _set_runtime_state(
                deep_gemm, "installed downstream stock"
            )
            if installed_state != stock_state:
                raise RuntimeError("installed W2 runtime state differs from W13 stock")

            os.environ.update(
                {
                    "SGLANG_GLM52_OPT": "1",
                    "SGLANG_GLM52_OPT_PROFILE": "hotspot_candidates",
                    "SGLANG_GLM52_OPT_OPS": "moe_w13",
                    "SGLANG_GLM52_OPT_M_BUCKETS": "moe_gate_proj:16|32",
                    "SGLANG_GLM52_HOTSPOT_MODULE": str(provider_path),
                }
            )
            from sglang.srt.layers.glm52_opt import hotspot_provider

            hotspot_provider._reset_hotspot_provider_for_tests()
            if not hotspot_provider.initialize_hotspot_provider(gpu_id=gpu_id):
                raise RuntimeError("W13 API-v1 provider was not initialized")
            provider_state = hotspot_provider.provider_state()
            provider_module = sys.modules[provider_state["module_name"]]
            provider_object = getattr(provider_module, "_PROVIDER", None)
            if (
                provider_object is None
                or tuple(provider_object.config) != self.config
                or provider_module.PROVIDER_INFO.get("git_commit")
                != CANDIDATE_COMMIT
                or not provider_module.PROVIDER_INFO.get("name", "").startswith(
                    "infini_kernel_glm52_moe_w13_decode"
                )
            ):
                raise RuntimeError("W13 API-v1 provider identity/config mismatch")
            self.candidate = provider_object._module
            if self.candidate is None:
                raise RuntimeError("W13 provider candidate module is absent")
            candidate_state = _read_runtime_state(self.candidate)
            independence = _prove_runtime_state_independence(
                self.stock, self.candidate
            )
            candidate_snapshot = _snapshot(candidate_cache)
            provider_snapshot = provider_object.identity.get("jit_artifacts")
            if candidate_snapshot != provider_snapshot:
                raise RuntimeError("provider/candidate cache identity differs")

            probe = self.manifest_path.parent / (
                f"unbound-harness-stock-probe-{os.getpid()}"
            )
            if probe.exists():
                raise RuntimeError(f"W13 cache probe already exists: {probe}")
            os.environ["DG_JIT_CACHE_DIR"] = str(probe)
            os.environ["SGLANG_DG_CACHE_DIR"] = str(probe)
            _launch_stock(self.stock, tensors, EXPECTED_M_VALUES[0])
            torch_module.cuda.synchronize(device)
            if (
                probe.exists()
                or _snapshot(stock_cache) != stock_snapshot
                or _snapshot(candidate_cache) != candidate_snapshot
            ):
                raise RuntimeError("W13 cache ownership changed after freeze")

            self._provider_callback = hotspot_provider.run_moe_masked
            self.identity = {
                "manifest": str(self.manifest_path),
                "manifest_sha256": _sha256(self.manifest_path),
                "manifest_schema": manifest["schema_version"],
                "variant": variant,
                "config": list(self.config),
                "provider": {
                    "path": str(provider_path),
                    "sha256": _sha256(provider_path),
                    "state": provider_state,
                    "identity": dict(provider_object.identity),
                },
                "runtime_state": {
                    "installed_downstream": installed_state,
                    "stock": stock_state,
                    "candidate": candidate_state,
                },
                "state_independence": independence,
                "modules": {
                    "stock": _module_identity(stock_record, stock_snapshot),
                    "candidate": _module_identity(
                        candidate_record, candidate_snapshot
                    ),
                },
                "broad_precompile_enabled": bool(
                    compile_utils._ENABLE_JIT_DEEPGEMM_PRECOMPILE
                ),
                "jit_use_nvrtc": False,
                "candidate_call_path": (
                    "sglang.glm52_opt.hotspot_provider.run_moe_masked"
                    " -> API-v1 provider moe_w13 -> exact DeepGEMM symbol"
                ),
            }
        finally:
            if tensors is not None:
                del tensors
            torch_module.cuda.empty_cache()
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def stock_launcher(self, lhs, rhs, out, masked_m, expected_m):
        return self.stock.fp8_m_grouped_gemm_nt_masked(
            lhs,
            rhs,
            out,
            masked_m,
            expected_m,
            compiled_dims="nk",
            disable_ue8m0_cast=True,
        )

    def candidate_launcher(self, lhs, rhs, out, masked_m, expected_m):
        return self._provider_callback(
            "moe_gate_proj",
            lhs=lhs,
            rhs=rhs,
            out=out,
            masked_m=masked_m,
            expected_m=expected_m,
        )
