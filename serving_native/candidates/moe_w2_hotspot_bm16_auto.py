"""Fail-closed current-source GLM-5.2 MoE W2 BM16-auto candidate."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

CANDIDATE_IDENTITY = (
    "glm52-moe-w2-decode-bm16-auto-vector-valid-store-postrun-counters-v5"
)
DECLARED_FALLBACK = False
KERNEL_SYMBOL = "infini_kernel_glm52_moe_w2_decode_bm16_auto"
HOTSPOT_GOAL = "moe_w2_decode_bm16_auto_v1"

_HARNESS_ROOT = Path(__file__).resolve().parents[2]
_PROVIDER = (
    _HARNESS_ROOT
    / "serving_native"
    / "providers"
    / "moe_w2_decode_bm16_auto.py"
).resolve()
_IDENTITY_ROOT = Path(
    "/home/qinhaiyan/glm52-hotspot-goal-runs/cache/"
    "moe_w2_decode/identity_v1"
)
_READY_DIR = (
    _IDENTITY_ROOT
    / "ready"
    / "17a5e23c0bd3cac16d88a1054047c4488fd9ca46b34fcca25560f83fdca7858b"
)
_READY = _READY_DIR / "READY.json"
_READY_SIGNATURE = _READY_DIR / "READY.json.sha256"
_STOCK_DIR = (
    _IDENTITY_ROOT
    / "bundles"
    / "stock"
    / "5cbda917dfc2be33362a3fcf9a0a7a7a240edc03db352b03a6beba20a2df4fb4"
)
_CANDIDATE_DIR = (
    _IDENTITY_ROOT
    / "bundles"
    / "candidate"
    / "4b2eac068b797b3d798c3bc24ecdf1b4c72284868a1ef3a565137bcb6ce32734"
)
_PACKAGE_PARENT = (
    _IDENTITY_ROOT
    / "packages"
    / "d0d260979de692d86cbc4df8c80283f879c4218e72cc77fefae31bcc5825e2a2"
)
_PACKAGE = _PACKAGE_PARENT / "deep_gemm"
_DG_CACHE = Path(
    "/home/qinhaiyan/glm52-hotspot-goal-runs/cache/moe_w2_decode/deepgemm"
)
_DSO_SHA256 = "395655ab609ef5037fe3bb93f4a2d813c047f1265fd5f30554948dd8d0e51780"

ARTIFACT_PATHS = tuple(
    str(path)
    for path in (
        _PROVIDER,
        _READY,
        _READY_SIGNATURE,
        _STOCK_DIR / "manifest.json",
        _STOCK_DIR / "manifest.json.sha256",
        _CANDIDATE_DIR / "manifest.json",
        _CANDIDATE_DIR / "manifest.json.sha256",
    )
)

_STATE: dict[str, Any] = {}
_PROCESS_STATE: dict[str, Any] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"identity document is not an object: {path}")
    return value


def _verify_signature(path: Path, signature: Path) -> str:
    fields = signature.read_text().strip().split()
    if len(fields) != 2 or fields[1] != path.name:
        raise RuntimeError(f"invalid detached SHA-256 signature: {signature}")
    digest = _sha256(path)
    if fields[0] != digest:
        raise RuntimeError(f"SHA-256 signature mismatch: {path}")
    return digest


def _verify_bundle(role: str, root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    digest = _verify_signature(
        manifest_path,
        root / "manifest.json.sha256",
    )
    if digest != root.name:
        raise RuntimeError(f"{role} manifest is not content addressed")
    manifest = _read_json(manifest_path)
    if manifest.get("role") != role:
        raise RuntimeError(f"{role} manifest role mismatch")
    for relative, expected in manifest.get("artifact_sha256", {}).items():
        path = root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"{role} frozen artifact drifted: {path}")
    return manifest


def _verify_identity(runtime) -> dict[str, Any]:
    ready_sha256 = _verify_signature(_READY, _READY_SIGNATURE)
    if ready_sha256 != _READY_DIR.name:
        raise RuntimeError("READY record is not content addressed")
    ready = _read_json(_READY)
    stock = _verify_bundle("stock", _STOCK_DIR)
    candidate = _verify_bundle("candidate", _CANDIDATE_DIR)
    expected_pair = (
        runtime.workload.params.get("decode_m"),
        runtime.workload.params.get("expected_m"),
    )
    if (
        runtime.workload.params.get("hotspot_goal") != HOTSPOT_GOAL
        or runtime.workload.family
        not in {"moe_grouped_masked", "moe_compute_region"}
        or expected_pair not in {(16, 4), (16, 5), (32, 8), (32, 9)}
    ):
        raise RuntimeError(
            f"candidate does not own workload {runtime.workload.name}"
        )
    if (
        ready.get("candidate_symbol") != KERNEL_SYMBOL
        or ready.get("candidate_block_m") != 16
        or ready.get("automatic_stage_selected") != 12
        or ready.get("stock_block_m") != 128
        or ready.get("stock_stage") != 8
        or ready.get("sm_budget") != 148
        or ready.get("pdl_required") is not True
        or ready.get("material_identities") != 2
        or ready.get("production_default") is not False
    ):
        raise RuntimeError("READY constants do not match the selected candidate")
    if (
        stock.get("entry_symbol") == KERNEL_SYMBOL
        or candidate.get("entry_symbol") != KERNEL_SYMBOL
        or stock.get("source_sha256") == candidate.get("source_sha256")
        or stock.get("resources", {}).get("spill_load_bytes") != 0
        or candidate.get("resources", {}).get("spill_load_bytes") != 0
        or stock.get("resources", {}).get("spill_store_bytes") != 0
        or candidate.get("resources", {}).get("spill_store_bytes") != 0
        or stock.get("resources", {}).get("registers") != 36
        or candidate.get("resources", {}).get("registers") != 43
        or stock.get("resources", {}).get("stack_bytes") != 0
        or candidate.get("resources", {}).get("stack_bytes") != 0
        or stock.get("resources", {}).get("local_bytes") != 0
        or candidate.get("resources", {}).get("local_bytes") != 0
        or candidate.get("instructions", {}).get("tma_store") != 2
        or candidate.get("instructions", {}).get("two_sm_instructions") != 25
    ):
        raise RuntimeError("stock/candidate binary identities do not separate")
    for hint in ("4", "5", "8", "9"):
        stock_config = stock.get("config_by_expected_m", {}).get(hint, {})
        candidate_config = candidate.get("config_by_expected_m", {}).get(
            hint,
            {},
        )
        if (
            stock_config.get("block_m") != 128
            or stock_config.get("num_stages") != 8
            or candidate_config.get("block_m") != 16
            or candidate_config.get("num_stages") != 12
        ):
            raise RuntimeError(f"generated template mismatch for expected-M {hint}")
    if (
        not (_PACKAGE / "_C.so").is_file()
        or _sha256(_PACKAGE / "_C.so") != _DSO_SHA256
        or ready.get("runtime_dso_sha256") != _DSO_SHA256
    ):
        raise RuntimeError("runtime package DSO differs from READY")
    return {
        "ready": ready,
        "ready_sha256": ready_sha256,
        "stock": stock,
        "candidate": candidate,
    }


def prepare_process() -> None:
    """Select the content-addressed package before SGLang imports DeepGEMM."""

    if _PROCESS_STATE:
        raise RuntimeError("candidate process prepared twice")
    if "deep_gemm" in sys.modules:
        loaded = getattr(sys.modules["deep_gemm"], "__file__", None)
        raise RuntimeError(
            "DeepGEMM was imported before candidate process preparation: "
            f"{loaded}"
        )
    environment = {
        "SGLANG_GLM52_OPT": "1",
        "SGLANG_GLM52_OPT_PROFILE": "hotspot_candidates",
        "SGLANG_GLM52_OPT_OPS": "moe_w2",
        "SGLANG_GLM52_OPT_M_BUCKETS": "moe_down_proj:16|32",
        "SGLANG_GLM52_HOTSPOT_MODULE": str(_PROVIDER),
        # Empty disables dispatch's optional diagnostic file. Counters remain
        # in memory and are retained in the signed result.
        "SGLANG_GLM52_OPT_HIT_FILE": "",
        "DG_JIT_CACHE_DIR": str(_DG_CACHE),
        # The signed CPU bundle and JIT key include ptxas resource diagnostics.
        # This affects compiler flags/cache identity, not the launched kernel.
        "DG_JIT_PTXAS_VERBOSE": "1",
        "SGLANG_DG_CACHE_DIR": str(_DG_CACHE),
        "GLM52_MOE_W2_RUNTIME_PACKAGE": str(_PACKAGE),
        "GLM52_MOE_W2_DSO_SHA256": _DSO_SHA256,
    }
    previous = {name: os.environ.get(name) for name in environment}
    os.environ.update(environment)
    sys.path.insert(0, str(_PACKAGE_PARENT))
    _PROCESS_STATE.update(
        {
            "environment": environment,
            "previous_environment": previous,
            "path_inserted": str(_PACKAGE_PARENT),
        }
    )


def prepare_runtime(runtime) -> None:
    if _STATE:
        raise RuntimeError("candidate runtime prepared twice")
    if not _PROCESS_STATE:
        raise RuntimeError("candidate process preparation was skipped")
    identity = _verify_identity(runtime)
    previous_opt = os.environ.get("SGLANG_GLM52_OPT")
    os.environ["SGLANG_GLM52_OPT"] = "1"
    try:
        deep_gemm = importlib.import_module("deep_gemm")
        if Path(deep_gemm.__file__).resolve().parent != _PACKAGE.resolve():
            raise RuntimeError("a non-task-local DeepGEMM package was imported")

        from sglang.srt.layers.glm52_opt import dispatch, hotspot_provider
        from sglang.srt.layers.glm52_opt.context import set_forward_mode
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        dispatch._HIT_COUNTS.clear()
        dispatch._MISS_COUNTS.clear()
        hotspot_provider._reset_hotspot_provider_for_tests()
        set_forward_mode(
            ForwardMode.DECODE,
            runtime.workload.params["decode_m"],
        )
        if not hotspot_provider.initialize_hotspot_provider(
            gpu_id=runtime.local_rank
        ):
            raise RuntimeError("SGLang API-v1 provider was not selected")
        provider_state = hotspot_provider.provider_state()
        provider_module = sys.modules.get(provider_state.get("module_name", ""))
        if (
            provider_state.get("ready") is not True
            or provider_state.get("selected_ops") != ["moe_down_proj"]
            or provider_module is None
        ):
            raise RuntimeError("SGLang provider registration did not close")
        _STATE.update(
            {
                "previous_opt": previous_opt,
                "deep_gemm": deep_gemm,
                "dispatch": dispatch,
                "hotspot_provider": hotspot_provider,
                "provider_module": provider_module,
                "identity": identity,
                "workload": runtime.workload.name,
                "decode_m": runtime.workload.params["decode_m"],
                "selected_invocations": 0,
            }
        )
    except BaseException:
        try:
            from sglang.srt.layers.glm52_opt.context import set_forward_mode

            set_forward_mode(None)
        except Exception:
            pass
        if previous_opt is None:
            os.environ.pop("SGLANG_GLM52_OPT", None)
        else:
            os.environ["SGLANG_GLM52_OPT"] = previous_opt
        raise


def _selected_w2(lhs, rhs, out, masked_m, expected_m):
    dispatch = _STATE.get("dispatch")
    provider_module = _STATE.get("provider_module")
    if dispatch is None or provider_module is None:
        raise RuntimeError("candidate invoked before runtime preparation")
    selected = dispatch.try_dispatch_moe_masked(
        lhs,
        rhs,
        out,
        masked_m,
        expected_m,
    )
    if not selected:
        raise RuntimeError(
            "candidate dispatch declined; stock fallback is forbidden in the "
            "selected-candidate arm"
        )
    _STATE["selected_invocations"] += 1
    return None


def run(inputs, runtime):
    if runtime.workload.family == "moe_compute_region":
        return runtime.run_moe_compute_region(inputs, w2_gemm=_selected_w2)

    from sglang.srt.layers.glm52_opt.context import op_context

    with op_context("moe_down_proj"):
        returned = _selected_w2(
            (inputs["activation_fp8"], inputs["activation_scale"]),
            (inputs["weight_fp8"], inputs["weight_scale"]),
            inputs["out"],
            inputs["masked_m"],
            inputs["expected_m"],
        )
    if returned is not None:
        raise RuntimeError("candidate leaf violated the stock None return ABI")
    return inputs["observed_out"]


def runtime_evidence() -> dict[str, Any]:
    if not _STATE:
        raise RuntimeError("candidate evidence requested before preparation")
    identity = _STATE["identity"]
    dispatch = _STATE["dispatch"]
    provider_module = _STATE["provider_module"]
    provider = provider_module.runtime_evidence()
    return {
        "contract": "glm52-moe-w2-decode-hotspot-v1",
        "workload": _STATE.get("workload"),
        "hotspot_goal": HOTSPOT_GOAL,
        "candidate_identity": CANDIDATE_IDENTITY,
        "candidate_symbol": KERNEL_SYMBOL,
        "candidate_symbol_prefix": "infini_kernel_glm52_moe_w2_decode",
        "deepgemm_base": "edcf77b276965de8f03cdc47c23f01b08bf7c7ab",
        "sglang_base": "83d313104d089bcd2af26b28453ff880f1e6a80b",
        "cutlass_base": "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8",
        "fmt_base": "553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28",
        "runtime_package": str(_PACKAGE),
        "runtime_dso_sha256": _DSO_SHA256,
        "dg_jit_cache_dir": str(_DG_CACHE),
        "ready_path": str(_READY),
        "ready_sha256": identity["ready_sha256"],
        "stock_manifest_path": str(_STOCK_DIR / "manifest.json"),
        "stock_manifest_sha256": _STOCK_DIR.name,
        "candidate_manifest_path": str(
            _CANDIDATE_DIR / "manifest.json"
        ),
        "candidate_manifest_sha256": _CANDIDATE_DIR.name,
        "stock_jit_key": identity["stock"]["jit_digest"],
        "candidate_jit_key": identity["candidate"]["jit_digest"],
        "stock_kernel_name": identity["stock"]["kernel_name"],
        "candidate_kernel_name": identity["candidate"]["kernel_name"],
        "stock_block_m": 128,
        "stock_num_stages": 8,
        "candidate_block_m": 16,
        "candidate_num_stages": 12,
        "automatic_stage_selected": 12,
        "num_sms": 148,
        "pdl": True,
        "compiled_dims": "nk",
        "jit_ptxas_verbose": True,
        "recipe": [1, 1, 128],
        "disable_ue8m0_cast": True,
        "production_default": False,
        "fallback_after_selection": False,
        "selected_invocations": _STATE["selected_invocations"],
        "dispatch_hits": dict(dispatch._HIT_COUNTS),
        "dispatch_misses": dict(dispatch._MISS_COUNTS),
        "allowed_preselection_miss": (
            None
            if _STATE["workload"].startswith("moe_w2_hotspot_decode_")
            else (
                "moe_no_spec:moe_gate_proj:decode:"
                f"m{_STATE['decode_m']}"
            )
        ),
        "provider_state": _STATE["hotspot_provider"].provider_state(),
        "provider": provider,
    }


def runtime_artifact_paths() -> tuple[str, ...]:
    paths = [
        _READY,
        _READY_SIGNATURE,
        _STOCK_DIR / "manifest.json",
        _STOCK_DIR / "manifest.json.sha256",
        _CANDIDATE_DIR / "manifest.json",
        _CANDIDATE_DIR / "manifest.json.sha256",
        _PACKAGE / "_C.so",
    ]
    for root in (_STOCK_DIR, _CANDIDATE_DIR):
        paths.extend(
            root / name
            for name in (
                "kernel.cu",
                "kernel.cubin",
                "kernel.ptx",
                "kernel.sass",
                "resource.txt",
                "symbols.txt",
                "ptxas.log",
                "build_commands.json",
            )
        )
    return tuple(str(path) for path in paths)


def cleanup_runtime(_runtime) -> None:
    if not _STATE:
        return
    from sglang.srt.layers.glm52_opt.context import set_forward_mode

    set_forward_mode(None)
    _STATE["hotspot_provider"]._reset_hotspot_provider_for_tests()
    previous_opt = _STATE["previous_opt"]
    if previous_opt is None:
        os.environ.pop("SGLANG_GLM52_OPT", None)
    else:
        os.environ["SGLANG_GLM52_OPT"] = previous_opt
    _STATE.clear()


def cleanup_process() -> None:
    if not _PROCESS_STATE:
        return
    previous = _PROCESS_STATE["previous_environment"]
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    path = _PROCESS_STATE["path_inserted"]
    if sys.path and sys.path[0] == path:
        sys.path.pop(0)
    _PROCESS_STATE.clear()
