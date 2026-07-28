"""Task26 exact-post1 W2 em8/BM16/stage11 candidate."""

from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace

CANDIDATE_IDENTITY = "task26-em8_bm16_stage11-v4-exact-post1-edcf77b"
DECLARED_FALLBACK = False

_HARNESS_ROOT = Path(__file__).resolve().parents[2]
_SGLANG_ROOT = Path(
    os.environ.get("SGLANG_ROOT", _HARNESS_ROOT.parent / "sglang")
).resolve()
_OVERLAY_SOURCE = _SGLANG_ROOT / "third_party" / "deepgemm_w2_em8_bm16_stage11_v4"
ARTIFACT_PATHS = tuple(
    str(path)
    for path in (
        _OVERLAY_SOURCE / "README.md",
        _OVERLAY_SOURCE / "base_lock.json",
        _OVERLAY_SOURCE / "source.patch",
        _OVERLAY_SOURCE / "build_tool.patch",
        _OVERLAY_SOURCE / "core_source_hashes.sha256",
        _OVERLAY_SOURCE / "build_overlay.sh",
        _OVERLAY_SOURCE / "overlay_manifest.py",
        _OVERLAY_SOURCE / "publish_ready.sh",
        _OVERLAY_SOURCE / "ready_bundle.py",
        _OVERLAY_SOURCE / "run_with_exact_post1_stock.sh",
        _OVERLAY_SOURCE / "verify_source_reproducibility.sh",
    )
)

_STATE: dict[str, object] = {}
_INPUT_CALLABLE_KEY = "_glm52_w2_em8_bm16_stage11_v4_candidate_callable"
_TASK_DG_CACHE = Path(
    "/home/qinhaiyan/glm52-v2-goal-runs/cache/"
    "26-moe_w2_decode_scoped_bm16/em8_bm16_stage11_v4/deepgemm"
)
_GENERATED_TEMPLATE_RE = re.compile(
    r"sm100_fp8_fp4_gemm_1d1d_impl<.*?"
    r"\b0\s*,\s*6144\s*,\s*2048\s*,\s*(\d+)\s*,\s*128\s*,\s*128"
    r"\s*,\s*32\s*,\s*128\s*,\s*128\s*,\s*128\s*,\s*(\d+)"
    r"\s*,\s*128\s*,\s*128\b",
    re.DOTALL,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _temporary_environment(values: dict[str, str]):
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield previous
    finally:
        for name, prior in previous.items():
            if prior is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prior


def _refresh_jit_artifacts() -> tuple[Path, Path]:
    evidence = _STATE.get("evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError(  # noqa: TRY004 - invalid lifecycle state, not caller type
            "stage11 JIT evidence requested before preparation"
        )
    expected_name = evidence.get("candidate_jit_identity")
    if not isinstance(expected_name, str):
        raise RuntimeError(  # noqa: TRY004 - malformed prepared runtime state
            "stage11 candidate JIT identity is missing"
        )

    cache_root = Path(str(evidence.get("dg_jit_cache_dir", ""))).resolve()
    if cache_root != _TASK_DG_CACHE.resolve():
        raise RuntimeError(f"stage11 JIT cache drifted from task root: {cache_root}")
    kernel_root = (cache_root / "cache").resolve()
    matches: list[tuple[Path, str]] = []
    directory_re = re.compile(rf"kernel\.{re.escape(expected_name)}\.([0-9a-f]{{32}})")
    if kernel_root.is_dir():
        for path in kernel_root.glob(f"kernel.{expected_name}.*"):
            resolved = path.resolve()
            match = directory_re.fullmatch(resolved.name)
            if (
                resolved.is_dir()
                and resolved.parent == kernel_root
                and match is not None
            ):
                matches.append((resolved, match.group(1)))
    if len(matches) != 1:
        raise RuntimeError(
            "stage11 exact JIT cache identity must resolve to one kernel "
            f"directory, found {len(matches)} for {expected_name}"
        )

    kernel_dir, cache_key = matches[0]
    source_path = kernel_dir / "kernel.cu"
    cubin_path = kernel_dir / "kernel.cubin"
    if not source_path.is_file() or not cubin_path.is_file():
        raise RuntimeError(f"stage11 JIT cache entry is incomplete: {kernel_dir}")
    generated_source = source_path.read_text()
    template_match = _GENERATED_TEMPLATE_RE.search(generated_source)
    if (
        template_match is None
        or int(template_match.group(1)) != 16
        or int(template_match.group(2)) != 11
    ):
        raise RuntimeError(
            "stage11 generated source does not encode exact positional "
            "BM16 and num_stages=11 template configuration"
        )

    evidence.update(
        {
            "jit_cache_kernel_name": expected_name,
            "jit_cache_kernel_dir": str(kernel_dir),
            "jit_cache_key": cache_key,
            "jit_cache_source_path": str(source_path),
            "jit_cache_source_sha256": _sha256(source_path),
            "jit_cache_cubin_path": str(cubin_path),
            "jit_cache_cubin_sha256": _sha256(cubin_path),
            "jit_cache_generated_impl": "sm100_fp8_fp4_gemm_1d1d_impl",
            "jit_cache_generated_block_m": 16,
            "jit_cache_generated_num_stages": 11,
            "jit_cache_template_segment": [
                0,
                6144,
                2048,
                16,
                128,
                128,
                32,
                128,
                128,
                128,
                11,
                128,
                128,
            ],
        }
    )
    return source_path, cubin_path


def _refresh_ready_artifacts() -> tuple[Path, ...]:
    evidence = _STATE.get("evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError(  # noqa: TRY004 - invalid lifecycle state
            "stage11-v4 READY evidence requested before preparation"
        )
    required_hashes = (
        ("ready_path", "ready_sha256"),
        ("manifest_path", "manifest_sha256"),
        ("source_replay_path", "source_replay_sha256"),
        ("build_provenance_path", "build_provenance_sha256"),
    )
    paths: list[Path] = []
    for path_field, hash_field in required_hashes:
        path_raw = evidence.get(path_field)
        digest = evidence.get(hash_field)
        if not isinstance(path_raw, str) or not isinstance(digest, str):
            raise RuntimeError(  # noqa: TRY004 - malformed prepared evidence
                f"stage11-v4 READY evidence lacks {path_field}"
            )
        path = Path(path_raw).resolve()
        if not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"stage11-v4 READY artifact drifted: {path}")
        paths.append(path)

    manifest = json.loads(paths[1].read_text())
    bundle_dir = paths[1].parent
    stock_extension = bundle_dir / "stock/site/deep_gemm/_C.so"
    candidate_relative = manifest.get("candidate", {}).get("package_relpath")
    if not isinstance(candidate_relative, str):
        raise RuntimeError(  # noqa: TRY004 - malformed bound manifest
            "stage11-v4 manifest lacks candidate package_relpath"
        )
    candidate_extension = bundle_dir / candidate_relative / "_C.so"
    for role, path, record in (
        ("stock", stock_extension, manifest.get("stock", {})),
        ("candidate", candidate_extension, manifest.get("candidate", {})),
    ):
        expected = record.get("extension_sha256")
        if (
            not path.is_file()
            or not isinstance(expected, str)
            or _sha256(path) != expected
        ):
            raise RuntimeError(f"stage11-v4 {role} extension drifted: {path}")
        paths.append(path)
    return tuple(paths)


def prepare_runtime(runtime) -> None:
    """Prepare both independent DeviceRuntimes after scheduler GPU assignment."""
    workload = runtime.workload
    expected_pair = (
        workload.params.get("decode_m"),
        workload.params.get("expected_m"),
    )
    if (
        workload.family not in {"moe_grouped_masked", "moe_compute_region"}
        or expected_pair != (32, 8)
        or "candidate_jit_identity" not in workload.params
        or workload.params.get("candidate_variant") != "em8_bm16_stage11"
        or workload.params.get("candidate_variant_version") != 4
    ):
        raise RuntimeError(
            f"em8/BM16/stage11 candidate does not support {workload.name}"
        )
    if _STATE:
        raise RuntimeError("stage11 candidate runtime was prepared twice")

    # update_deep_gemm_config reads these once during worker-style setup.
    # Restore OPT0 immediately afterward so the authoritative reference cannot
    # enter any legacy glm52 replacement.
    setup_environment = {
        "SGLANG_GLM52_OPT": "1",
        "SGLANG_GLM52_OPT_PROFILE": "moe_w2_em8_bm16_stage11_v4",
        "SGLANG_GLM52_OPT_OPS": "moe_down_proj",
        "SGLANG_DEEPGEMM_PDL": "1",
    }
    with _temporary_environment(setup_environment) as saved_environment:
        from sglang.srt.layers.deep_gemm_wrapper import entrypoint

        context_factory = entrypoint.update_deep_gemm_config(
            runtime.local_rank,
            SimpleNamespace(
                chunked_prefill_size=8192,
                base_gpu_id=runtime.local_rank,
            ),
        )

    contract = entrypoint._W2_BM16_PREPARED_CONTRACT
    if context_factory is None or contract is None:
        raise RuntimeError("stage11 runtime preparation produced no launch token")

    from sglang.srt.layers.glm52_opt.context import op_context
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    stack = ExitStack()
    try:
        stack.enter_context(context_factory(ForwardMode.DECODE, expected_pair[0]))
        stack.enter_context(op_context("moe_down_proj"))
    except BaseException:
        stack.close()
        raise

    evidence = contract.evidence()
    evidence.update(
        {
            "workload": workload.name,
            "decode_m": expected_pair[0],
            "expected_m": expected_pair[1],
            "candidate_jit_identity": workload.params["candidate_jit_identity"],
            "forward_mode": "DECODE",
            "op_tag": "moe_down_proj",
            "variant_name": "em8_bm16_stage11",
            "variant_version": 4,
            "masked_block_m_override": 16,
            "masked_num_stages_override": 11,
            "predeclared_fallback": "em8_bm16_stage10",
            "fallback_eligible": False,
            "setup_environment_restored": {
                name: os.environ.get(name) == previous
                for name, previous in saved_environment.items()
            },
            "reference_opt_level_after_prepare": os.environ.get("SGLANG_GLM52_OPT"),
            "ready_verified_before_runtime": True,
        }
    )
    _STATE.update(
        {
            "stack": stack,
            "entrypoint": entrypoint,
            "evidence": evidence,
        }
    )


def cleanup_runtime(_runtime) -> None:
    stack = _STATE.get("stack")
    try:
        if stack is not None:
            stack.close()
    finally:
        _STATE.clear()


def runtime_evidence() -> dict:
    evidence = _STATE.get("evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError(  # noqa: TRY004 - invalid lifecycle state, not caller type
            "stage11 runtime evidence requested before preparation"
        )
    _refresh_ready_artifacts()
    _refresh_jit_artifacts()
    return dict(evidence)


def runtime_artifact_paths() -> tuple[str, ...]:
    paths = (*_refresh_ready_artifacts(), *_refresh_jit_artifacts())
    return tuple(str(path) for path in paths)


def _candidate_callable(inputs):
    gemm = inputs.get(_INPUT_CALLABLE_KEY)
    if gemm is not None:
        return gemm
    entrypoint = _STATE.get("entrypoint")
    if entrypoint is None:
        raise RuntimeError("stage11 candidate runtime is not prepared")

    sink = SimpleNamespace()
    sink.set_masked_down_gemm = lambda value: setattr(sink, "gemm", value)
    if "w2_weight_fp8" in inputs:
        w2_weight = inputs["w2_weight_fp8"]
        w2_scale = inputs["w2_weight_scale"]
    else:
        w2_weight = inputs["weight_fp8"]
        w2_scale = inputs["weight_scale"]
    entrypoint.configure_w2_bm16_masked_down_gemm(
        sink,
        w2_weight=w2_weight,
        w2_scale=w2_scale,
        block_shape=[128, 128],
        deep_gemm_backend=True,
        is_fp4_experts=False,
        use_mxfp8=False,
    )
    gemm = getattr(sink, "gemm", None)
    if gemm is None:
        raise RuntimeError("stage11 per-layer callable was not bound")
    inputs[_INPUT_CALLABLE_KEY] = gemm
    return gemm


def run(inputs, runtime):
    gemm = _candidate_callable(inputs)
    if runtime.workload.family == "moe_compute_region":
        return runtime.run_moe_compute_region(inputs, w2_gemm=gemm)
    result = gemm(
        (inputs["activation_fp8"], inputs["activation_scale"]),
        (inputs["weight_fp8"], inputs["weight_scale"]),
        inputs["out"],
        inputs["masked_m"],
        inputs["expected_m"],
    )
    if result is not None:
        raise RuntimeError(
            "stage11 candidate violated the stock no-overlap None return ABI"
        )
    return inputs["observed_out"]
