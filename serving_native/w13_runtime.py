"""Leased-GPU owner for independent same-source W13 stock/candidate modules."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any


class W13Runtime:
    """Bind, warm, freeze, and expose two independent DeepGEMM runtimes.

    Construction is intentionally performed by ``Runtime`` only after
    ``torch.cuda.set_device``.  Importing this module is CPU-only.
    """

    def __init__(
        self,
        torch_module: Any,
        device: Any,
        *,
        manifest_path: Path,
        variant: str,
    ) -> None:
        from sglang.srt.layers.glm52_opt import w13_decode

        self.torch = torch_module
        self.device = device
        self.variant = variant
        self.config = w13_decode.VARIANT_CONFIGS.get(variant)
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
        stock_record, _ = w13_decode._variant_record(self.manifest_path, "stock")
        candidate_record, _ = w13_decode._variant_record(
            self.manifest_path, "candidate"
        )
        stock_cache = Path(stock_record["jit_cache"]).resolve()
        candidate_cache = Path(candidate_record["jit_cache"]).resolve()
        if stock_cache == candidate_cache:
            raise RuntimeError("W13 stock and candidate cache roots alias")

        saved = {
            name: os.environ.get(name)
            for name in (
                "DG_JIT_CACHE_DIR",
                "SGLANG_DG_CACHE_DIR",
                "DG_JIT_USE_NVRTC",
                "SGL_DG_USE_NVRTC",
            )
        }
        tensors = None
        try:
            # Bind exact stock before importing compile_utils; its import-time
            # cache rewrite can no longer redirect this lazy Compiler.
            os.environ["DG_JIT_USE_NVRTC"] = "0"
            os.environ["SGL_DG_USE_NVRTC"] = "0"
            self._select_cache(stock_cache)
            self.stock, stock_record, _ = w13_decode.load_variant(
                self.manifest_path,
                "stock",
                module_name=f"deep_gemm_w13_stock_harness_{os.getpid()}",
            )
            stock_state = w13_decode._set_required_runtime_state(
                self.stock, "harness stock"
            )
            tensors = w13_decode._allocate_warm_inputs(device)
            for expected_m in (4, 5, 8, 9):
                w13_decode._launch_named_config(self.stock, tensors, expected_m, None)
            torch_module.cuda.synchronize(device)
            stock_snapshot = w13_decode._cache_snapshot(stock_cache)
            if not stock_snapshot:
                raise RuntimeError(
                    "same-source stock cache is empty after named warmup"
                )

            compile_utils = importlib.import_module(
                "sglang.srt.layers.deep_gemm_wrapper.compile_utils"
            )
            compile_utils._ENABLE_JIT_DEEPGEMM_PRECOMPILE = False

            # The installed module owns stock W2 in the containing region.
            import deep_gemm

            deep_gemm.set_pdl(w13_decode.REQUIRED_PDL)
            deep_gemm.set_num_sms(w13_decode.REQUIRED_NUM_SMS)
            deep_gemm.set_tc_util(w13_decode.REQUIRED_TC_UTIL)
            installed_state = w13_decode._read_runtime_state(deep_gemm)
            if installed_state != stock_state:
                raise RuntimeError(
                    "installed downstream DeepGEMM runtime differs from W13 stock: "
                    f"{installed_state} != {stock_state}"
                )

            self._select_cache(candidate_cache)
            self.candidate, candidate_record, _ = w13_decode.load_variant(
                self.manifest_path,
                "candidate",
                module_name=f"deep_gemm_w13_candidate_harness_{os.getpid()}",
            )
            candidate_state = w13_decode._set_required_runtime_state(
                self.candidate, "harness candidate"
            )
            independence = w13_decode._prove_runtime_state_independence(
                self.stock, self.candidate
            )
            for expected_m in (4, 5, 8, 9):
                w13_decode._launch_named_config(
                    self.candidate,
                    tensors,
                    expected_m,
                    self.config,
                )
            torch_module.cuda.synchronize(device)
            candidate_snapshot = w13_decode._cache_snapshot(candidate_cache)
            if not candidate_snapshot:
                raise RuntimeError(
                    "same-source candidate cache is empty after named warmup"
                )

            probe_cache = self.manifest_path.parent / (
                f"unbound-harness-cache-probe-{os.getpid()}"
            )
            if probe_cache.exists():
                raise RuntimeError(
                    f"cache-bind probe path already exists: {probe_cache}"
                )
            self._select_cache(probe_cache)
            w13_decode._launch_named_config(self.stock, tensors, 4, None)
            w13_decode._launch_named_config(self.candidate, tensors, 4, self.config)
            torch_module.cuda.synchronize(device)
            if probe_cache.exists():
                raise RuntimeError(
                    f"a W13 compiler escaped its cache owner: {probe_cache}"
                )
            if w13_decode._cache_snapshot(stock_cache) != stock_snapshot:
                raise RuntimeError("same-source stock cache changed after freeze")
            if w13_decode._cache_snapshot(candidate_cache) != candidate_snapshot:
                raise RuntimeError("same-source candidate cache changed after freeze")

            self.identity = {
                "manifest": str(self.manifest_path),
                "manifest_sha256": w13_decode._sha256(self.manifest_path),
                "variant": variant,
                "config": list(self.config),
                "runtime_state": {
                    "installed_downstream": installed_state,
                    "stock": stock_state,
                    "candidate": candidate_state,
                },
                "state_independence": independence,
                "modules": {
                    "stock": w13_decode._module_identity(stock_record, stock_snapshot),
                    "candidate": w13_decode._module_identity(
                        candidate_record, candidate_snapshot
                    ),
                },
                "broad_precompile_enabled": bool(
                    compile_utils._ENABLE_JIT_DEEPGEMM_PRECOMPILE
                ),
                "jit_use_nvrtc": bool(int(os.environ["DG_JIT_USE_NVRTC"])),
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

    @staticmethod
    def _select_cache(path: Path) -> None:
        os.environ["DG_JIT_CACHE_DIR"] = str(path)
        os.environ["SGLANG_DG_CACHE_DIR"] = str(path)

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
        return self.candidate.fp8_m_grouped_gemm_nt_masked(
            lhs,
            rhs,
            out,
            masked_m,
            expected_m,
            compiled_dims="nk",
            disable_ue8m0_cast=True,
            w13_config=self.config,
        )
