"""Runner for fixed, serving-native GLM-5.2 workloads.

Local tasks invoke the exact production kernel ABI.  Distributed tasks must be
launched with torchrun; their latency is the maximum CUDA-event latency across
all ranks, which is the value that gates the serving step.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import socket
import statistics
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PAIRED_SGLANG_ROOT = REPO_ROOT.parent / "sglang"
DEFAULT_SGLANG_ROOT = (
    PAIRED_SGLANG_ROOT
    if PAIRED_SGLANG_ROOT.is_dir()
    else Path("/home/qinhaiyan/sglang")
)
SGLANG_ROOT = Path(os.environ.get("SGLANG_ROOT", DEFAULT_SGLANG_ROOT)).resolve()
SGLANG_PYTHON = SGLANG_ROOT / "python"
CALLABLE_CANDIDATE_API = "callable_v1"
TRUSTED_CONFIG_CANDIDATE_API = "reference_with_config_v1"
MOE_OUTPUT_POISON = -57344.0
if str(SGLANG_PYTHON) not in sys.path:
    sys.path.insert(0, str(SGLANG_PYTHON))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving_native.audit_result import audit_document
from serving_native.contract_v2 import (
    MIN_REQUIRED_SERIES,
    PERFORMANCE_THRESHOLD,
    SCHEMA_VERSION,
    canonical_sha256,
    clock_sample,
    collect_hardware_provenance,
    collect_import_provenance,
    file_artifact,
    git_repository,
    inspect_cuda_graph,
    latency_summary,
    profile_cuda_callable,
    runtime_state_delta,
    runtime_state_snapshot,
    utc_now,
)
from serving_native.workloads import WORKLOADS, Workload, as_dict, get_workload


@dataclass
class TaskResult:
    observed: Any
    state: Any = None


def _load_candidate(path: str | None) -> ModuleType | None:
    if path is None:
        return None
    candidate_path = Path(path).expanduser().resolve()
    if candidate_path.is_dir():
        candidate_path = candidate_path / "candidate.py"
    if not candidate_path.is_file():
        raise FileNotFoundError(f"candidate not found: {candidate_path}")
    spec = importlib.util.spec_from_file_location(
        "serving_native_candidate", candidate_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import candidate: {candidate_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    declared_api = getattr(module, "CANDIDATE_API", None)
    if declared_api is None:
        if not callable(getattr(module, "run", None)):
            raise TypeError(f"{candidate_path} must export run(inputs, runtime)")
        candidate_api = CALLABLE_CANDIDATE_API
    elif declared_api == TRUSTED_CONFIG_CANDIDATE_API:
        if callable(getattr(module, "run", None)):
            raise TypeError(
                f"{candidate_path}: {TRUSTED_CONFIG_CANDIDATE_API} is "
                "declarative and must not export run()"
            )
        config = getattr(module, "CANDIDATE_CONFIG", None)
        if not isinstance(config, dict) or not config:
            raise TypeError(
                f"{candidate_path}: {TRUSTED_CONFIG_CANDIDATE_API} requires "
                "a non-empty CANDIDATE_CONFIG dictionary"
            )
        candidate_api = TRUSTED_CONFIG_CANDIDATE_API
    else:
        raise TypeError(f"{candidate_path}: unsupported CANDIDATE_API={declared_api!r}")
    module.__candidate_path__ = str(candidate_path)
    module.__candidate_api__ = candidate_api
    artifact_paths: list[str] = []
    for value in getattr(module, "ARTIFACT_PATHS", ()):
        artifact = Path(value).expanduser()
        if not artifact.is_absolute():
            artifact = candidate_path.parent / artifact
        artifact = artifact.resolve()
        if not artifact.is_file():
            raise FileNotFoundError(f"candidate artifact not found: {artifact}")
        artifact_paths.append(str(artifact))
    module.__candidate_artifact_paths__ = artifact_paths
    return module


class CallAccounting:
    """Record reached A/B paths outside the persisted timing samples."""

    def __init__(self) -> None:
        self.phase = "setup"
        self.reference_calls = 0
        self.candidate_hits = 0
        self.candidate_fallbacks = 0
        self.candidate_reference_delegations = 0
        self.candidate_trusted_config_calls = 0
        self.by_phase: dict[str, dict[str, int]] = {}

    def _increment(self, key: str) -> None:
        phase = self.by_phase.setdefault(
            self.phase,
            {
                "reference_calls": 0,
                "candidate_hits": 0,
                "candidate_fallbacks": 0,
                "candidate_reference_delegations": 0,
                "candidate_trusted_config_calls": 0,
            },
        )
        phase[key] += 1
        setattr(self, key, getattr(self, key) + 1)

    def reference(self) -> None:
        self._increment("reference_calls")

    def candidate(
        self,
        *,
        fallback: bool,
        reference_delegated: bool,
        trusted_config: bool,
    ) -> None:
        self._increment("candidate_hits")
        if fallback:
            self._increment("candidate_fallbacks")
        if reference_delegated:
            self._increment("candidate_reference_delegations")
        if trusted_config:
            self._increment("candidate_trusted_config_calls")

    def graph_replay(
        self,
        implementation: str,
        *,
        fallback: bool = False,
        reference_delegated: bool = False,
        trusted_config: bool = False,
    ) -> None:
        if implementation == "reference":
            self.reference()
        else:
            self.candidate(
                fallback=fallback,
                reference_delegated=reference_delegated,
                trusted_config=trusted_config,
            )

    def render(self, candidate_module: ModuleType) -> dict[str, Any]:
        identity_control = bool(getattr(candidate_module, "IDENTITY_CONTROL", False))
        return {
            "reference": {
                "identity": "SGLANG_GLM52_OPT=0 production path",
                "call_count": self.reference_calls,
            },
            "candidate": {
                "identity": str(
                    getattr(
                        candidate_module,
                        "CANDIDATE_IDENTITY",
                        Path(candidate_module.__candidate_path__).name,
                    )
                ),
                "api": str(candidate_module.__candidate_api__),
                "identity_control": identity_control,
                "declared_fallback": bool(
                    getattr(candidate_module, "DECLARED_FALLBACK", False)
                ),
                "hit_count": self.candidate_hits,
                "fallback_count": self.candidate_fallbacks,
                "reference_delegations": self.candidate_reference_delegations,
                "trusted_config_call_count": self.candidate_trusted_config_calls,
                "by_phase": self.by_phase,
            },
        }


class Runtime:
    def __init__(
        self,
        workload: Workload,
        candidate_module: ModuleType | None = None,
    ):
        import torch

        # A stale side-channel glm52_opt.env must not turn the reference into
        # another candidate.  Apply it once (matching worker startup), then pin
        # this isolated benchmark process to the production OPT0 path.
        from sglang.srt.layers.glm52_opt.config import ensure_glm52_env

        ensure_glm52_env()
        os.environ["SGLANG_GLM52_OPT"] = "0"

        self.torch = torch
        self.workload = workload
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.tp_group = None
        self.moe_ep_group = None
        self.device_group = None
        self.deep_ep = None
        self.deep_ep_buffer = None
        self._deepep_buffer_facade = None
        self._normal_dispatch_config = None
        self._normal_combine_config = None
        self.accounting = CallAccounting()
        self._inside_candidate = False
        self._candidate_reference_calls = 0
        self.w13_runtime = None

        if not torch.cuda.is_available():
            raise RuntimeError("serving-native workloads require CUDA")
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

        if workload.family in ("moe_grouped_masked", "moe_w13_region") and (
            workload.name.startswith("moe_w13_")
        ):
            manifest = os.environ.get("SGLANG_GLM52_W13_DECODE_MANIFEST", "").strip()
            if not manifest:
                raise RuntimeError(
                    "W13 serving-native workloads require "
                    "SGLANG_GLM52_W13_DECODE_MANIFEST"
                )
            variant = (
                str(getattr(candidate_module, "W13_VARIANT", "")).strip().lower()
                if candidate_module is not None
                else ""
            )
            if not variant:
                raise RuntimeError(
                    "W13 serving-native candidate must declare W13_VARIANT"
                )
            from serving_native.w13_runtime import W13Runtime

            self.w13_runtime = W13Runtime(
                torch,
                self.device,
                manifest_path=Path(manifest),
                variant=variant,
            )

        if workload.distributed:
            self._init_distributed()
        elif self.world_size != 1:
            raise RuntimeError(
                f"local task {workload.name} must not run inside WORLD_SIZE={self.world_size}"
            )

    def _init_distributed(self) -> None:
        if self.world_size != self.workload.world_size:
            raise RuntimeError(
                f"{self.workload.name} fixes world_size={self.workload.world_size}, "
                f"but torchrun supplied WORLD_SIZE={self.world_size}"
            )

        from sglang.srt.distributed import init_distributed_environment
        from sglang.srt.distributed.parallel_state import (
            get_moe_ep_group,
            get_tp_group,
            initialize_model_parallel,
        )

        init_distributed_environment(
            world_size=self.world_size,
            rank=self.rank,
            local_rank=self.local_rank,
            distributed_init_method="env://",
            backend="nccl",
        )
        # Both fixed lanes use attention TP=1 and full-world MoE EP.  The
        # workload world_size selects TP4/DP4/EP4 or TP8/DP8/EP8.
        initialize_model_parallel(
            tensor_model_parallel_size=self.world_size,
            expert_model_parallel_size=self.world_size,
            attention_data_parallel_size=self.world_size,
        )
        self.tp_group = get_tp_group()
        self.moe_ep_group = get_moe_ep_group()
        coordinator = (
            self.moe_ep_group
            if self.workload.family.startswith("deepep_")
            else self.tp_group
        )
        self.device_group = coordinator.device_group
        self.torch.distributed.barrier(group=self.device_group)

    def close(self) -> None:
        if self.workload.distributed and self.torch.distributed.is_initialized():
            self.torch.distributed.barrier(group=self.device_group)
            try:
                from sglang.srt.distributed.parallel_state import destroy_model_parallel

                destroy_model_parallel()
            finally:
                self.torch.distributed.destroy_process_group()

    def _generator(self, offset: int = 0):
        generator = self.torch.Generator(device=self.device)
        generator.manual_seed(20260722 + self.rank * 1009 + offset)
        return generator

    def build_inputs(self) -> dict[str, Any]:
        family = self.workload.family
        if family == "packed_fp8_gemm":
            return self._build_packed_fp8_gemm()
        if family == "bf16_linear":
            return self._build_bf16_linear()
        if family == "moe_grouped_masked":
            return self._build_moe_grouped_masked()
        if family == "moe_w13_region":
            return self._build_moe_w13_region()
        if family == "moe_swiglu_quant":
            return self._build_moe_swiglu_quant()
        if family == "dsa_trtllm":
            return self._build_dsa_trtllm()
        if family == "allgather":
            return self._build_allgather()
        if family == "allreduce":
            return self._build_allreduce()
        if family.startswith("deepep_"):
            return self._build_deepep()
        raise NotImplementedError(f"unsupported family: {family}")

    def _build_packed_fp8_gemm(self) -> dict[str, Any]:
        import deep_gemm
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )
        from sglang.srt.layers.quantization.fp8_utils import transform_scale_ue8m0

        p = self.workload.params
        m, n, k = p["m"], p["n"], p["k"]
        x_bf16 = self.torch.randn(
            (m, k),
            device=self.device,
            dtype=self.torch.bfloat16,
            generator=self._generator(1),
        )
        x_fp8, x_scale = sglang_per_token_group_quant_fp8(
            x_bf16,
            128,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        weight_bf16 = self.torch.randn(
            (n, k),
            device=self.device,
            dtype=self.torch.bfloat16,
            generator=self._generator(2),
        )
        weight_fp8, weight_scale_blocks = deep_gemm.utils.math.per_block_cast_to_fp8(
            weight_bf16, use_ue8m0=True, gran_k=128
        )
        weight_scale = transform_scale_ue8m0(weight_scale_blocks, mn=n)
        del x_bf16, weight_bf16, weight_scale_blocks
        if x_scale.dtype != self.torch.int32 or weight_scale.dtype != self.torch.int32:
            raise RuntimeError(
                "B200 production task requires packed int32 UE8M0 scales"
            )
        return {
            "x_fp8": x_fp8,
            "weight_fp8": weight_fp8,
            "x_scale": x_scale,
            "weight_scale": weight_scale,
            "block_size": [128, 128],
        }

    def _build_bf16_linear(self) -> dict[str, Any]:
        p = self.workload.params
        return {
            "x": self.torch.randn(
                (p["m"], p["k"]),
                device=self.device,
                dtype=self.torch.bfloat16,
                generator=self._generator(3),
            ),
            "weight": self.torch.randn(
                (p["n"], p["k"]),
                device=self.device,
                dtype=self.torch.bfloat16,
                generator=self._generator(4),
            ),
        }

    def _fixed_decode_masked_m(self, params: dict[str, Any]):
        return self._decode_mask_metadata(params)["masked_m"]

    def _decode_mask_metadata(self, params: dict[str, Any]) -> dict[str, Any]:
        """Create two CPU-known masks before capture; never observe device masks."""

        experts = params["experts_per_rank"]
        assignments = params["valid_assignments"]
        host_counts = []
        for seed in (20260722, 20260723):
            generator = self.torch.Generator(device="cpu")
            generator.manual_seed(seed)
            expert_ids = self.torch.randint(
                experts,
                (assignments,),
                dtype=self.torch.int64,
                generator=generator,
            )
            counts = self.torch.bincount(expert_ids, minlength=experts).to(
                dtype=self.torch.int32
            )
            host_counts.append(counts)
        if self.torch.equal(host_counts[0], host_counts[1]):
            raise RuntimeError("deterministic graph masks unexpectedly alias")
        initial_cpu = tuple(int(value) for value in host_counts[0].tolist())
        replay_cpu = tuple(int(value) for value in host_counts[1].tolist())
        return {
            "masked_m": host_counts[0].to(device=self.device),
            "masked_m_replay": host_counts[1].to(device=self.device),
            "masked_m_initial_cpu": initial_cpu,
            "masked_m_replay_cpu": replay_cpu,
        }

    def _build_moe_grouped_masked(self) -> dict[str, Any]:
        import deep_gemm

        p = self.workload.params
        experts, slab, k, n = (
            p["experts_per_rank"],
            p["expert_slab"],
            p["k"],
            p["n"],
        )
        mask_metadata = self._decode_mask_metadata(p)
        activations_bf16 = self.torch.randn(
            (experts, slab, k),
            device=self.device,
            dtype=self.torch.bfloat16,
            generator=self._generator(5),
        )
        weights_bf16 = self.torch.randn(
            (experts, n, k),
            device=self.device,
            dtype=self.torch.bfloat16,
            generator=self._generator(6),
        ) * (k**-0.5)
        activation_pairs = [
            deep_gemm.utils.math.per_token_cast_to_fp8(
                activations_bf16[expert], use_ue8m0=True
            )
            for expert in range(experts)
        ]
        weight_pairs = [
            deep_gemm.utils.math.per_block_cast_to_fp8(
                weights_bf16[expert], use_ue8m0=True
            )
            for expert in range(experts)
        ]
        activation_fp8 = self.torch.stack([pair[0] for pair in activation_pairs])
        activation_scale = self.torch.stack([pair[1] for pair in activation_pairs])
        weight_fp8 = self.torch.stack([pair[0] for pair in weight_pairs])
        weight_scale = self.torch.stack([pair[1] for pair in weight_pairs])
        activation_scale = deep_gemm.transform_sf_into_required_layout(
            activation_scale,
            mn=slab,
            k=k,
            recipe=(1, 128, 128),
            num_groups=experts,
            is_sfa=True,
        )
        weight_scale = deep_gemm.transform_sf_into_required_layout(
            weight_scale,
            mn=n,
            k=k,
            recipe=(1, 128, 128),
            num_groups=experts,
            is_sfa=False,
        )
        del activations_bf16, weights_bf16, activation_pairs, weight_pairs
        if (
            activation_scale.dtype != self.torch.int32
            or weight_scale.dtype != self.torch.int32
        ):
            raise RuntimeError("production MoE task requires packed int32 UE8M0 scales")
        return {
            "activation_fp8": activation_fp8,
            "activation_scale": activation_scale,
            "weight_fp8": weight_fp8,
            "weight_scale": weight_scale,
            "out": self.torch.empty(
                (experts, slab, n), device=self.device, dtype=self.torch.bfloat16
            ),
            "expected_m": p["expected_m"],
            **mask_metadata,
        }

    def _build_moe_w13_region(self) -> dict[str, Any]:
        import deep_gemm

        inputs = self._build_moe_grouped_masked()
        p = self.workload.params
        experts = p["experts_per_rank"]
        slab = p["expert_slab"]
        w2_k = p["w2_k"]
        w2_n = p["w2_n"]
        w2_bf16 = self.torch.randn(
            (experts, w2_n, w2_k),
            device=self.device,
            dtype=self.torch.bfloat16,
            generator=self._generator(7),
        ) * (w2_k**-0.5)
        w2_pairs = [
            deep_gemm.utils.math.per_block_cast_to_fp8(w2_bf16[expert], use_ue8m0=True)
            for expert in range(experts)
        ]
        w2_weight_fp8 = self.torch.stack([pair[0] for pair in w2_pairs])
        w2_weight_scale = self.torch.stack([pair[1] for pair in w2_pairs])
        w2_weight_scale = deep_gemm.transform_sf_into_required_layout(
            w2_weight_scale,
            mn=w2_n,
            k=w2_k,
            recipe=(1, 128, 128),
            num_groups=experts,
            is_sfa=False,
        )
        del w2_bf16, w2_pairs
        if w2_weight_scale.dtype != self.torch.int32:
            raise RuntimeError("W13 region requires packed int32 W2 scales")
        inputs.update(
            {
                "w2_weight_fp8": w2_weight_fp8,
                "w2_weight_scale": w2_weight_scale,
                "down_out": self.torch.empty(
                    (experts, slab, w2_n),
                    device=self.device,
                    dtype=self.torch.bfloat16,
                ),
            }
        )
        return inputs

    def _build_moe_swiglu_quant(self) -> dict[str, Any]:
        p = self.workload.params
        experts, slab, gate_up = p["experts_per_rank"], p["expert_slab"], p["gate_up"]
        masked_m = self._fixed_decode_masked_m(p)
        return {
            "gateup_output": self.torch.randn(
                (experts, slab, gate_up),
                device=self.device,
                dtype=self.torch.bfloat16,
                generator=self._generator(5),
            ),
            "masked_m": masked_m,
            "group_size": p["group_size"],
            "topk": p["topk"],
        }

    def _build_dsa_trtllm(self) -> dict[str, Any]:
        p = self.workload.params
        batch, heads, head_dim = p["batch"], p["heads"], p["head_dim"]
        context, topk, page = p["context"], p["sparse_topk"], p["page_size"]
        tokens_per_seq = ((context + page - 1) // page) * page
        num_pages = batch * tokens_per_seq // page
        query = (
            self.torch.randn(
                (batch, 1, heads, head_dim),
                device=self.device,
                dtype=self.torch.float32,
                generator=self._generator(6),
            )
            * 0.05
        ).to(self.torch.float8_e4m3fn)
        kv_cache = (
            self.torch.randn(
                (num_pages, 1, page, head_dim),
                device=self.device,
                dtype=self.torch.float32,
                generator=self._generator(7),
            )
            * 0.05
        ).to(self.torch.float8_e4m3fn)
        block_tables = self.torch.full(
            (batch, 1, topk), -1, dtype=self.torch.int32, device=self.device
        )
        effective = min(context, topk)
        for batch_idx in range(batch):
            base = batch_idx * tokens_per_seq
            block_tables[batch_idx, 0, :effective] = self.torch.arange(
                base, base + effective, dtype=self.torch.int32, device=self.device
            )
        return {
            "query": query,
            "kv_cache": kv_cache,
            "workspace": self.torch.zeros(
                256 * 1024 * 1024, dtype=self.torch.uint8, device=self.device
            ),
            "block_tables": block_tables,
            "seq_lens": self.torch.full(
                (batch,), context, dtype=self.torch.int32, device=self.device
            ),
            "max_seq_len": context,
            "sparse_topk": topk,
            "bmm1_scale": head_dim**-0.5,
        }

    def _build_allgather(self) -> dict[str, Any]:
        p = self.workload.params
        local = self.torch.randn(
            (p["local_tokens"], p["hidden"]),
            device=self.device,
            dtype=self.torch.bfloat16,
            generator=self._generator(8),
        )
        return {
            "local": local,
            "output": self.torch.empty(
                (p["local_tokens"] * self.world_size, p["hidden"]),
                device=self.device,
                dtype=local.dtype,
            ),
        }

    def _build_allreduce(self) -> dict[str, Any]:
        p = self.workload.params
        source = self.torch.randn(
            (p["local_tokens"], p["hidden"]),
            device=self.device,
            dtype=self.torch.bfloat16,
            generator=self._generator(13),
        )
        return {"source": source, "local": self.torch.empty_like(source)}

    def prepare_inputs(self, inputs: dict[str, Any]) -> None:
        """Restore destructive collective inputs outside the timed window."""
        if self.workload.family == "allreduce":
            inputs["local"].copy_(inputs["source"])
        if self.workload.family in ("moe_grouped_masked", "moe_w13_region"):
            inputs["out"].fill_(MOE_OUTPUT_POISON)
            if self.workload.family == "moe_w13_region":
                inputs["down_out"].fill_(MOE_OUTPUT_POISON)

    def _init_deepep_buffer(self, params: dict[str, Any]) -> None:
        if self.deep_ep_buffer is not None:
            return
        try:
            import deep_ep
        except ImportError as exc:
            raise RuntimeError(
                "DeepEP tasks require the same deep_ep package used by the target SGLang image"
            ) from exc

        from sglang.srt.layers.moe.token_dispatcher.deepep import (
            DeepEPBuffer,
            DeepEPConfig,
        )
        from sglang.srt.layers.moe.utils import DeepEPMode

        hidden = params["hidden"]
        experts = params["experts"]
        max_tokens = params.get("max_dispatch_tokens", params["local_tokens"])
        self.deep_ep = deep_ep
        self._deepep_buffer_facade = DeepEPBuffer
        deepep_config = DeepEPConfig.get_instance()
        self._normal_dispatch_config = (
            deepep_config.normal_dispatch_config
            or deep_ep.Buffer.get_dispatch_config(self.world_size)
        )
        self._normal_combine_config = (
            deepep_config.normal_combine_config
            or deep_ep.Buffer.get_combine_config(self.world_size)
        )
        # Use SGLang's facade rather than reconstructing a raw Buffer here.  It
        # owns the exact NVL/RDMA sizing, AUTO-mode QP count, MNNVL/fabric
        # flags, and CUDA-version compatibility used by the serving process.
        self.deep_ep_buffer = DeepEPBuffer.get_deepep_buffer(
            self.device_group,
            hidden,
            2,  # BF16 input bytes, matching the GLM-5.2 dispatcher.
            DeepEPMode.AUTO,
            max_tokens,
            experts,
        )

    def _build_deepep(self) -> dict[str, Any]:
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        p = self.workload.params
        self._init_deepep_buffer(p)
        x_bf16 = self.torch.randn(
            (p["local_tokens"], p["hidden"]),
            device=self.device,
            dtype=self.torch.bfloat16,
            generator=self._generator(9),
        )
        scores = self.torch.randn(
            (p["local_tokens"], p["experts"]),
            device=self.device,
            dtype=self.torch.float32,
            generator=self._generator(10),
        )
        topk_idx = scores.topk(p["topk"], dim=-1, sorted=False).indices.to(
            self.torch.int64
        )
        topk_weights = self.torch.softmax(
            scores.gather(1, topk_idx), dim=-1, dtype=self.torch.float32
        )
        inputs: dict[str, Any] = {
            "x_bf16": x_bf16,
            "topk_idx": topk_idx,
            "topk_weights": topk_weights,
        }

        if self.workload.family.startswith("deepep_normal"):
            inputs["x_comm"] = sglang_per_token_group_quant_fp8(
                x_bf16,
                128,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
            if self.workload.family == "deepep_normal_combine":
                dispatched = self._run_deepep_normal_dispatch(inputs, config=None)
                recv_x = dispatched.state["recv_x"]
                recv_values = recv_x[0] if isinstance(recv_x, tuple) else recv_x
                inputs["combine_x"] = self.torch.randn(
                    recv_values.shape,
                    device=self.device,
                    dtype=self.torch.bfloat16,
                    generator=self._generator(11),
                )
                inputs["handle"] = dispatched.state["handle"]
        elif self.workload.family == "deepep_ll_combine":
            dispatched = self._run_deepep_ll_dispatch(inputs)
            recv_x = dispatched.state["recv_x"]
            recv_values = recv_x[0] if isinstance(recv_x, tuple) else recv_x
            inputs["combine_x"] = self.torch.randn(
                recv_values.shape,
                device=self.device,
                dtype=self.torch.bfloat16,
                generator=self._generator(12),
            )
            inputs["handle"] = dispatched.state["handle"]
        return inputs

    def _config(self, config: Any, fallback: Any) -> Any:
        if config is None:
            return fallback
        if isinstance(config, dict):
            return self.deep_ep.Config(**config)
        return config

    def _observe_masked_output(
        self,
        inputs: dict[str, Any],
        output_name: str,
    ) -> Any:
        """Return the full stable output buffer without reading the device mask."""

        return inputs[output_name]

    def _production_w13_launcher(self, *args, **kwargs):
        if self.w13_runtime is not None:
            return self.w13_runtime.stock_launcher(*args, **kwargs)
        from sglang.srt.layers.deep_gemm_wrapper.entrypoint import (
            grouped_gemm_nt_f8f8bf16_masked,
        )

        return grouped_gemm_nt_f8f8bf16_masked(*args, **kwargs)

    def candidate_w13(self, inputs: dict[str, Any]) -> TaskResult:
        if self.w13_runtime is None:
            raise RuntimeError("candidate W13 runtime is not initialized")
        if self.workload.family == "moe_grouped_masked":
            return self.run_w13_leaf(
                inputs,
                launcher=self.w13_runtime.candidate_launcher,
            )
        if self.workload.family == "moe_w13_region":
            return self.run_w13_region(
                inputs,
                launcher=self.w13_runtime.candidate_launcher,
            )
        raise RuntimeError(
            f"W13 candidate cannot run workload family {self.workload.family}"
        )

    def run_w13_leaf(
        self,
        inputs: dict[str, Any],
        *,
        launcher: Callable[..., Any],
    ) -> TaskResult:
        returned = launcher(
            (inputs["activation_fp8"], inputs["activation_scale"]),
            (inputs["weight_fp8"], inputs["weight_scale"]),
            inputs["out"],
            inputs["masked_m"],
            inputs["expected_m"],
        )
        if returned is not None:
            raise RuntimeError(
                "masked W13 no-overlap launcher must return exactly None"
            )
        return TaskResult(self._observe_masked_output(inputs, "out"))

    def run_w13_region(
        self,
        inputs: dict[str, Any],
        *,
        launcher: Callable[..., Any],
    ) -> TaskResult:
        from sglang.srt.layers.deep_gemm_wrapper.entrypoint import (
            grouped_gemm_nt_f8f8bf16_masked,
        )
        from sglang.srt.layers.moe.moe_runner.deep_gemm import (
            _varlen_deep_gemm_silu_mul_quant,
        )

        self.run_w13_leaf(inputs, launcher=launcher)
        down_input, down_input_scale = _varlen_deep_gemm_silu_mul_quant(
            inputs["out"],
            inputs["masked_m"],
            group_size=self.workload.params["group_size"],
            topk=self.workload.params["topk"],
        )
        if down_input_scale.dtype != self.torch.int32:
            raise RuntimeError(
                "W13 containing region did not preserve production packed-int32 quant"
            )
        returned = grouped_gemm_nt_f8f8bf16_masked(
            (down_input, down_input_scale),
            (inputs["w2_weight_fp8"], inputs["w2_weight_scale"]),
            inputs["down_out"],
            inputs["masked_m"],
            inputs["expected_m"],
        )
        if returned is not None:
            raise RuntimeError("masked W2 no-overlap launcher must return exactly None")
        return TaskResult(self._observe_masked_output(inputs, "down_out"))

    def reference(self, inputs: dict[str, Any], *, config: Any = None) -> TaskResult:
        if self._inside_candidate:
            self._candidate_reference_calls += 1
        else:
            self.accounting.reference()
        family = self.workload.family
        if family == "packed_fp8_gemm":
            from sglang.kernels.ops.quantization.fp8_kernel import (
                w8a8_block_fp8_matmul_deepgemm,
            )

            out = w8a8_block_fp8_matmul_deepgemm(
                inputs["x_fp8"],
                inputs["weight_fp8"],
                inputs["x_scale"],
                inputs["weight_scale"],
                inputs["block_size"],
                self.torch.bfloat16,
            )
            return TaskResult(out)
        if family == "bf16_linear":
            return TaskResult(
                self.torch.nn.functional.linear(inputs["x"], inputs["weight"])
            )
        if family == "moe_grouped_masked":
            return self.run_w13_leaf(
                inputs,
                launcher=self._production_w13_launcher,
            )
        if family == "moe_w13_region":
            return self.run_w13_region(
                inputs,
                launcher=self._production_w13_launcher,
            )
        if family == "moe_swiglu_quant":
            from sglang.srt.layers.moe.moe_runner.deep_gemm import (
                _varlen_deep_gemm_silu_mul_quant,
            )

            return TaskResult(
                _varlen_deep_gemm_silu_mul_quant(
                    inputs["gateup_output"],
                    inputs["masked_m"],
                    group_size=inputs["group_size"],
                    topk=inputs["topk"],
                )
            )
        if family == "dsa_trtllm":
            import flashinfer.decode

            out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
                query=inputs["query"],
                kv_cache=inputs["kv_cache"],
                workspace_buffer=inputs["workspace"],
                qk_nope_head_dim=192,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                block_tables=inputs["block_tables"],
                seq_lens=inputs["seq_lens"],
                max_seq_len=inputs["max_seq_len"],
                sparse_mla_top_k=inputs["sparse_topk"],
                bmm1_scale=inputs["bmm1_scale"],
                backend="trtllm-gen",
            )
            return TaskResult(
                out.squeeze(1) if out.ndim == 4 and out.shape[1] == 1 else out
            )
        if family == "allgather":
            self.tp_group.all_gather_into_tensor(inputs["output"], inputs["local"])
            return TaskResult(inputs["output"])
        if family == "allreduce":
            return TaskResult(self.tp_group.all_reduce(inputs["local"]))
        if family == "deepep_normal_dispatch":
            return self._run_deepep_normal_dispatch(inputs, config=config)
        if family == "deepep_normal_combine":
            cfg = self._config(config, self._normal_combine_config)
            combined, _, event = self.deep_ep_buffer.combine(
                inputs["combine_x"], inputs["handle"], config=cfg, async_finish=False
            )
            if event is not None and hasattr(event, "current_stream_wait"):
                event.current_stream_wait()
            return TaskResult(combined)
        if family == "deepep_ll_dispatch":
            return self._run_deepep_ll_dispatch(inputs)
        if family == "deepep_ll_combine":
            combined, event, _hook = self.deep_ep_buffer.low_latency_combine(
                x=inputs["combine_x"],
                topk_idx=inputs["topk_idx"],
                topk_weights=inputs["topk_weights"],
                handle=inputs["handle"],
                async_finish=True,
                return_recv_hook=False,
            )
            event.current_stream_wait()
            return TaskResult(combined)
        raise NotImplementedError(family)

    def _run_deepep_normal_dispatch(
        self, inputs: dict[str, Any], config: Any
    ) -> TaskResult:
        self._deepep_buffer_facade.set_dispatch_mode_as_normal()
        buffer = self.deep_ep_buffer
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = buffer.get_dispatch_layout(
            inputs["topk_idx"], self.workload.params["experts"]
        )
        recv_x, recv_ids, recv_weights, recv_counts, handle, event = buffer.dispatch(
            inputs["x_comm"],
            topk_idx=inputs["topk_idx"],
            topk_weights=inputs["topk_weights"],
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=False,
            expert_alignment=128,
            config=self._config(config, self._normal_dispatch_config),
        )
        if event is not None and hasattr(event, "current_stream_wait"):
            event.current_stream_wait()
        valid = recv_ids.ne(-1).any(dim=-1) if recv_ids is not None else None
        observed_x = recv_x
        if valid is not None:
            if isinstance(recv_x, tuple):
                observed_x = tuple(t[valid] for t in recv_x)
            else:
                observed_x = recv_x[valid]
        observed = (
            observed_x,
            recv_ids[valid],
            recv_weights[valid],
            tuple(recv_counts),
        )
        return TaskResult(observed, {"handle": handle, "recv_x": recv_x})

    def _run_deepep_ll_dispatch(self, inputs: dict[str, Any]) -> TaskResult:
        self._deepep_buffer_facade.set_dispatch_mode_as_low_latency()
        p = self.workload.params
        recv_x, recv_count, handle, event, _hook = (
            self.deep_ep_buffer.low_latency_dispatch(
                inputs["x_bf16"],
                inputs["topk_idx"],
                p["max_dispatch_tokens"],
                p["experts"],
                use_fp8=True,
                async_finish=True,
                return_recv_hook=False,
                round_scale=True,
                use_ue8m0=True,
            )
        )
        event.current_stream_wait()
        values = recv_x[0] if isinstance(recv_x, tuple) else recv_x
        scales = recv_x[1] if isinstance(recv_x, tuple) else None
        valid_values = []
        valid_scales = []
        for expert_idx, count in enumerate(recv_count.tolist()):
            valid_values.append(values[expert_idx, :count])
            if scales is not None:
                valid_scales.append(scales[expert_idx, :count])
        observed = (valid_values, valid_scales, recv_count)
        return TaskResult(observed, {"handle": handle, "recv_x": recv_x})

    def barrier(self) -> None:
        if self.workload.distributed:
            self.torch.distributed.barrier(group=self.device_group)

    def rank_max(self, latency_ms: float) -> float:
        if not self.workload.distributed:
            return latency_ms
        value = self.torch.tensor(
            [latency_ms], device=self.device, dtype=self.torch.float64
        )
        self.torch.distributed.all_reduce(
            value, op=self.torch.distributed.ReduceOp.MAX, group=self.device_group
        )
        return float(value.item())


def _iter_pairs(value: Any, prefix: str = "output"):
    if isinstance(value, TaskResult):
        yield from _iter_pairs(value.observed, prefix)
    elif hasattr(value, "shape") and hasattr(value, "dtype"):
        yield prefix, value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _iter_pairs(value[key], f"{prefix}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _iter_pairs(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def _compare(reference: TaskResult, candidate: TaskResult) -> None:
    import torch

    ref_items = list(_iter_pairs(reference))
    cand_items = list(_iter_pairs(candidate))
    if len(ref_items) != len(cand_items):
        raise AssertionError(
            f"output structure differs: {len(ref_items)} != {len(cand_items)}"
        )
    for (ref_name, ref_value), (cand_name, cand_value) in zip(ref_items, cand_items):
        if ref_name != cand_name:
            raise AssertionError(f"output structure differs: {ref_name} != {cand_name}")
        if torch.is_tensor(ref_value):
            if not torch.is_tensor(cand_value):
                raise AssertionError(f"{ref_name}: candidate is not a tensor")
            if ref_value.shape != cand_value.shape:
                raise AssertionError(
                    f"{ref_name}: shape {tuple(ref_value.shape)} != {tuple(cand_value.shape)}"
                )
            if ref_value.dtype != cand_value.dtype:
                raise AssertionError(
                    f"{ref_name}: dtype {ref_value.dtype} != {cand_value.dtype}"
                )
            ref_f = (
                ref_value.float() if ref_value.dtype.is_floating_point else ref_value
            )
            cand_f = (
                cand_value.float() if cand_value.dtype.is_floating_point else cand_value
            )
            if ref_value.dtype.is_floating_point:
                if not torch.allclose(
                    ref_f, cand_f, rtol=2e-2, atol=2e-2, equal_nan=False
                ):
                    diff = (ref_f - cand_f).abs().max().item()
                    raise AssertionError(f"{ref_name}: max abs diff {diff}")
            elif not torch.equal(ref_value, cand_value):
                raise AssertionError(f"{ref_name}: integer tensor mismatch")
        elif ref_value != cand_value:
            raise AssertionError(f"{ref_name}: {ref_value!r} != {cand_value!r}")


def _compare_masked(
    reference: TaskResult,
    candidate: TaskResult,
    counts: tuple[int, ...],
) -> None:
    """Compare only semantically valid rows using CPU-owned mask metadata."""

    import torch

    ref = reference.observed
    cand = candidate.observed
    if not torch.is_tensor(ref) or not torch.is_tensor(cand):
        raise AssertionError("masked output must be represented by one full tensor")
    if ref.shape != cand.shape:
        raise AssertionError(
            f"masked output shape {tuple(ref.shape)} != {tuple(cand.shape)}"
        )
    if ref.dtype != cand.dtype:
        raise AssertionError(f"masked output dtype {ref.dtype} != {cand.dtype}")
    if ref.ndim != 3 or ref.shape[0] != len(counts):
        raise AssertionError(
            "masked output does not match CPU-owned expert-count metadata"
        )
    for expert, count in enumerate(counts):
        if count < 0 or count > ref.shape[1]:
            raise AssertionError(f"invalid CPU-owned masked_m[{expert}]={count}")
        _compare(
            TaskResult(ref[expert, :count]),
            TaskResult(cand[expert, :count]),
        )


def _compare_for_workload(
    runtime: Runtime,
    inputs: dict[str, Any],
    reference: TaskResult,
    candidate: TaskResult,
    *,
    counts: tuple[int, ...] | None = None,
) -> None:
    if runtime.workload.family in {"moe_grouped_masked", "moe_w13_region"}:
        _compare_masked(
            reference,
            candidate,
            counts if counts is not None else inputs["masked_m_initial_cpu"],
        )
        return
    _compare(reference, candidate)


def _clone_observed(value: Any) -> Any:
    """Freeze correctness output without adding copies to timed calls."""
    if hasattr(value, "clone") and hasattr(value, "shape"):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_observed(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_observed(item) for item in value)
    if isinstance(value, list):
        return [_clone_observed(item) for item in value]
    return value


def _clone_result(result: TaskResult) -> TaskResult:
    return TaskResult(_clone_observed(result.observed))


def _exact_compare(left: TaskResult, right: TaskResult) -> None:
    import torch

    left_items = list(_iter_pairs(left))
    right_items = list(_iter_pairs(right))
    if len(left_items) != len(right_items):
        raise AssertionError("deterministic replay changed output structure")
    for (left_name, left_value), (right_name, right_value) in zip(
        left_items, right_items
    ):
        if left_name != right_name:
            raise AssertionError(
                f"deterministic replay structure differs: {left_name} != {right_name}"
            )
        if torch.is_tensor(left_value):
            if (
                not torch.is_tensor(right_value)
                or left_value.dtype != right_value.dtype
                or left_value.shape != right_value.shape
                or not torch.equal(left_value, right_value)
            ):
                raise AssertionError(f"{left_name}: repeated replay is not exact")
        elif left_value != right_value:
            raise AssertionError(f"{left_name}: repeated replay differs")


def _results_exactly_equal(left: TaskResult, right: TaskResult) -> bool:
    try:
        _exact_compare(left, right)
    except AssertionError:
        return False
    return True


def _tensor_pointers(value: Any, prefix: str = "value") -> dict[str, int]:
    import torch

    pointers: dict[str, int] = {}
    if isinstance(value, TaskResult):
        return _tensor_pointers(value.observed, prefix)
    if torch.is_tensor(value):
        pointers[prefix] = int(value.data_ptr())
    elif isinstance(value, dict):
        for key in sorted(value):
            pointers.update(_tensor_pointers(value[key], f"{prefix}.{key}"))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            pointers.update(_tensor_pointers(item, f"{prefix}[{index}]"))
    return pointers


def _poison_observed(value: Any) -> int:
    import torch

    if isinstance(value, TaskResult):
        return _poison_observed(value.observed)
    if torch.is_tensor(value):
        if value.dtype.is_floating_point:
            value.fill_(float("nan"))
        elif value.dtype == torch.bool:
            value.fill_(True)
        else:
            value.fill_(torch.iinfo(value.dtype).max)
        return 1
    if isinstance(value, dict):
        return sum(_poison_observed(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_poison_observed(item) for item in value)
    return 0


def _poison_is_visible(value: Any) -> bool:
    import torch

    checks: list[bool] = []
    for _name, item in _iter_pairs(value):
        if not torch.is_tensor(item):
            continue
        if item.dtype.is_floating_point:
            checks.append(bool(torch.isnan(item).all().item()))
        elif item.dtype == torch.bool:
            checks.append(bool(item.all().item()))
        else:
            checks.append(bool(item.eq(torch.iinfo(item.dtype).max).all().item()))
    return bool(checks) and all(checks)


def _candidate_result(
    module: ModuleType,
    inputs: dict[str, Any],
    runtime: Runtime,
) -> TaskResult:
    before = runtime._candidate_reference_calls
    runtime._inside_candidate = True
    try:
        candidate_api = module.__candidate_api__
        if candidate_api == TRUSTED_CONFIG_CANDIDATE_API:
            value = runtime.reference(
                inputs,
                config=module.CANDIDATE_CONFIG,
            )
        else:
            value = module.run(inputs, runtime)
    finally:
        runtime._inside_candidate = False
    delegated = runtime._candidate_reference_calls > before
    identity_control = bool(getattr(module, "IDENTITY_CONTROL", False))
    trusted_config = delegated and candidate_api == TRUSTED_CONFIG_CANDIDATE_API
    fallback = delegated and not (identity_control or trusted_config)
    runtime.accounting.candidate(
        fallback=fallback,
        reference_delegated=delegated,
        trusted_config=trusted_config,
    )
    return value if isinstance(value, TaskResult) else TaskResult(value)


def _correctness_snapshot(
    runtime: Runtime,
    inputs: dict[str, Any],
    fn: Callable[[], TaskResult],
) -> TaskResult:
    runtime.prepare_inputs(inputs)
    runtime.barrier()
    result = fn()
    runtime.torch.cuda.synchronize(runtime.device)
    return _clone_result(result)


@dataclass
class EagerReplica:
    runtime: Runtime
    implementation: str
    fn: Callable[[], TaskResult]

    capture_id: None = None
    stream: None = None
    fallback: bool = False

    def invoke(self) -> TaskResult:
        return self.fn()


@dataclass
class GraphReplica:
    runtime: Runtime
    implementation: str
    graph: Any
    stream: Any
    captured_result: TaskResult
    capture_id: str
    fallback: bool
    reference_delegated: bool
    trusted_config: bool
    details: dict[str, Any]
    input_pointers: dict[str, int]
    output_pointers: dict[str, int]

    def record_replay(self) -> None:
        self.runtime.accounting.graph_replay(
            self.implementation,
            fallback=self.fallback,
            reference_delegated=self.reference_delegated,
            trusted_config=self.trusted_config,
        )

    def replay(self) -> TaskResult:
        self.graph.replay()
        return self.captured_result


class ExecutionPool:
    def __init__(self, replicas: list[EagerReplica | GraphReplica]):
        if not replicas:
            raise ValueError("execution pool requires at least one replica")
        self.replicas = replicas
        self.invocation = 0

    def next(self) -> EagerReplica | GraphReplica:
        replica = self.replicas[self.invocation % len(self.replicas)]
        self.invocation += 1
        return replica

    def reset(self) -> None:
        self.invocation = 0


def _invoke_sync(
    runtime: Runtime,
    pool: ExecutionPool,
) -> tuple[TaskResult, str | None]:
    torch = runtime.torch
    runtime.prepare_inputs(runtime._active_inputs)
    runtime.barrier()
    replica = pool.next()
    if isinstance(replica, GraphReplica):
        current = torch.cuda.current_stream(runtime.device)
        replica.stream.wait_stream(current)
        replica.record_replay()
        with torch.cuda.stream(replica.stream):
            result = replica.replay()
        replica.stream.synchronize()
        current.wait_stream(replica.stream)
        return result, replica.capture_id
    result = replica.invoke()
    torch.cuda.synchronize(runtime.device)
    return result, None


def _time_one(
    runtime: Runtime,
    pool: ExecutionPool,
) -> tuple[float, str | None]:
    torch = runtime.torch
    runtime.prepare_inputs(runtime._active_inputs)
    runtime.barrier()
    replica = pool.next()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if isinstance(replica, GraphReplica):
        current = torch.cuda.current_stream(runtime.device)
        replica.stream.wait_stream(current)
        replica.record_replay()
        with torch.cuda.stream(replica.stream):
            start.record(replica.stream)
            replica.replay()
            end.record(replica.stream)
        end.synchronize()
        current.wait_stream(replica.stream)
        latency = float(start.elapsed_time(end))
        capture_id = replica.capture_id
    else:
        start.record()
        replica.invoke()
        end.record()
        end.synchronize()
        latency = float(start.elapsed_time(end))
        capture_id = None
    return runtime.rank_max(latency), capture_id


def _graph_runtime_preflight(runtime: Runtime) -> None:
    """Load graph/introspection machinery before any audited capture."""
    torch = runtime.torch
    from cuda.bindings import driver as _cuda_driver  # noqa: F401

    stream = torch.cuda.Stream(device=runtime.device)
    value = torch.zeros((1,), device=runtime.device)
    stream.wait_stream(torch.cuda.current_stream(runtime.device))
    # keep_graph=True is required for driver-level node inspection through
    # raw_cuda_graph(); replay lazily instantiates the executable graph.
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        value.add_(1)
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    inspected = inspect_cuda_graph(graph.raw_cuda_graph())
    if inspected["node_count"] < 1:
        raise RuntimeError("CUDA graph preflight captured no nodes")


def _capture_one_graph(
    runtime: Runtime,
    inputs: dict[str, Any],
    fn: Callable[[], TaskResult],
    *,
    implementation: str,
    capture_id: str,
    candidate_artifacts: list[Path],
) -> tuple[GraphReplica, dict[str, Any]]:
    torch = runtime.torch
    stream = torch.cuda.Stream(device=runtime.device)
    current = torch.cuda.current_stream(runtime.device)
    stream.wait_stream(current)

    runtime.accounting.phase = f"{capture_id}:warmup"
    with torch.cuda.stream(stream):
        for _ in range(3):
            runtime.prepare_inputs(inputs)
            fn()
    stream.synchronize()
    current.wait_stream(stream)

    before = runtime_state_snapshot(candidate_artifacts)
    fallback_before = runtime.accounting.candidate_fallbacks
    delegation_before = runtime.accounting.candidate_reference_delegations
    trusted_config_before = runtime.accounting.candidate_trusted_config_calls
    runtime.accounting.phase = f"{capture_id}:capture"
    with torch.cuda.stream(stream):
        runtime.prepare_inputs(inputs)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        captured_result = fn()
    stream.synchronize()
    current.wait_stream(stream)
    after = runtime_state_snapshot(candidate_artifacts)
    observation = runtime_state_delta(
        before,
        after,
        phase=f"{capture_id}:capture",
    )
    if not isinstance(captured_result, TaskResult):
        captured_result = TaskResult(captured_result)
    fallback = (
        implementation == "candidate"
        and runtime.accounting.candidate_fallbacks > fallback_before
    )
    reference_delegated = (
        implementation == "candidate"
        and runtime.accounting.candidate_reference_delegations > delegation_before
    )
    trusted_config = (
        implementation == "candidate"
        and runtime.accounting.candidate_trusted_config_calls > trusted_config_before
    )
    raw_graph_handle = int(graph.raw_cuda_graph())
    inspected = inspect_cuda_graph(raw_graph_handle)
    default_stream = torch.cuda.default_stream(runtime.device)
    details = {
        "capture_id": capture_id,
        "implementation": implementation,
        "raw_graph_handle": raw_graph_handle,
        "stream_id": int(stream.cuda_stream),
        "default_stream_id": int(default_stream.cuda_stream),
        "non_default_stream": int(stream.cuda_stream)
        != int(default_stream.cuda_stream),
        **inspected,
        "stable_input_pointers": False,
        "stable_output_pointers": False,
        "input_mutation_replayed": False,
        "masked_m_mutation_replayed": (
            False
            if runtime.workload.family in ("moe_grouped_masked", "moe_w13_region")
            else None
        ),
        "output_poison_replayed": False,
        "untouched_masked_regions_preserved": (
            False
            if runtime.workload.family in ("moe_grouped_masked", "moe_w13_region")
            else None
        ),
        "deterministic_replay": False,
        "approved_tolerance_passed": False,
        "fallback": fallback,
        "reference_delegated": reference_delegated,
        "trusted_config": trusted_config,
    }
    replica = GraphReplica(
        runtime=runtime,
        implementation=implementation,
        graph=graph,
        stream=stream,
        captured_result=captured_result,
        capture_id=capture_id,
        fallback=fallback,
        reference_delegated=reference_delegated,
        trusted_config=trusted_config,
        details=details,
        input_pointers=_tensor_pointers(inputs, "inputs"),
        output_pointers=_tensor_pointers(captured_result, "output"),
    )
    return replica, observation


def _replay_for_validation(
    replica: GraphReplica,
    inputs: dict[str, Any],
    *,
    poison: bool,
) -> tuple[TaskResult, bool]:
    torch = replica.runtime.torch
    current = torch.cuda.current_stream(replica.runtime.device)
    replica.stream.wait_stream(current)
    poison_visible = False
    with torch.cuda.stream(replica.stream):
        if poison:
            if replica.runtime.workload.family in (
                "moe_grouped_masked",
                "moe_w13_region",
            ):
                replica.runtime.prepare_inputs(inputs)
            else:
                count = _poison_observed(replica.captured_result)
                if count < 1:
                    raise AssertionError("graph output contains no poisonable tensor")
    replica.stream.synchronize()
    if poison:
        if replica.runtime.workload.family in (
            "moe_grouped_masked",
            "moe_w13_region",
        ):
            output_names = ["out"]
            if replica.runtime.workload.family == "moe_w13_region":
                output_names.append("down_out")
            poison_visible = all(
                bool(inputs[name].eq(MOE_OUTPUT_POISON).all().item())
                for name in output_names
            )
        else:
            poison_visible = _poison_is_visible(replica.captured_result)
        if not poison_visible:
            raise AssertionError("graph output poison was not observable before replay")
    replica.record_replay()
    with torch.cuda.stream(replica.stream):
        result = replica.replay()
    replica.stream.synchronize()
    current.wait_stream(replica.stream)
    return result, poison_visible


def _masked_store_limit(count: int, block_m: int, slab: int) -> int:
    if count <= 0:
        return 0
    return min(((count + block_m - 1) // block_m) * block_m, slab)


def _masked_store_blocks(
    runtime: Runtime,
    *,
    implementation: str,
    reference_delegated: bool,
) -> dict[str, int]:
    # The pinned same-source stock W13 and installed W2 heuristics select
    # store_block_m=128 for these exact shapes. The tracked candidate runtime
    # supplies its exact BM. MGroupedMasked predicates CTA scheduling, then
    # stores whole tiles.
    w13_block_m = 128
    if implementation == "candidate" and not reference_delegated:
        if runtime.w13_runtime is None:
            raise AssertionError("candidate W13 store envelope has no runtime")
        w13_block_m = int(runtime.w13_runtime.config[0])
    blocks = {"out": w13_block_m}
    if runtime.workload.family == "moe_w13_region":
        blocks["down_out"] = 128
    return blocks


def _assert_untouched_masked_regions(
    runtime: Runtime,
    inputs: dict[str, Any],
    counts: tuple[int, ...],
    *,
    implementation: str = "reference",
    reference_delegated: bool = False,
) -> dict[str, dict[str, int]]:
    """Check poison outside every scheduled full-tile store envelope."""

    blocks = _masked_store_blocks(
        runtime,
        implementation=implementation,
        reference_delegated=reference_delegated,
    )
    observations = {}
    for output_name, block_m in blocks.items():
        output = inputs[output_name]
        padding_rows_written = 0
        untouched_rows_checked = 0
        slab = int(output.shape[1])
        for expert, count in enumerate(counts):
            store_limit = _masked_store_limit(count, block_m, slab)
            padding = output[expert, count:store_limit]
            if padding.numel():
                padding_rows_written += int(
                    padding.ne(MOE_OUTPUT_POISON).any(dim=-1).sum().item()
                )
            untouched = output[expert, store_limit:]
            untouched_rows_checked += int(untouched.shape[0])
            if not bool(untouched.eq(MOE_OUTPUT_POISON).all().item()):
                raise AssertionError(
                    f"{output_name}[{expert}, {store_limit}:] changed outside "
                    "scheduled masked-store tiles "
                    f"(masked_m={count}, block_m={block_m})"
                )
        observations[output_name] = {
            "store_block_m": block_m,
            "padding_rows_written": padding_rows_written,
            "untouched_rows_checked": untouched_rows_checked,
        }
    return observations


def _validate_graph_replicas(
    runtime: Runtime,
    inputs: dict[str, Any],
    eager_reference_fn: Callable[[], TaskResult],
    reference_snapshot: TaskResult,
    replicas: list[GraphReplica],
) -> None:
    supported_families = {
        "packed_fp8_gemm",
        "moe_grouped_masked",
        "moe_w13_region",
    }
    if runtime.workload.family not in supported_families:
        raise RuntimeError(
            f"schema-v2 graph validation does not support {runtime.workload.family}"
        )
    is_masked = runtime.workload.family in {
        "moe_grouped_masked",
        "moe_w13_region",
    }
    mutation_target = inputs["activation_fp8"] if is_masked else inputs["x_fp8"]
    original = mutation_target.clone()
    original_mask = inputs["masked_m"].clone() if is_masked else None
    runtime.accounting.phase = "graph_validation"
    mutation_target.zero_()
    if is_masked:
        inputs["masked_m"].copy_(inputs["masked_m_replay"])
    runtime.torch.cuda.synchronize(runtime.device)
    mutated_reference = _correctness_snapshot(
        runtime,
        inputs,
        eager_reference_fn,
    )
    if _results_exactly_equal(reference_snapshot, mutated_reference):
        raise AssertionError("post-capture input mutation did not change the output")

    for replica in replicas:
        if _tensor_pointers(inputs, "inputs") != replica.input_pointers:
            raise AssertionError(f"{replica.capture_id}: input pointers changed")
        first_result, poison_visible = _replay_for_validation(
            replica,
            inputs,
            poison=True,
        )
        first_snapshot = _clone_result(first_result)
        _compare_for_workload(
            runtime,
            inputs,
            mutated_reference,
            first_snapshot,
            counts=inputs["masked_m_replay_cpu"] if is_masked else None,
        )
        masked_store_observation = None
        if is_masked:
            masked_store_observation = _assert_untouched_masked_regions(
                runtime,
                inputs,
                inputs["masked_m_replay_cpu"],
                implementation=replica.implementation,
                reference_delegated=replica.reference_delegated,
            )
        if _tensor_pointers(first_result, "output") != replica.output_pointers:
            raise AssertionError(f"{replica.capture_id}: output pointers changed")
        second_result, second_poison_visible = _replay_for_validation(
            replica,
            inputs,
            poison=True,
        )
        second_snapshot = _clone_result(second_result)
        _exact_compare(first_snapshot, second_snapshot)
        _compare_for_workload(
            runtime,
            inputs,
            mutated_reference,
            second_snapshot,
            counts=inputs["masked_m_replay_cpu"] if is_masked else None,
        )
        if is_masked:
            second_masked_store_observation = _assert_untouched_masked_regions(
                runtime,
                inputs,
                inputs["masked_m_replay_cpu"],
                implementation=replica.implementation,
                reference_delegated=replica.reference_delegated,
            )
            if second_masked_store_observation != masked_store_observation:
                raise AssertionError(
                    f"{replica.capture_id}: masked-store envelope changed on replay"
                )
        replica.details.update(
            {
                "stable_input_pointers": (
                    _tensor_pointers(inputs, "inputs") == replica.input_pointers
                ),
                "stable_output_pointers": (
                    _tensor_pointers(second_result, "output") == replica.output_pointers
                ),
                "input_mutation_replayed": True,
                "masked_m_mutation_replayed": True if is_masked else None,
                "output_poison_replayed": poison_visible and second_poison_visible,
                "untouched_masked_regions_preserved": (True if is_masked else None),
                "masked_store_contract": (
                    "poison preserved outside scheduled full store_block_m tiles"
                    if is_masked
                    else None
                ),
                "masked_store_observation": masked_store_observation,
                "deterministic_replay": True,
                "approved_tolerance_passed": True,
            }
        )

    mutation_target.copy_(original)
    if is_masked:
        assert original_mask is not None
        inputs["masked_m"].copy_(original_mask)
    runtime.torch.cuda.synchronize(runtime.device)
    restored_reference = _correctness_snapshot(
        runtime,
        inputs,
        eager_reference_fn,
    )
    _compare_for_workload(
        runtime,
        inputs,
        reference_snapshot,
        restored_reference,
    )
    for replica in replicas:
        replayed, poison_visible = _replay_for_validation(
            replica,
            inputs,
            poison=True,
        )
        _compare_for_workload(
            runtime,
            inputs,
            restored_reference,
            _clone_result(replayed),
        )
        if is_masked:
            _assert_untouched_masked_regions(
                runtime,
                inputs,
                inputs["masked_m_initial_cpu"],
                implementation=replica.implementation,
                reference_delegated=replica.reference_delegated,
            )
        replica.details["output_poison_replayed"] = (
            replica.details["output_poison_replayed"] and poison_visible
        )
        replica.details["approved_tolerance_passed"] = True
        replica.details["stable_input_pointers"] = (
            _tensor_pointers(inputs, "inputs") == replica.input_pointers
        )
        replica.details["stable_output_pointers"] = (
            _tensor_pointers(replayed, "output") == replica.output_pointers
        )


def _build_graph_series(
    runtime: Runtime,
    inputs: dict[str, Any],
    eager_reference_fn: Callable[[], TaskResult],
    eager_candidate_fn: Callable[[], TaskResult],
    reference_snapshot: TaskResult,
    *,
    series_id: str,
    candidate_artifacts: list[Path],
) -> tuple[ExecutionPool, ExecutionPool, dict[str, Any], list[dict[str, Any]]]:
    captures: list[GraphReplica] = []
    observations: list[dict[str, Any]] = []
    capture_plan = (
        ("reference", eager_reference_fn, "R-first"),
        ("candidate", eager_candidate_fn, "C-after-R"),
        ("candidate", eager_candidate_fn, "C-first"),
        ("reference", eager_reference_fn, "R-after-C"),
    )
    for implementation, fn, suffix in capture_plan:
        replica, observation = _capture_one_graph(
            runtime,
            inputs,
            fn,
            implementation=implementation,
            capture_id=f"{series_id}:{suffix}",
            candidate_artifacts=candidate_artifacts,
        )
        captures.append(replica)
        observations.append(observation)
    raw_handles = [int(replica.graph.raw_cuda_graph()) for replica in captures]
    if len(set(raw_handles)) != len(raw_handles):
        raise AssertionError("reference/candidate graph captures are not independent")
    _validate_graph_replicas(
        runtime,
        inputs,
        eager_reference_fn,
        reference_snapshot,
        captures,
    )
    graph_record = {
        "capture_policy": "bidirectional_R-C_then_C-R_round_robin",
        "reference_candidate_captured_independently": True,
        "captures": [replica.details for replica in captures],
    }
    reference_pool = ExecutionPool([captures[0], captures[3]])
    candidate_pool = ExecutionPool([captures[1], captures[2]])
    return reference_pool, candidate_pool, graph_record, observations


def _series_order(start_order: str, pair_index: int) -> str:
    if pair_index % 2 == 0:
        return start_order
    return "BA" if start_order == "AB" else "AB"


def _performance_estimates(
    raw_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute all promotion estimators from complete ordered pairs."""

    if not raw_samples or len(raw_samples) % 2 != 0:
        raise RuntimeError("performance estimates require complete ordered pairs")
    reference_values: list[float] = []
    candidate_values: list[float] = []
    by_order: dict[str, list[float]] = {"AB": [], "BA": []}
    for offset in range(0, len(raw_samples), 2):
        pair = raw_samples[offset : offset + 2]
        order = pair[0].get("order")
        if order not in by_order or pair[1].get("order") != order:
            raise RuntimeError("performance estimate pair order is malformed")
        values = {
            str(sample.get("implementation")): float(sample["latency_ms"])
            for sample in pair
        }
        if set(values) != {"reference", "candidate"}:
            raise RuntimeError("performance estimate pair is incomplete")
        reference_ms = values["reference"]
        candidate_ms = values["candidate"]
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (reference_ms, candidate_ms)
        ):
            raise RuntimeError("performance estimate latency is non-finite")
        ratio = reference_ms / candidate_ms
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise RuntimeError("performance estimate ratio is non-finite")
        reference_values.append(reference_ms)
        candidate_values.append(candidate_ms)
        by_order[order].append(ratio)
    if not by_order["AB"] or not by_order["BA"]:
        raise RuntimeError("both AB and BA samples are required")

    pooled = statistics.median(reference_values) / statistics.median(candidate_values)
    ab_estimate = statistics.median(by_order["AB"])
    ba_estimate = statistics.median(by_order["BA"])
    order_balanced = math.sqrt(ab_estimate * ba_estimate)
    estimates = (pooled, order_balanced, ab_estimate, ba_estimate)
    if not all(math.isfinite(value) and value > 0.0 for value in estimates):
        raise RuntimeError("required performance estimate is non-finite")
    return {
        "contract": "finite_pooled_order_balanced_ab_ba_v1",
        "pair_count": len(raw_samples) // 2,
        "pooled_speedup": pooled,
        "order_balanced_speedup": order_balanced,
        "ab_median_speedup": ab_estimate,
        "ba_median_speedup": ba_estimate,
        "all_finite": True,
    }


def _four_estimator_gate_passes(estimates: dict[str, Any]) -> bool:
    return all(
        math.isfinite(float(estimates[field]))
        and float(estimates[field]) >= PERFORMANCE_THRESHOLD
        for field in (
            "pooled_speedup",
            "order_balanced_speedup",
            "ab_median_speedup",
            "ba_median_speedup",
        )
    )


def _measure_series(
    runtime: Runtime,
    reference_pool: ExecutionPool,
    candidate_pool: ExecutionPool,
    *,
    series_index: int,
    series_id: str,
    execution_mode: str,
    warmup: int,
    repeat: int,
    candidate_artifacts: list[Path],
    graph_record: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pools = {
        "reference": reference_pool,
        "candidate": candidate_pool,
    }
    start_order = "AB" if series_index % 2 == 0 else "BA"
    runtime.accounting.phase = f"{series_id}:warmup"
    for pair_index in range(warmup):
        order = _series_order(start_order, pair_index)
        implementations = (
            ("reference", "candidate") if order == "AB" else ("candidate", "reference")
        )
        for implementation in implementations:
            _invoke_sync(runtime, pools[implementation])
    reference_pool.reset()
    candidate_pool.reset()

    before = runtime_state_snapshot(candidate_artifacts)
    runtime.accounting.phase = f"{series_id}:timing"
    raw_samples: list[dict[str, Any]] = []
    reference_values: list[float] = []
    candidate_values: list[float] = []
    for pair_index in range(repeat):
        order = _series_order(start_order, pair_index)
        implementations = (
            ("reference", "candidate") if order == "AB" else ("candidate", "reference")
        )
        for position, implementation in enumerate(implementations):
            latency_ms, capture_id = _time_one(
                runtime,
                pools[implementation],
            )
            sample: dict[str, Any] = {
                "sequence": len(raw_samples),
                "pair_index": pair_index,
                "position": position,
                "order": order,
                "implementation": implementation,
                "label": "A" if implementation == "reference" else "B",
                "latency_ms": latency_ms,
            }
            if capture_id is not None:
                sample["graph_capture_id"] = capture_id
            raw_samples.append(sample)
            if implementation == "reference":
                reference_values.append(latency_ms)
            else:
                candidate_values.append(latency_ms)
    after = runtime_state_snapshot(candidate_artifacts)
    observation = runtime_state_delta(
        before,
        after,
        phase=f"{series_id}:timing",
    )
    paired_speedups = [
        reference_ms / candidate_ms
        for reference_ms, candidate_ms in zip(
            reference_values,
            candidate_values,
        )
    ]
    median_speedup = statistics.median(paired_speedups)
    performance_estimates = _performance_estimates(raw_samples)
    strict_estimator_pass = _four_estimator_gate_passes(performance_estimates)
    series = {
        "series_index": series_index,
        "series_id": series_id,
        "independent": True,
        "execution_mode": execution_mode,
        "start_order": start_order,
        "warmup_pairs": warmup,
        "repeat": repeat,
        "raw_ordered_samples": raw_samples,
        "reference": latency_summary(reference_values),
        "candidate": latency_summary(candidate_values),
        "paired_speedups": paired_speedups,
        "median_speedup": median_speedup,
        "performance_estimates": performance_estimates,
        # Paired median is diagnostic only. Promotion requires both order
        # strata plus pooled and order-balanced estimates.
        "passes_3pct_gate": strict_estimator_pass,
    }
    if graph_record is not None:
        series["graph"] = graph_record
    return series, observation


def _profile_eager_pool(
    runtime: Runtime,
    pool: ExecutionPool,
    *,
    phase: str,
) -> dict[str, Any]:
    runtime.accounting.phase = phase
    pool.reset()

    def invoke() -> TaskResult:
        result, _capture_id = _invoke_sync(runtime, pool)
        return result

    return profile_cuda_callable(runtime.torch, invoke)


def _candidate_artifacts(module: ModuleType) -> list[Path]:
    return [
        Path(module.__candidate_path__).resolve(),
        *[Path(path).resolve() for path in module.__candidate_artifact_paths__],
    ]


def _render_artifacts(module: ModuleType) -> list[dict[str, Any]]:
    runner = Path(__file__).resolve()
    workloads = (HERE / "workloads.py").resolve()
    candidate = Path(module.__candidate_path__).resolve()
    artifacts = [
        file_artifact("runner", runner),
        file_artifact("workloads", workloads),
        file_artifact("candidate", candidate),
    ]
    seen = {runner, workloads, candidate}
    for index, path_raw in enumerate(module.__candidate_artifact_paths__):
        path = Path(path_raw).resolve()
        if path in seen:
            continue
        seen.add(path)
        artifacts.append(file_artifact(f"candidate_artifact_{index:02d}", path))
    return artifacts


def _write_result(path: Path, rendered: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(rendered + "\n")
    os.replace(temporary, path)


def run_task(args: argparse.Namespace) -> int:
    started_utc = utc_now()
    workload = get_workload(args.task)
    execution_mode = args.execution_mode or "eager"
    if execution_mode not in workload.execution_modes:
        raise RuntimeError(
            f"{workload.name} supports execution modes "
            f"{', '.join(workload.execution_modes)}, not {execution_mode}"
        )
    candidate_path = args.candidate or str(HERE / "candidates" / "reference.py")
    candidate_module = _load_candidate(candidate_path)
    assert candidate_module is not None
    if (
        candidate_module.__candidate_api__ == TRUSTED_CONFIG_CANDIDATE_API
        and workload.family not in {"deepep_normal_dispatch", "deepep_normal_combine"}
    ):
        raise RuntimeError(
            f"{TRUSTED_CONFIG_CANDIDATE_API} is runner-owned and only valid "
            "for DeepEP normal dispatch/combine workloads"
        )
    artifacts_for_snapshot = _candidate_artifacts(candidate_module)
    run_id = (
        started_utc.replace("-", "").replace(":", "").replace(".", "")
        + f"-{uuid.uuid4().hex[:8]}"
    )
    runtime = Runtime(workload, candidate_module)
    if runtime.w13_runtime is not None:
        known = set(artifacts_for_snapshot)
        for module in runtime.w13_runtime.identity["modules"].values():
            cache = Path(module["jit_cache"])
            for relative in module["jit_artifacts"]:
                artifact = (cache / relative).resolve()
                if artifact not in known:
                    known.add(artifact)
                    artifacts_for_snapshot.append(artifact)
    runtime._active_inputs = None
    try:
        hardware = collect_hardware_provenance(runtime.torch, runtime.device)
        clock_samples = [clock_sample()]
        inputs = runtime.build_inputs()
        runtime._active_inputs = inputs
        eager_reference_fn = lambda: runtime.reference(inputs)
        eager_candidate_fn = lambda: _candidate_result(
            candidate_module,
            inputs,
            runtime,
        )

        runtime.accounting.phase = "pre_timing_correctness"
        reference_snapshot = _correctness_snapshot(
            runtime,
            inputs,
            eager_reference_fn,
        )
        candidate_snapshot = _correctness_snapshot(
            runtime,
            inputs,
            eager_candidate_fn,
        )
        _compare_for_workload(
            runtime,
            inputs,
            reference_snapshot,
            candidate_snapshot,
        )

        runtime.accounting.phase = "jit_warmup"
        warmup_before = runtime_state_snapshot(artifacts_for_snapshot)
        eager_reference_pool = ExecutionPool(
            [EagerReplica(runtime, "reference", eager_reference_fn)]
        )
        eager_candidate_pool = ExecutionPool(
            [EagerReplica(runtime, "candidate", eager_candidate_fn)]
        )
        for index in range(max(3, args.warmup)):
            order = (
                ("reference", "candidate")
                if index % 2 == 0
                else ("candidate", "reference")
            )
            for implementation in order:
                _invoke_sync(
                    runtime,
                    (
                        eager_reference_pool
                        if implementation == "reference"
                        else eager_candidate_pool
                    ),
                )
        # CUDA-event setup belongs to warmup, not to an audited timing phase.
        _time_one(runtime, eager_reference_pool)
        _time_one(runtime, eager_candidate_pool)
        if execution_mode == "cuda_graph":
            _graph_runtime_preflight(runtime)
        warmup_after = runtime_state_snapshot(artifacts_for_snapshot)
        warmup_delta = runtime_state_delta(
            warmup_before,
            warmup_after,
            phase="jit_warmup",
        )

        series_results: list[dict[str, Any]] = []
        jit_observations: list[dict[str, Any]] = []
        for series_index in range(args.series):
            series_id = f"{run_id}:series-{series_index + 1:02d}"
            graph_record = None
            if execution_mode == "cuda_graph":
                (
                    reference_pool,
                    candidate_pool,
                    graph_record,
                    capture_observations,
                ) = _build_graph_series(
                    runtime,
                    inputs,
                    eager_reference_fn,
                    eager_candidate_fn,
                    reference_snapshot,
                    series_id=series_id,
                    candidate_artifacts=artifacts_for_snapshot,
                )
                jit_observations.extend(capture_observations)
            else:
                reference_pool = ExecutionPool(
                    [EagerReplica(runtime, "reference", eager_reference_fn)]
                )
                candidate_pool = ExecutionPool(
                    [EagerReplica(runtime, "candidate", eager_candidate_fn)]
                )
            series, timing_observation = _measure_series(
                runtime,
                reference_pool,
                candidate_pool,
                series_index=series_index,
                series_id=series_id,
                execution_mode=execution_mode,
                warmup=args.warmup,
                repeat=args.repeat,
                candidate_artifacts=artifacts_for_snapshot,
                graph_record=graph_record,
            )
            series_results.append(series)
            jit_observations.append(timing_observation)

        runtime.accounting.phase = "post_timing_correctness"
        if execution_mode == "cuda_graph":
            reference_pool.reset()
            candidate_pool.reset()
            post_reference, _ = _invoke_sync(runtime, reference_pool)
            post_candidate, _ = _invoke_sync(runtime, candidate_pool)
            _compare_for_workload(
                runtime,
                inputs,
                reference_snapshot,
                _clone_result(post_reference),
            )
            _compare_for_workload(
                runtime,
                inputs,
                reference_snapshot,
                _clone_result(post_candidate),
            )
        else:
            post_reference = _correctness_snapshot(
                runtime,
                inputs,
                eager_reference_fn,
            )
            post_candidate = _correctness_snapshot(
                runtime,
                inputs,
                eager_candidate_fn,
            )
            _compare_for_workload(
                runtime,
                inputs,
                reference_snapshot,
                post_reference,
            )
            _compare_for_workload(
                runtime,
                inputs,
                reference_snapshot,
                post_candidate,
            )

        runtime.accounting.phase = "fresh_inputs_correctness"
        fresh_inputs = runtime.build_inputs()
        fresh_reference = _correctness_snapshot(
            runtime,
            fresh_inputs,
            lambda: runtime.reference(fresh_inputs),
        )
        fresh_candidate = _correctness_snapshot(
            runtime,
            fresh_inputs,
            lambda: _candidate_result(
                candidate_module,
                fresh_inputs,
                runtime,
            ),
        )
        _compare_for_workload(
            runtime,
            fresh_inputs,
            fresh_reference,
            fresh_candidate,
        )

        kernel_profiles = None
        if execution_mode == "eager":
            kernel_profiles = {
                "reference": _profile_eager_pool(
                    runtime,
                    eager_reference_pool,
                    phase="profiler_reference",
                ),
                "candidate": _profile_eager_pool(
                    runtime,
                    eager_candidate_pool,
                    phase="profiler_candidate",
                ),
            }

        clock_samples.append(clock_sample())
        workload_record = as_dict(workload)
        all_raw_samples = [
            sample
            for series in series_results
            for sample in series["raw_ordered_samples"]
        ]
        all_reference = [
            float(sample["latency_ms"])
            for sample in all_raw_samples
            if sample["implementation"] == "reference"
        ]
        all_candidate = [
            float(sample["latency_ms"])
            for sample in all_raw_samples
            if sample["implementation"] == "candidate"
        ]
        aggregate_estimates = _performance_estimates(all_raw_samples)
        required_estimates_finite = bool(
            aggregate_estimates["all_finite"]
            and all(
                series["performance_estimates"]["all_finite"]
                for series in series_results
            )
        )
        every_series_passes = all(
            series["passes_3pct_gate"] for series in series_results
        )
        identity_control = bool(getattr(candidate_module, "IDENTITY_CONTROL", False))
        implementation_record = runtime.accounting.render(candidate_module)
        fallback_count = implementation_record["candidate"]["fallback_count"]
        reference_delegations = implementation_record["candidate"][
            "reference_delegations"
        ]
        candidate_api = implementation_record["candidate"]["api"]
        performance_gate_passed = (
            every_series_passes
            and required_estimates_finite
            and not identity_control
            and fallback_count == 0
            and (
                reference_delegations == 0
                or candidate_api == TRUSTED_CONFIG_CANDIDATE_API
            )
        )
        hardware["clock_samples"] = clock_samples
        execution_record: dict[str, Any] = {
            "mode": execution_mode,
            "timer": "CUDA events; maximum rank latency for distributed workloads",
            "reference_candidate_captured_separately": (execution_mode == "cuda_graph"),
            "capture_stream": (
                "independent non-default streams"
                if execution_mode == "cuda_graph"
                else None
            ),
            "graph_capture_policy": (
                "bidirectional_R-C_then_C-R_round_robin"
                if execution_mode == "cuda_graph"
                else None
            ),
            "kernel_profiles": kernel_profiles,
        }
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "result_kind": "serving_native_v2",
            "run": {
                "run_id": run_id,
                "started_utc": started_utc,
                "finished_utc": utc_now(),
                "command": [
                    str(Path(sys.executable).resolve()),
                    str(Path(__file__).resolve()),
                    *sys.argv[1:],
                ],
                "cwd": str(Path.cwd().resolve()),
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "requested_series": args.series,
                "warmup": args.warmup,
                "repeat": args.repeat,
            },
            "workload": workload_record,
            "execution": execution_record,
            "correctness": {
                "status": "pass",
                "pre_timing_reference": True,
                "pre_timing_candidate": True,
                "post_timing_reference": True,
                "post_timing_candidate": True,
                "fresh_inputs_post_timing": True,
                "graph_validation": (True if execution_mode == "cuda_graph" else None),
                "tolerance": {
                    "dtype_and_shape_exact": True,
                    "rtol": 2e-2,
                    "atol": 2e-2,
                    "integer": "exact",
                    "deterministic_graph_replay": "exact",
                },
            },
            "provenance": {
                "workload_sha256": canonical_sha256(workload_record),
                "artifacts": _render_artifacts(candidate_module),
                "imports": collect_import_provenance(
                    candidate_module.__name__,
                ),
                "repositories": {
                    "kernel_harness": git_repository(REPO_ROOT),
                    "sglang": git_repository(SGLANG_ROOT),
                },
                "hardware": hardware,
                "jit": {
                    "warmup_completed": True,
                    "warmup_activity": warmup_delta,
                    "capture_or_timing_detected": any(
                        not observation["clean"] for observation in jit_observations
                    ),
                    "observations": jit_observations,
                },
                "cache_paths": {
                    name: os.environ.get(name)
                    for name in (
                        "DG_JIT_CACHE_DIR",
                        "SGLANG_DG_CACHE_DIR",
                        "TRITON_CACHE_DIR",
                        "TORCH_EXTENSIONS_DIR",
                    )
                },
                "w13_runtime": (
                    runtime.w13_runtime.identity
                    if runtime.w13_runtime is not None
                    else None
                ),
            },
            "implementations": implementation_record,
            "series": series_results,
            "reference": latency_summary(all_reference),
            "candidate": {
                "path": candidate_module.__candidate_path__,
                "api": candidate_api,
                "identity_control": identity_control,
                "series_median_speedups": [
                    series["median_speedup"] for series in series_results
                ],
                **latency_summary(all_candidate),
            },
            "aggregate": {
                "required_series": MIN_REQUIRED_SERIES,
                "completed_series": len(series_results),
                "threshold": PERFORMANCE_THRESHOLD,
                "every_series_passes_3pct": every_series_passes,
                "series_gate_contract": ("all_four_estimates_each_series_gte_1p03_v1"),
                "required_estimates_finite": required_estimates_finite,
                "performance_estimates": aggregate_estimates,
                "performance_gate_passed": performance_gate_passed,
                "identity_control_forced_non_win": identity_control,
            },
        }
        if runtime.rank == 0:
            audit = audit_document(result, verify_files=True)
            if not audit["valid"]:
                raise RuntimeError(
                    "generated schema-v2 result failed its own audit: "
                    + "; ".join(audit["errors"])
                )
            result["self_audit"] = audit
            rendered = json.dumps(result, indent=2, sort_keys=True)
            print(rendered, flush=True)
            if args.output:
                _write_result(Path(args.output).expanduser().resolve(), rendered)
        return 0
    finally:
        runtime.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(WORKLOADS))
    parser.add_argument("--candidate")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--series", type=int, default=MIN_REQUIRED_SERIES)
    parser.add_argument(
        "--execution-mode",
        choices=("eager", "cuda_graph"),
        default=None,
    )
    parser.add_argument("--output")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--describe")
    args = parser.parse_args()
    if not args.list and args.describe is None and args.task is None:
        parser.error("one of --list, --describe, or --task is required")
    if args.warmup < 1 or args.repeat < 2:
        parser.error("--warmup must be >= 1 and --repeat must be >= 2")
    if args.series < MIN_REQUIRED_SERIES:
        parser.error(f"--series must be >= {MIN_REQUIRED_SERIES} for the V2 contract")
    return args


def main() -> int:
    args = parse_args()
    if args.list:
        for workload in WORKLOADS.values():
            print(
                f"{workload.name:38s} phase={workload.phase:7s} "
                f"world={workload.world_size} family={workload.family} "
                f"modes={','.join(workload.execution_modes)}"
            )
        return 0
    if args.describe is not None:
        print(
            json.dumps(as_dict(get_workload(args.describe)), indent=2, sort_keys=True)
        )
        return 0
    return run_task(args)


if __name__ == "__main__":
    raise SystemExit(main())
