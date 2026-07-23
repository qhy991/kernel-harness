"""Runner for fixed, serving-native GLM-5.2 workloads.

Local tasks invoke the exact production kernel ABI.  Distributed tasks must be
launched with torchrun; their latency is the maximum CUDA-event latency across
all ranks, which is the value that gates the serving step.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Optional


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SGLANG_ROOT = Path(os.environ.get("SGLANG_ROOT", "/home/qinhaiyan/sglang")).resolve()
SGLANG_PYTHON = SGLANG_ROOT / "python"
if str(SGLANG_PYTHON) not in sys.path:
    sys.path.insert(0, str(SGLANG_PYTHON))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving_native.workloads import WORKLOADS, Workload, as_dict, get_workload


@dataclass
class TaskResult:
    observed: Any
    state: Any = None


def _load_candidate(path: Optional[str]) -> Optional[ModuleType]:
    if path is None:
        return None
    candidate_path = Path(path).expanduser().resolve()
    if candidate_path.is_dir():
        candidate_path = candidate_path / "candidate.py"
    if not candidate_path.is_file():
        raise FileNotFoundError(f"candidate not found: {candidate_path}")
    spec = importlib.util.spec_from_file_location("serving_native_candidate", candidate_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import candidate: {candidate_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run", None)):
        raise TypeError(f"{candidate_path} must export run(inputs, runtime)")
    return module


class Runtime:
    def __init__(self, workload: Workload):
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

        if not torch.cuda.is_available():
            raise RuntimeError("serving-native workloads require CUDA")
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

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
            (m, k), device=self.device, dtype=self.torch.bfloat16, generator=self._generator(1)
        )
        x_fp8, x_scale = sglang_per_token_group_quant_fp8(
            x_bf16,
            128,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        weight_bf16 = self.torch.randn(
            (n, k), device=self.device, dtype=self.torch.bfloat16, generator=self._generator(2)
        )
        weight_fp8, weight_scale_blocks = deep_gemm.utils.math.per_block_cast_to_fp8(
            weight_bf16, use_ue8m0=True, gran_k=128
        )
        weight_scale = transform_scale_ue8m0(weight_scale_blocks, mn=n)
        del x_bf16, weight_bf16, weight_scale_blocks
        if x_scale.dtype != self.torch.int32 or weight_scale.dtype != self.torch.int32:
            raise RuntimeError("B200 production task requires packed int32 UE8M0 scales")
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
        experts = params["experts_per_rank"]
        assignments = params["valid_assignments"]
        generator = self.torch.Generator(device="cpu")
        generator.manual_seed(20260722)
        expert_ids = self.torch.randint(
            experts, (assignments,), dtype=self.torch.int64, generator=generator
        )
        return self.torch.bincount(expert_ids, minlength=experts).to(
            device=self.device, dtype=self.torch.int32
        )

    def _build_moe_grouped_masked(self) -> dict[str, Any]:
        import deep_gemm

        p = self.workload.params
        experts, slab, k, n = (
            p["experts_per_rank"],
            p["expert_slab"],
            p["k"],
            p["n"],
        )
        masked_m = self._fixed_decode_masked_m(p)
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
        if activation_scale.dtype != self.torch.int32 or weight_scale.dtype != self.torch.int32:
            raise RuntimeError("production MoE task requires packed int32 UE8M0 scales")
        return {
            "activation_fp8": activation_fp8,
            "activation_scale": activation_scale,
            "weight_fp8": weight_fp8,
            "weight_scale": weight_scale,
            "out": self.torch.empty(
                (experts, slab, n), device=self.device, dtype=self.torch.bfloat16
            ),
            "masked_m": masked_m,
            "expected_m": p["expected_m"],
        }

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
        topk_idx = scores.topk(p["topk"], dim=-1, sorted=False).indices.to(self.torch.int64)
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

    def reference(self, inputs: dict[str, Any], *, config: Any = None) -> TaskResult:
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
            return TaskResult(self.torch.nn.functional.linear(inputs["x"], inputs["weight"]))
        if family == "moe_grouped_masked":
            from sglang.srt.layers.deep_gemm_wrapper.entrypoint import (
                grouped_gemm_nt_f8f8bf16_masked,
            )

            grouped_gemm_nt_f8f8bf16_masked(
                (inputs["activation_fp8"], inputs["activation_scale"]),
                (inputs["weight_fp8"], inputs["weight_scale"]),
                inputs["out"],
                inputs["masked_m"],
                inputs["expected_m"],
            )
            valid = [
                inputs["out"][expert, : int(count)]
                for expert, count in enumerate(inputs["masked_m"].tolist())
            ]
            return TaskResult(valid)
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
            return TaskResult(out.squeeze(1) if out.ndim == 4 and out.shape[1] == 1 else out)
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
            combined, event, hook = self.deep_ep_buffer.low_latency_combine(
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

    def _run_deepep_normal_dispatch(self, inputs: dict[str, Any], config: Any) -> TaskResult:
        self._deepep_buffer_facade.set_dispatch_mode_as_normal()
        buffer = self.deep_ep_buffer
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = buffer.get_dispatch_layout(inputs["topk_idx"], self.workload.params["experts"])
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
        observed = (observed_x, recv_ids[valid], recv_weights[valid], tuple(recv_counts))
        return TaskResult(observed, {"handle": handle, "recv_x": recv_x})

    def _run_deepep_ll_dispatch(self, inputs: dict[str, Any]) -> TaskResult:
        self._deepep_buffer_facade.set_dispatch_mode_as_low_latency()
        p = self.workload.params
        recv_x, recv_count, handle, event, hook = self.deep_ep_buffer.low_latency_dispatch(
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
        value = self.torch.tensor([latency_ms], device=self.device, dtype=self.torch.float64)
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
        raise AssertionError(f"output structure differs: {len(ref_items)} != {len(cand_items)}")
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
            ref_f = ref_value.float() if ref_value.dtype.is_floating_point else ref_value
            cand_f = cand_value.float() if cand_value.dtype.is_floating_point else cand_value
            if ref_value.dtype.is_floating_point:
                if not torch.allclose(ref_f, cand_f, rtol=2e-2, atol=2e-2, equal_nan=False):
                    diff = (ref_f - cand_f).abs().max().item()
                    raise AssertionError(f"{ref_name}: max abs diff {diff}")
            elif not torch.equal(ref_value, cand_value):
                raise AssertionError(f"{ref_name}: integer tensor mismatch")
        elif ref_value != cand_value:
            raise AssertionError(f"{ref_name}: {ref_value!r} != {cand_value!r}")


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


def _measure(
    runtime: Runtime,
    inputs: dict[str, Any],
    fn: Callable[[], TaskResult],
    warmup: int,
    repeat: int,
) -> list[float]:
    torch = runtime.torch
    for _ in range(warmup):
        runtime.prepare_inputs(inputs)
        runtime.barrier()
        fn()
        torch.cuda.synchronize(runtime.device)
    values: list[float] = []
    for _ in range(repeat):
        runtime.prepare_inputs(inputs)
        runtime.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(runtime.rank_max(float(start.elapsed_time(end))))
    return values


def _measure_paired(
    runtime: Runtime,
    inputs: dict[str, Any],
    reference_fn: Callable[[], TaskResult],
    candidate_fn: Callable[[], TaskResult],
    warmup: int,
    repeat: int,
) -> tuple[list[float], list[float]]:
    """Interleave A/B samples to reduce clock, cache, and temperature drift."""
    torch = runtime.torch

    def one(fn: Callable[[], TaskResult]) -> float:
        runtime.prepare_inputs(inputs)
        runtime.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        return runtime.rank_max(float(start.elapsed_time(end)))

    for index in range(warmup):
        pair = (
            (reference_fn, candidate_fn)
            if index % 2 == 0
            else (candidate_fn, reference_fn)
        )
        for fn in pair:
            runtime.prepare_inputs(inputs)
            runtime.barrier()
            fn()
            torch.cuda.synchronize(runtime.device)

    reference_values: list[float] = []
    candidate_values: list[float] = []
    for index in range(repeat):
        if index % 2 == 0:
            reference_values.append(one(reference_fn))
            candidate_values.append(one(candidate_fn))
        else:
            candidate_values.append(one(candidate_fn))
            reference_values.append(one(reference_fn))
    return reference_values, candidate_values


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered)) - 1))
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "p95_ms": ordered[p95_index],
    }


def run_task(args: argparse.Namespace) -> int:
    workload = get_workload(args.task)
    candidate_module = _load_candidate(args.candidate)
    runtime = Runtime(workload)
    try:
        inputs = runtime.build_inputs()
        runtime.prepare_inputs(inputs)
        runtime.barrier()
        reference_once = runtime.reference(inputs)
        runtime.torch.cuda.synchronize(runtime.device)
        reference_snapshot = TaskResult(_clone_observed(reference_once.observed))

        candidate_once = None
        if candidate_module is not None:
            runtime.prepare_inputs(inputs)
            candidate_once = candidate_module.run(inputs, runtime)
            if not isinstance(candidate_once, TaskResult):
                candidate_once = TaskResult(candidate_once)
            runtime.torch.cuda.synchronize(runtime.device)
            _compare(reference_snapshot, candidate_once)

        if candidate_module is None:
            reference_values = _measure(
                runtime,
                inputs,
                lambda: runtime.reference(inputs),
                warmup=args.warmup,
                repeat=args.repeat,
            )
            candidate_values = None
        else:
            reference_values, candidate_values = _measure_paired(
                runtime,
                inputs,
                lambda: runtime.reference(inputs),
                lambda: _candidate_result(candidate_module, inputs, runtime),
                warmup=args.warmup,
                repeat=args.repeat,
            )
        result: dict[str, Any] = {
            "schema_version": 1,
            "workload": as_dict(workload),
            "reference": _summary(reference_values),
            "reference_policy": "SGLANG_GLM52_OPT=0 production path",
            "execution_mode": "eager_cuda_event",
            "timing_contract": (
                "interleaved paired A/B; maximum CUDA-event latency across ranks"
                if candidate_module is not None
                else "maximum CUDA-event latency across ranks"
            ),
            "candidate": None,
        }
        if candidate_module is not None:
            assert candidate_values is not None
            candidate_summary = _summary(candidate_values)
            candidate_summary["path"] = str(Path(args.candidate).expanduser().resolve())
            paired_ratios = [
                ref_ms / cand_ms
                for ref_ms, cand_ms in zip(reference_values, candidate_values)
            ]
            ordered_ratios = sorted(paired_ratios)
            candidate_summary["speedup"] = statistics.median(paired_ratios)
            candidate_summary["passes_3pct_median_gate"] = (
                candidate_summary["speedup"] >= 1.03
            )
            candidate_summary["paired_p10_speedup"] = ordered_ratios[
                min(len(ordered_ratios) - 1, int(0.1 * len(ordered_ratios)))
            ]
            candidate_summary["paired_p90_speedup"] = ordered_ratios[
                min(len(ordered_ratios) - 1, max(0, int(0.9 * len(ordered_ratios)) - 1))
            ]
            result["candidate"] = candidate_summary

        if runtime.rank == 0:
            rendered = json.dumps(result, indent=2, sort_keys=True)
            print(rendered, flush=True)
            if args.output:
                output_path = Path(args.output).expanduser().resolve()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(rendered + "\n")
        return 0
    finally:
        runtime.close()


def _candidate_result(module: ModuleType, inputs: dict[str, Any], runtime: Runtime) -> TaskResult:
    value = module.run(inputs, runtime)
    return value if isinstance(value, TaskResult) else TaskResult(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(WORKLOADS))
    parser.add_argument("--candidate")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--output")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--describe")
    args = parser.parse_args()
    if not args.list and args.describe is None and args.task is None:
        parser.error("one of --list, --describe, or --task is required")
    if args.warmup < 0 or args.repeat < 1:
        parser.error("--warmup must be >= 0 and --repeat must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    if args.list:
        for workload in WORKLOADS.values():
            print(
                f"{workload.name:38s} phase={workload.phase:7s} "
                f"world={workload.world_size} family={workload.family}"
            )
        return 0
    if args.describe is not None:
        print(json.dumps(as_dict(get_workload(args.describe)), indent=2, sort_keys=True))
        return 0
    return run_task(args)


if __name__ == "__main__":
    raise SystemExit(main())
