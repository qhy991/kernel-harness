"""ROCm / AMD MI300X backend for the GLM-5.2 harness contracts.

This module mirrors the AMD rewardbench standard under
``origin/main:rewardbench/amd``:

* MI300X roofline constants: HBM 5.3 TB/s, FP8 2.6149 PFLOP/s, BF16 1.3074
  PFLOP/s.
* gfx942 FP8 uses ``torch.float8_e4m3fnuz`` with the deployed scale convention
  ``FP8_MAX = 224.0``.
* AMD does not use Blackwell UE8M0 / power-of-two scales.
* Timing is HIP graph capture/replay by default, with HIP event fallback.

The provider keeps correctness runnable through torch-native fallbacks when
AITER is not installed. Those fallbacks are for harness plumbing; official
baseline numbers still need to be collected on an MI300X ROCm environment.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import math
import os
import sys
from functools import lru_cache
from pathlib import Path

import torch

from .base import DeviceProfile


os.environ.setdefault("SGLANG_USE_AITER", "1")
_REPO = Path(__file__).resolve().parents[3]
_CACHE_ROOT = _REPO / "reports" / "cache" / "rocm_amd"
for _name in ("aiter_configs", "triton", "sglang", "xdg"):
    (_CACHE_ROOT / _name).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("AITER_CONFIG_DIR", str(_CACHE_ROOT / "aiter_configs"))
os.environ.setdefault("TRITON_CACHE_DIR", str(_CACHE_ROOT / "triton"))
os.environ.setdefault("SGLANG_CACHE_DIR", str(_CACHE_ROOT / "sglang"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))
_LOCAL_MOE_CONFIG_DIR = _REPO / "rewardbench" / "amd" / "sglang_moe_configs"
if _LOCAL_MOE_CONFIG_DIR.exists():
    os.environ.setdefault("SGLANG_MOE_CONFIG_DIR", str(_LOCAL_MOE_CONFIG_DIR))

FP8_MAX = 224.0
BLOCK = 128


PROFILE = DeviceProfile(
    id="amd-mi300x",
    platform="rocm",
    accelerator="AMD MI300X",
    deployment="MI300X-DP1-TP1-EP32",
    fp8_dtype_name="float8_e4m3fnuz",
    peaks={
        "hbm_bytes_per_s": 5.3e12,
        "fp8": 2.6149e15,
        "bf16": 1.3074e15,
    },
    peaks_source=(
        "rewardbench/amd MI300X constants: HBM 5.3 TB/s, FP8 e4m3 "
        "2614.9 TFLOP/s, BF16 1307.4 TFLOP/s"
    ),
)


BASELINE_CAVEAT = (
    "AMD/SGLang MI300X baseline: FP8 fnuz with FP8_MAX=224.0 and plain fp32 "
    "scales (no UE8M0). Dense FP8 GEMM is routed through "
    "sglang.srt.layers.quantization.fp8_utils.aiter_w8a8_block_fp8_linear, "
    "which selects AITER Triton blockscale GEMM on gfx942. Sparse MLA attempts "
    "SGLang's AMD default tilelang DSA backend. Torch/AITER CK fallbacks are "
    "kept only to keep the harness debuggable when the production modules are "
    "not importable."
)


ACCEPTED_CANDIDATE_FORMS = (
    "Python / PyTorch / Triton ROCm - a .py defining run(inputs)",
    "HIP/C++ extension - a directory holding candidate.py plus extension sources; "
    "candidate.py owns compilation/import outside the timed kernel body",
)


def _pkg_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _fp8_dtype():
    return getattr(torch, PROFILE.fp8_dtype_name)


def _scale_min(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(min=1e-12)


def _scaled_mm(x_fp8, w_fp8_nk, x_scale, w_scale, out_dtype=torch.bfloat16):
    """hipBLASLt FP8 GEMM: x[M,K] @ w[N,K].T -> out[M,N]."""
    return torch._scaled_mm(
        x_fp8,
        w_fp8_nk.t(),
        scale_a=x_scale,
        scale_b=w_scale,
        out_dtype=out_dtype,
    )


def _quant_per_tensor(x: torch.Tensor):
    scale = _scale_min(x.abs().float().amax()) / FP8_MAX
    q = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(_fp8_dtype())
    return q.contiguous(), scale.view(1).to(x.device)


def _dequant_tensor(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x_fp8.float() * scale.float().reshape(-1)[0]


def _dequant_token_blockwise(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    m, k = x_fp8.shape
    if scale.ndim == 0 or scale.numel() == 1:
        return _dequant_tensor(x_fp8, scale)
    return (
        x_fp8.float()
        .view(m, k // BLOCK, BLOCK)
        .mul(scale.float().view(m, k // BLOCK, 1))
        .view(m, k)
    )


def _dequant_block_blockwise(w_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    n, k = w_fp8.shape
    if scale.ndim == 0 or scale.numel() == 1:
        return _dequant_tensor(w_fp8, scale)
    n_blocks = scale.shape[0]
    padded_n = n_blocks * BLOCK
    if padded_n != n:
        padded = torch.zeros(padded_n, k, dtype=w_fp8.dtype, device=w_fp8.device)
        padded[:n] = w_fp8
    else:
        padded = w_fp8
    out = (
        padded.float()
        .view(n_blocks, BLOCK, k // BLOCK, BLOCK)
        .mul(scale.float()[:, None, :, None])
        .view(padded_n, k)
    )
    return out[:n].contiguous()


def _blockwise_reference_mm(x_fp8, w_fp8, x_scale, w_scale):
    x = _dequant_token_blockwise(x_fp8, x_scale)
    w = _dequant_block_blockwise(w_fp8, w_scale)
    return (x @ w.t()).to(torch.bfloat16)


def _add_source_tree(env_name: str, default: str, suffix: str = "") -> None:
    root = Path(os.environ.get(env_name, default)).expanduser()
    path = root / suffix if suffix else root
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


@lru_cache(maxsize=1)
def _sglang_gemm_fn():
    _add_source_tree("SGLANG_ROOT", "/opt/devmachine/lichangye/repos/sglang", "python")
    _add_source_tree("AITER_ROOT", "/opt/devmachine/lichangye/repos/aiter")
    try:
        from sglang.srt.layers.quantization.fp8_utils import (
            aiter_w8a8_block_fp8_linear,
        )
    except Exception:
        return None
    return aiter_w8a8_block_fp8_linear


def _try_sglang_gemm(x_fp8, w_fp8, x_scale, w_scale):
    fn = _sglang_gemm_fn()
    if fn is None:
        return None
    return fn(
        x_fp8,
        w_fp8,
        [BLOCK, BLOCK],
        w_scale,
        x_scale,
    )


@lru_cache(maxsize=1)
def _aiter_ck_gemm_fn():
    try:
        mod = importlib.import_module("aiter.ops.gemm_op_a8w8")
        return getattr(mod, "gemm_a8w8_blockscale")
    except Exception:
        return None


def _try_aiter_ck_gemm(x_fp8, w_fp8, x_scale, w_scale):
    fn = _aiter_ck_gemm_fn()
    if fn is None:
        return None
    return fn(x_fp8, w_fp8, x_scale, w_scale, dtype=torch.bfloat16)


def _normalize_tilelang_sparse_out(out: torch.Tensor) -> torch.Tensor:
    return out.squeeze(0) if out.ndim == 4 and out.shape[0] == 1 else out


@lru_cache(maxsize=1)
def _tilelang_sparse_fwd_fn():
    _add_source_tree("SGLANG_ROOT", "/opt/devmachine/lichangye/repos/sglang", "python")
    try:
        from sglang.srt.layers.attention.dsa.tilelang_kernel import (
            tilelang_sparse_fwd,
        )
    except Exception:
        return None
    return tilelang_sparse_fwd


def _try_sglang_tilelang_sparse_mla(inputs: dict):
    fn = _tilelang_sparse_fwd_fn()
    if fn is None:
        return None

    kv = inputs["kv"]
    if kv.ndim == 2:
        kv = kv.unsqueeze(1)
    indices = inputs["indices"]
    if indices.ndim == 2:
        indices = indices.unsqueeze(1)
    return _normalize_tilelang_sparse_out(
        fn(
            q=inputs["q"],
            kv=kv,
            indices=indices.to(torch.int32),
            sm_scale=float(inputs["sm_scale"]),
            d_v=int(inputs["d_v"]),
        )
    )


class AmdRewardbenchProvider:
    id = "aiter-torch-reference"
    platform = "rocm"
    capabilities = frozenset({"gemm", "bmm", "moe", "moe_fused", "mla", "score"})
    required_modules: tuple[str, ...] = ()
    baseline_caveat = BASELINE_CAVEAT
    accepted_candidate_forms = ACCEPTED_CANDIDATE_FORMS

    def supports(self, op: str, phase: str) -> bool:
        del op, phase
        return True

    def baseline_name(self, family: str, phase: str) -> str:
        if family == "gemm":
            return "sglang.fp8_utils.aiter_w8a8_block_fp8_linear"
        if family == "bmm":
            return "per-head torch._scaled_mm loop"
        if family == "moe":
            return "per-expert torch._scaled_mm loop"
        if family == "moe_fused":
            return "sglang.fused_moe total Routed Expert Gate+Up/Down"
        if family == "mla":
            return "sglang.tilelang_sparse_fwd / gather+chunked fallback"
        if phase == "prefill":
            return "aiter.ops.triton.fp8_mqa_logits"
        return "batched torch._scaled_mm + weighted sum (paged-cost decode)"

    def per_token_cast(self, tensor, *, use_ue8m0: bool):
        del use_ue8m0
        m, k = tensor.shape
        if k % BLOCK:
            raise ValueError(f"ROCm blockwise FP8 expects K divisible by {BLOCK}, got {k}")
        blocks = tensor.float().view(m, k // BLOCK, BLOCK)
        scale = _scale_min(blocks.abs().amax(dim=-1)) / FP8_MAX
        q = (
            blocks.div(scale.unsqueeze(-1))
            .clamp(-FP8_MAX, FP8_MAX)
            .to(_fp8_dtype())
            .view(m, k)
        )
        return q.contiguous(), scale.contiguous()

    def per_block_cast(self, tensor, *, use_ue8m0: bool):
        del use_ue8m0
        n, k = tensor.shape
        if k % BLOCK:
            raise ValueError(f"ROCm blockwise FP8 expects K divisible by {BLOCK}, got {k}")
        n_ceil = math.ceil(n / BLOCK) * BLOCK
        if n_ceil != n:
            padded = torch.zeros(n_ceil, k, dtype=tensor.dtype, device=tensor.device)
            padded[:n] = tensor
        else:
            padded = tensor
        blocks = padded.float().view(n_ceil // BLOCK, BLOCK, k // BLOCK, BLOCK)
        scale = _scale_min(blocks.abs().amax(dim=(1, 3))) / FP8_MAX
        q = (
            blocks.div(scale[:, None, :, None])
            .clamp(-FP8_MAX, FP8_MAX)
            .to(_fp8_dtype())
            .view(n_ceil, k)
        )
        return q[:n].contiguous(), scale.contiguous()

    def align_scale(self, scale):
        return scale.contiguous()

    def paged_mqa_metadata(self, seqlens, block_size: int):
        del seqlens, block_size
        return None

    def reference(self, op: str, phase: str, family: str, inputs: dict):
        del op
        if family == "gemm":
            y = _try_sglang_gemm(
                inputs["x_fp8"],
                inputs["w_fp8"],
                inputs["x_scale"],
                inputs["w_scale"],
            )
            if y is None:
                y = _try_aiter_ck_gemm(
                    inputs["x_fp8"],
                    inputs["w_fp8"],
                    inputs["x_scale"],
                    inputs["w_scale"],
                )
            if y is None:
                try:
                    y = _scaled_mm(
                        inputs["x_fp8"],
                        inputs["w_fp8"],
                        inputs["x_scale"],
                        inputs["w_scale"],
                    )
                except Exception:
                    y = _blockwise_reference_mm(
                        inputs["x_fp8"],
                        inputs["w_fp8"],
                        inputs["x_scale"],
                        inputs["w_scale"],
                    )
            return y

        if family == "bmm":
            return _bmm_reference(inputs)

        if family == "moe":
            return _moe_reference(inputs)

        if family == "moe_fused":
            return _fused_moe_reference(inputs)

        if family == "mla":
            y = _try_sglang_tilelang_sparse_mla(inputs)
            if y is not None:
                return y
            return _sparse_mla_reference(inputs)

        del phase
        return _mqa_score_reference(inputs)

    def version_info(self):
        return {
            "aiter": _pkg_version("aiter"),
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
        }


def _bmm_reference(inputs: dict):
    a = inputs["A_fp8"]
    b = inputs["B_fp8"]
    b_scale = inputs["B_scale"]
    outs = []
    for h in range(a.shape[0]):
        w_nk = b[h].t().contiguous()
        scale_b = b_scale if b_scale.numel() == 1 else b_scale[h]
        try:
            y = _scaled_mm(a[h], w_nk, inputs["A_scale"], scale_b)
        except Exception:
            y = (
                _dequant_tensor(a[h], inputs["A_scale"])
                @ _dequant_tensor(w_nk, scale_b).t()
            ).to(torch.bfloat16)
        outs.append(y)
    return torch.stack(outs, dim=0)


def _moe_reference(inputs: dict):
    out = inputs["out"]
    out.zero_()
    masked_m = inputs["masked_m"]
    for expert in range(inputs["E"]):
        rows = int(masked_m[expert].item())
        if rows <= 0:
            continue
        x = inputs["x_fp8"][expert, :rows]
        w = inputs["w_fp8"][expert]
        x_scale = inputs["x_scale"][expert]
        w_scale = inputs["w_scale"][expert]
        try:
            y = _scaled_mm(x, w, x_scale, w_scale)
        except Exception:
            y = _blockwise_reference_mm(x, w, x_scale, w_scale)
        out[expert, :rows].copy_(y)
    return out


@lru_cache(maxsize=1)
def _aiter_mqa_logits_fn():
    _add_source_tree("AITER_ROOT", "/opt/devmachine/lichangye/repos/aiter")
    try:
        from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits
    except Exception:
        return None
    return fp8_mqa_logits


def _try_aiter_mqa_logits(inputs: dict):
    fn = _aiter_mqa_logits_fn()
    if fn is None or "k_fp8" not in inputs:
        return None
    q = inputs["q_fp8"]
    weights = inputs["weights"]
    if weights.ndim == 3:
        weights_2d = weights.squeeze(-1)
    else:
        weights_2d = weights
    if q.ndim == 2:
        q = q.view(weights_2d.shape[0], weights_2d.shape[1], -1)
    k_scale = inputs["k_scale"]
    if k_scale.numel() == 1:
        k_scale = k_scale.expand(inputs["k_fp8"].shape[0]).contiguous()
    return fn(
        q,
        inputs["k_fp8"],
        k_scale,
        weights_2d,
        inputs["ks"],
        inputs["ke"],
    )


def _ensure_sglang_server_args() -> None:
    _add_source_tree("SGLANG_ROOT", "/opt/devmachine/lichangye/repos/sglang", "python")
    from sglang.srt.server_args import (
        ServerArgs,
        get_global_server_args,
        set_global_server_args_for_scheduler,
    )

    try:
        get_global_server_args()
    except ValueError:
        set_global_server_args_for_scheduler(
            ServerArgs(model_path=os.environ.get("SGLANG_DUMMY_MODEL_PATH", "dummy"))
        )


def _fused_moe_reference(inputs: dict):
    _ensure_sglang_server_args()
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    topk_output = inputs.get("topk_output")
    if topk_output is None:
        topk_output = StandardTopKOutput(
            inputs["topk_weights"],
            inputs["topk_ids"],
            inputs["router_logits"],
        )
    cfg = inputs.get("moe_runner_config")
    if cfg is None:
        cfg = MoeRunnerConfig(
            **inputs["moe_config_kwargs"],
            params_dtype=inputs["hidden_states"].dtype,
        )
    return fused_moe(
        hidden_states=inputs["hidden_states"],
        w1=inputs["w1"],
        w2=inputs["w2"],
        topk_output=topk_output,
        moe_runner_config=cfg,
        use_fp8_w8a8=True,
        w1_scale=inputs["w1_scale"],
        w2_scale=inputs["w2_scale"],
        a1_scale=inputs["a1_scale"],
        a2_scale=inputs["a2_scale"],
    )


def _sparse_mla_reference(inputs: dict):
    q = inputs["q"]
    kv = inputs["kv"].view(inputs["kv"].shape[0], inputs["kv"].shape[-1])
    indices = inputs["indices"].view(inputs["indices"].shape[0], -1).long()
    sm_scale = float(inputs["sm_scale"])
    d_v = int(inputs["d_v"])
    s_q, n_heads, _ = q.shape
    topk = indices.shape[1]
    out = torch.empty(s_q, n_heads, d_v, dtype=torch.bfloat16, device=q.device)
    chunk = 256 if s_q >= 256 else s_q
    for start in range(0, s_q, chunk):
        end = min(start + chunk, s_q)
        gathered = kv[indices[start:end].reshape(-1)].view(end - start, topk, -1)
        q_chunk = q[start:end]
        scores = torch.einsum("chd,ckd->chk", q_chunk, gathered).float() * sm_scale
        probs = torch.softmax(scores, dim=-1).to(torch.bfloat16)
        out[start:end].copy_(torch.einsum("chk,ckd->chd", probs, gathered[..., :d_v]))
    return out


def _mqa_score_reference(inputs: dict):
    y = _try_aiter_mqa_logits(inputs)
    if y is not None:
        return y

    q = inputs["q_fp8"]
    if q.ndim == 3:
        m, h, hd = q.shape
        q = q.reshape(m * h, hd)
    else:
        h = inputs["weights"].shape[1]
        m = inputs["weights"].shape[0]
        hd = q.shape[-1]
    del hd
    k = inputs["k_fp8"]
    s = k.shape[0]
    q_scale = inputs.get(
        "q_scale",
        torch.ones(1, dtype=torch.float32, device=q.device),
    )
    k_scale = inputs["k_scale"]
    weights = inputs["weights"]
    if weights.ndim == 2:
        weights = weights.unsqueeze(-1)
    out = torch.empty(m, s, dtype=torch.float32, device=q.device)
    chunk = 8192
    q_deq = q.float()
    if q_scale.numel() != 1:
        q_deq = q_deq * q_scale.float().reshape(-1, 1)
    else:
        q_deq = q_deq * q_scale.float().reshape(1)
    for start in range(0, s, chunk):
        end = min(start + chunk, s)
        kc = k[start:end]
        if k_scale.numel() == 1:
            k_deq = _dequant_tensor(kc, k_scale)
        else:
            k_deq = kc.float() * k_scale[start:end].float().view(-1, 1)
        lg = (q_deq @ k_deq.t()).view(m, h, -1).relu()
        out[:, start:end] = (lg * weights).sum(dim=1)
    return out


class RocmBenchTimer:
    platform = "rocm"
    contract_id = "hipgraph-or-event-median"
    contract_description = (
        "HIP graph capture+replay by default, falling back to HIP event timing; "
        "setup/cloning is outside the measured region"
    )

    @property
    def id(self) -> str:
        return self.contract_id

    @property
    def description(self) -> str:
        return self.contract_description

    def available(self) -> bool:
        # Keep import/describe/sync usable on non-ROCm hosts. measure() enforces
        # the actual runtime requirement.
        return True

    def measure(self, fn, *, setup, warmup: int, rep: int, device):
        if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
            raise RuntimeError("ROCm/HIP device required for the AMD MI300X timer")
        torch.cuda.set_device(device)
        prefer_graph = os.environ.get("AMD_BENCH_NO_GRAPH", "0") != "1"
        if prefer_graph:
            try:
                return [self._measure_graph(fn, setup, warmup, rep)]
            except Exception:
                torch.cuda.synchronize()
        return self._measure_event(fn, setup, warmup, rep)

    def _call(self, fn, setup, args):
        if setup is None:
            return fn()
        return fn(args)

    def _next_args(self, setup):
        return None if setup is None else setup()

    def _measure_graph(self, fn, setup, warmup: int, rep: int) -> float:
        torch.cuda.synchronize()
        for _ in range(warmup):
            self._call(fn, setup, self._next_args(setup))
        torch.cuda.synchronize()

        captured_args = self._next_args(setup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(rep):
                self._call(fn, setup, captured_args)
        torch.cuda.synchronize()

        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / rep

    def _measure_event(self, fn, setup, warmup: int, rep: int) -> list[float]:
        torch.cuda.synchronize()
        for _ in range(warmup):
            self._call(fn, setup, self._next_args(setup))
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        for i in range(rep):
            args = self._next_args(setup)
            torch.cuda.synchronize()
            starts[i].record()
            self._call(fn, setup, args)
            ends[i].record()
        torch.cuda.synchronize()
        return [starts[i].elapsed_time(ends[i]) for i in range(rep)]


PROVIDER = AmdRewardbenchProvider()
TIMER = RocmBenchTimer()
