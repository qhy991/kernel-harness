"""Translate task-schema dtype strings to and from torch.dtype."""
import torch

_STR_TO_DTYPE = {
    "float64": torch.float64, "float32": torch.float32, "float16": torch.float16,
    "bfloat16": torch.bfloat16, "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "int64": torch.int64, "int32": torch.int32, "int16": torch.int16, "int8": torch.int8,
    "uint8": torch.uint8, "bool": torch.bool,
    # NVFP4 is represented in SGLang/FlashInfer as two packed e2m1 values per
    # byte plus separate FP8 block scales and FP32 global scales. Keep the task
    # schema explicit while mapping tensor storage to the actual packed dtype.
    "nvfp4": torch.uint8,
}
# optional newer dtypes
for _n in ("float4_e2m1fn_x2", "uint16", "uint32", "uint64"):
    if hasattr(torch, _n):
        _STR_TO_DTYPE[_n] = getattr(torch, _n)


def to_torch(dtype_str: str) -> torch.dtype:
    if not dtype_str:
        raise ValueError("empty dtype string")
    dt = _STR_TO_DTYPE.get(dtype_str)
    if dt is None:
        raise ValueError(f"unknown dtype: {dtype_str!r}")
    return dt
