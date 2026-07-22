"""GLM-5.2 operator definitions — **shim that routes to the per-platform impl**.

Historically this file was 1100 lines with 15 `IS_ROCM` branches. It has been
split into two independent modules so CUDA and AMD data flow, dtypes, reference
kernels, and inputs schemas cannot silently share a code path:

  * `testbench/harness/glm52_ops_cuda.py`  — CUDA/B200, `float8_e4m3fn`,
    deep_gemm + sgl_kernel, TMA/UE8M0 scale format
  * `testbench/harness/glm52_ops_amd.py`   — AMD/MI300X gfx942, `float8_e4m3fnuz`,
    aiter, `FP8_MAX=224.0`, no UE8M0

This shim exists **only** to preserve the ~15 existing
`from testbench.harness import glm52_ops` import sites — task candidates, tools,
and the evaluation harness — so they keep working without knowing which
platform is active. Any new code that is already platform-committed should
import the concrete module directly (`glm52_ops_cuda` or `glm52_ops_amd`),
not this shim.

Route is decided at import time by `get_backend().profile.platform`, which
reads `KERNEL_HARNESS_PLATFORM` (default `cuda`) via
`testbench/harness/backends/registry.py`.
"""
from __future__ import annotations

from testbench.harness.backends import get_backend

_PLATFORM = get_backend().profile.platform

if _PLATFORM == "cuda":
    from testbench.harness.glm52_ops_cuda import *  # noqa: F401,F403
    from testbench.harness import glm52_ops_cuda as _impl  # noqa: F401
elif _PLATFORM == "rocm":
    from testbench.harness.glm52_ops_amd import *  # noqa: F401,F403
    from testbench.harness import glm52_ops_amd as _impl  # noqa: F401
else:
    raise RuntimeError(
        f"glm52_ops shim: unsupported platform {_PLATFORM!r}. "
        "Set KERNEL_HARNESS_PLATFORM to 'cuda' or 'rocm', or import a "
        "concrete impl module (glm52_ops_cuda / glm52_ops_amd) directly."
    )

# Names the concrete impl exports but that `import *` skips (leading underscore
# or not in __all__): expose them explicitly so `glm52_ops._build_gemm` etc.
# keep working for the sync tool and validate_marks.
_INTERNAL = (
    "_round128", "_cast_to_fp8_rows", "_build_gemm", "_build_bmm", "_build_moe",
    "_build_moe_total", "_build_mla", "_build_score", "_main", "_diagnose",
    "_finish", "_wrap",
)
for _name in _INTERNAL:
    if hasattr(_impl, _name):
        globals()[_name] = getattr(_impl, _name)
del _name, _INTERNAL
