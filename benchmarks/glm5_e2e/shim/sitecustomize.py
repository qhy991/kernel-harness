"""Auto-loaded by Python on any interpreter that has this directory on
PYTHONPATH — the launcher scripts prepend `benchmarks/glm5_e2e/shim` and
`benchmarks/glm5_e2e/` so this file runs before `python -m
sglang.bench_one_batch` imports sglang.

Order of side effects:
  1. `import glm52_gfx942_shim` applies the seven gfx942 compat patches
     (runs on import; safe to call `.apply()` again).
  2. If `$KDA_E2E_OVERRIDES` names a readable .py file, import it and call
     its `register()` — user-supplied operator patches.

That's the whole coupling. `python -m sglang.bench_one_batch` runs unchanged
after this — no wrapper, no argv rewriting, no re-launching.
"""
from __future__ import annotations

import os
import sys


# 1. gfx942 compatibility shim. Idempotent.
try:
    import glm52_gfx942_shim as _shim  # noqa: F401 — side effects: patches
except Exception as _e:
    print(f"[sitecustomize] gfx942 shim skipped: {type(_e).__name__}: {_e}",
          file=sys.stderr, flush=True)

# 2. User-supplied operator overrides (optional).
_ov_path = os.environ.get("KDA_E2E_OVERRIDES", "").strip()
if _ov_path:
    try:
        from operator_overrides import apply_overrides, load_overrides
        apply_overrides(load_overrides(_ov_path))
    except Exception as _e:
        print(f"[sitecustomize] user overrides FAILED: {type(_e).__name__}: {_e}",
              file=sys.stderr, flush=True)
        raise
