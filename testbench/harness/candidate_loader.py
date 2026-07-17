"""Resolve a task's candidate kernel to a single `run(inputs) -> out` callable.

The canonical contract is `candidate.py` exposing `run(inputs: dict)`, where
`inputs` is the frozen dict from glm52_ops.build_inputs. That is the
one file an agent edits.

Everything else here is migration scaffolding. `solution.py` is the legacy
testbench contract (`run()` takes the reference's positional arguments), and
optimised solutions already exist on the kda/* branches; adapting it keeps that
work measurable instead of silently invisible. Resolution order is fixed and the
chosen path is always reported, so "which file did this number come from" is
never ambiguous:

    candidate.py  ->  impl.py  ->  solution.py (adapted)  ->  reference

Falling through to `reference` means the task measures the backend against
itself: cosine 1.0 and speedup ~1.0. That is a meaningful baseline, not a pass.
"""
from __future__ import annotations

import importlib.util
import inspect
from functools import partial
from pathlib import Path
from typing import Callable

_HARNESS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _HARNESS_DIR.parents[1]

_spec = importlib.util.spec_from_file_location("_tb_glm52_ops", _HARNESS_DIR / "glm52_ops.py")
ops = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ops)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _takes_inputs_dict(fn) -> bool:
    params = list(inspect.signature(fn).parameters)
    return len(params) == 1 and params[0] in ("inputs", "inp", "input")


def _adapt_positional(mod, fam: str, phase: str) -> Callable:
    """Wrap a legacy solution.py `run(<positional args>)` as `run(inputs)`."""
    if fam == "gemm":
        def run(inputs: dict):
            out = mod.run(inputs["x_fp8"], inputs["x_scale"],
                          inputs["w_fp8"], inputs["w_scale"])
            return out if out is not None else inputs.get("out")
        return run

    if fam == "bmm":
        def run(inputs: dict):
            return mod.run(inputs["A_fp8"], inputs["B_fp8"],
                           inputs["A_scale"], inputs["B_scale"])
        return run

    if fam == "moe":
        def run(inputs: dict):
            out = mod.run(inputs["x_fp8"], inputs["x_scale"],
                          inputs["w_fp8"], inputs["w_scale"],
                          inputs["masked_m"], inputs["expected_m"])
            return out if out is not None else inputs.get("out")
        return run

    if fam == "mla":
        def run(inputs: dict):
            return mod.run(inputs["q"], inputs["kv"], inputs["indices"],
                           inputs["sm_scale"], inputs["d_v"])
        return run

    if fam == "score":
        if phase == "prefill":
            def run(inputs: dict):
                return mod.run(inputs["q_fp8"], (inputs["k_fp8"], inputs["k_scale"]),
                               inputs["weights"], inputs["ks"], inputs["ke"],
                               clean_logits=False)
            return run

        def run(inputs: dict):
            return mod.run(inputs["q_fp8"], inputs["kv_cache_fp8"], inputs["weights"],
                           inputs["seqlens"], inputs["block_tables"],
                           inputs["schedule_metadata"], inputs["max_seq_len"],
                           clean_logits=False)
        return run

    raise ValueError(f"unknown family {fam}")


def load(task_dir: Path, op: str, phase: str) -> tuple[Callable, str, Path | None]:
    """Return (run_callable, source_label, source_path)."""
    fam = ops.family(op)

    for name in ("candidate.py", "impl.py"):
        path = task_dir / name
        if path.is_file():
            mod = _load_module(path, f"{name[:-3]}_{task_dir.name}")
            if not hasattr(mod, "run"):
                raise AttributeError(f"{path} defines no run(inputs)")
            if not _takes_inputs_dict(mod.run):
                raise TypeError(
                    f"{path}: run() must take a single `inputs` dict "
                    f"(got {list(inspect.signature(mod.run).parameters)})")
            return mod.run, str(path.relative_to(_REPO_ROOT)), path

    sol = task_dir / "solution.py"
    if sol.is_file():
        try:
            mod = _load_module(sol, f"solution_{task_dir.name}")
        except Exception as exc:
            raise RuntimeError(
                f"cannot import {sol} (legacy solution.py often pulls in sglang "
                f"top-level imports). Port it to candidate.py with run(inputs)."
            ) from exc
        if not hasattr(mod, "run"):
            raise AttributeError(f"{sol} defines no run()")
        fn = mod.run if _takes_inputs_dict(mod.run) else _adapt_positional(mod, fam, phase)
        return fn, str(sol.relative_to(_REPO_ROOT)), sol

    return partial(ops.reference, op, phase), "reference", None
