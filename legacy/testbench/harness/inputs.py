"""Axis resolution + input building + output normalization (self-contained).

Owns the definition-format semantics kernel-harness uses (var/const/expr axes, custom
get_inputs entrypoint, positional run() call, and tuple/dict/tensor returns).
"""
from __future__ import annotations

import ast
import operator
from typing import Any

import torch

from .dtypes import to_torch

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def eval_expr(expr: str, vars: dict[str, int]) -> int:
    """Safely evaluate an arithmetic axis expression over other axes (+ - * / // % ** ())."""
    def ev(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise TypeError(f"bad constant {node.value!r}")
        if isinstance(node, ast.Name):
            if node.id in vars:
                return vars[node.id]
            raise NameError(f"unknown axis {node.id}")
        if isinstance(node, ast.BinOp):
            return _BIN[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return _UN[type(node.op)](ev(node.operand))
        raise TypeError(f"unsupported expression node {type(node).__name__}")
    v = ev(ast.parse(expr, mode="eval").body)
    return int(v)


def resolve_axes(definition: dict, workload_axes: dict[str, int]) -> dict[str, int]:
    """const values + workload var values, then evaluate expr axes in dependency order."""
    axes = definition["axes"]
    resolved = {n: a["value"] for n, a in axes.items() if a["type"] == "const"}
    for n, a in axes.items():
        if a["type"] == "var":
            if n not in workload_axes:
                raise ValueError(f"workload missing var axis {n!r}")
            resolved[n] = workload_axes[n]
    # expr axes may reference each other; iterate to a fixpoint (max = #expr axes passes)
    exprs = {n: a["expression"] for n, a in axes.items() if a["type"] == "expr"}
    for _ in range(len(exprs) + 1):
        progressed = False
        for n, e in exprs.items():
            if n in resolved:
                continue
            try:
                resolved[n] = eval_expr(e, resolved)
                progressed = True
            except NameError:
                pass
        if all(n in resolved for n in exprs):
            break
        if not progressed:
            missing = [n for n in exprs if n not in resolved]
            raise ValueError(f"cannot resolve expr axes {missing}")
    return resolved


def build_inputs(get_inputs, definition: dict, resolved_axes: dict, device) -> list:
    """Call the recipe's custom get_inputs, return inputs positionally in definition order."""
    d = get_inputs(dict(resolved_axes), device)
    order = list(definition["inputs"].keys())
    missing = [k for k in order if k not in d]
    if missing:
        raise KeyError(f"get_inputs did not return {missing}")
    return [d[k] for k in order]


def normalize_outputs(out: Any, definition: dict, device) -> list[torch.Tensor]:
    """run()'s return -> list of tensors matched positionally to definition.outputs order."""
    names = list(definition["outputs"].keys())
    dtypes = {k: to_torch(v["dtype"]) for k, v in definition["outputs"].items()}

    def to_t(name, v):
        if isinstance(v, torch.Tensor):
            return v.to(device) if v.device != torch.device(device) else v
        return torch.tensor(v, dtype=dtypes[name], device=device)

    if isinstance(out, dict):
        return [to_t(n, out[n]) for n in names]
    if isinstance(out, (tuple, list)):
        if len(out) != len(names):
            raise RuntimeError(f"run() returned {len(out)} outputs but {len(names)} defined")
        return [to_t(n, v) for n, v in zip(names, out)]
    # single tensor / scalar
    if len(names) != 1:
        raise RuntimeError("run() returned a single value but multiple outputs are defined")
    return [to_t(names[0], out)]
