"""Helpers to load harness modules whether evaluate_task used _sibling or packages."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_harness_module(name: str):
    key = f"_tb_{name}"
    if key in sys.modules:
        return sys.modules[key]
    pkg = f"testbench.harness.{name}"
    if pkg in sys.modules:
        return sys.modules[pkg]
    path = Path(__file__).resolve().parents[1] / f"{name}.py"
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod
