#!/usr/bin/env python3
"""GPU-free unit checks for the single-SGLANG_DIR resolver."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def _touch(root: Path, rel: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# marker\n")


class TestSglangResolve(unittest.TestCase):
    def test_defaults_have_only_sglang_dir(self):
        self.assertIn("SGLANG_DIR", C._DEFAULTS)
        self.assertNotIn("MM_M3_SGLANG_DIR", C._DEFAULTS)

    def test_resolve_rejects_mm_m3(self):
        with self.assertRaises(KeyError):
            C.resolve("MM_M3_SGLANG_DIR")

    def test_resolve_sglang_dir_ignores_pin(self):
        with mock.patch.dict(os.environ, {"SGLANG_DIR": "/tmp/fake-sglang"}, clear=False):
            # Re-resolve with env override without rewriting module globals.
            path = C.resolve("SGLANG_DIR")
            self.assertTrue(path.endswith("fake-sglang") or "fake-sglang" in path)
            self.assertEqual(C.resolve_sglang_dir("MM_M3_SGLANG_DIR"), path)
            self.assertEqual(C.resolve_sglang_dir(None), path)

    def test_usable_checkout_gate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.assertFalse(C.is_usable_sglang_checkout(root))
            self.assertIsNone(C.sglang_python_root(root))
            _touch(root, "python/sglang/srt/layers/quantization/fp8_kernel.py")
            self.assertTrue(C.is_usable_sglang_checkout(root))
            self.assertEqual(C.sglang_python_root(root), str((root / "python").resolve()))

    def test_m3_markers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.assertFalse(C.has_m3_kernels(root))
            for rel in C.M3_KERNEL_MARKERS:
                _touch(root, rel)
            self.assertTrue(C.has_m3_kernels(root))


if __name__ == "__main__":
    raise SystemExit(unittest.main())
