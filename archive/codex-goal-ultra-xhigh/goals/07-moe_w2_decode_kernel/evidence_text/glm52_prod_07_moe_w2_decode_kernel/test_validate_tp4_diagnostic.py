#!/usr/bin/env python3
"""CPU-only parser tests for the goal-07 TP4 evidence validator."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import validate_tp4_diagnostic as validator  # noqa: E402


class NsysKernelSummaryTest(unittest.TestCase):
    def test_accepts_status_preamble_and_name_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.csv"
            path.write_text(
                "Generating SQLite file /tmp/report.sqlite from /tmp/report.nsys-rep\n"
                "Processing [/tmp/report.sqlite] with [cuda_gpu_kern_sum.py]...\n"
                "Time (%),Total Time (ns),Instances,Name\n"
                "100.0,1234,1,sm100_fp8_fp4_gemm_1d1d_impl\n"
            )
            rows, name_field = validator.read_nsys_kernel_summary(path)
            self.assertEqual(name_field, "Name")
            self.assertEqual(rows[0][name_field], "sm100_fp8_fp4_gemm_1d1d_impl")

    def test_rejects_missing_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.csv"
            path.write_text("Generating SQLite file only\nnot,a,summary\n")
            with self.assertRaisesRegex(RuntimeError, "header missing"):
                validator.read_nsys_kernel_summary(path)

    def test_timing_summary_rejects_median_above_p95(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "median exceeds p95"):
            validator.validate_summary(
                {"min_ms": 1.0, "median_ms": 3.0, "p95_ms": 2.0},
                "synthetic",
            )

    def test_workload_log_requires_diagnostic_environment_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "workload.log"
            path.write_text("Only use 20 SMs for DeepEP communication\n")
            with self.assertRaisesRegex(RuntimeError, "ibgda_transport"):
                validator.validate_workload_log(path)


if __name__ == "__main__":
    unittest.main()
