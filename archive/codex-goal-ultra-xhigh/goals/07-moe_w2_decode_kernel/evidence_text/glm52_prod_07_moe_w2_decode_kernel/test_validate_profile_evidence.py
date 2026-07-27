#!/usr/bin/env python3
"""CPU-only parser tests for production-W2 profile evidence."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import validate_profile_evidence as validator  # noqa: E402


TARGET = "sm100_fp8_fp4_gemm_1d1d_impl"


class NsysCsvParserTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text)
        return path

    def test_kernel_summary_accepts_raw_nsys_preamble(self) -> None:
        summary = self.write(
            "summary.csv",
            "Generating SQLite file /tmp/report.sqlite from /tmp/report.nsys-rep\n"
            "Processing [/tmp/report.sqlite] with [cuda_gpu_kern_sum.py]...\n"
            "Time (%),Total Time (ns),Instances,Avg (ns),Name\n"
            f'100.0,75200,1,75200.0,"void deep_gemm::{TARGET}(int *)"\n',
        )
        self.assertEqual(
            validator.audit_nsys_extract(summary),
            {"target_rows": 1, "target_occurrences": 1},
        )

    def test_kernel_summary_rejects_missing_header(self) -> None:
        summary = self.write("bad.csv", "Generating SQLite file only\nnot,csv,data\n")
        with self.assertRaisesRegex(RuntimeError, "header missing"):
            validator.audit_nsys_extract(summary)

    def trace_paths(self, *, gpu_corrid: int = 1312) -> tuple[Path, Path, Path]:
        api = self.write(
            "api.csv",
            "Start (ns),Duration (ns),Name,Result,CorrID,Pid,Tid,T-Pri,Thread Name\n"
            "43233372,18843,cuProfilerStart,0,1298,1,1,20,python\n"
            "43703966,78512,cuLaunchKernelEx,0,1312,1,1,20,python\n"
            "43923969,28702,cudaDeviceSynchronize,0,1315,1,1,20,python\n",
        )
        gpu = self.write(
            "gpu.csv",
            "Start (ns),Duration (ns),CorrId,GrdX,GrdY,GrdZ,BlkX,BlkY,BlkZ,Reg/Trd,"
            "StcSMem (MB),DymSMem (MB),Bytes (MB),Throughput (MB/s),SrcMemKd,DstMemKd,"
            "Device,Ctx,GreenCtx,Strm,Name\n"
            f"43769847,75200,{gpu_corrid},148,1,1,256,1,1,36,0.000,0.214,,,,,"
            f"NVIDIA B200 (0),1,,7,{TARGET}\n",
        )
        execution = self.write(
            "exec.csv",
            "API Start (ns),API Dur (ns),Queue Start (ns),Queue Dur (ns),Kernel Start (ns),"
            "Kernel Dur (ns),Total Dur (ns),PID,TID,DevId,API Function,GridXYZ,BlockXYZ,Kernel Name\n"
            f"43703966,78512,,,43769847,75200,141081,1,1,0,cuLaunchKernelEx,"
            f" 148    1    1, 256    1    1,{TARGET}\n",
        )
        return api, gpu, execution

    def test_trace_exports_are_correlated(self) -> None:
        audit = validator.audit_nsys_trace_exports(*self.trace_paths())
        self.assertEqual(audit["correlation_id"], 1312)
        self.assertEqual(audit["kernel_duration_ns"], 75200)
        self.assertEqual(audit["queue_duration_ns"], None)
        self.assertEqual(audit["grid"], [148, 1, 1])
        self.assertEqual(audit["block"], [256, 1, 1])

    def test_trace_exports_reject_unmapped_correlation(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "did not map to one API row"):
            validator.audit_nsys_trace_exports(*self.trace_paths(gpu_corrid=9999))


if __name__ == "__main__":
    unittest.main()
