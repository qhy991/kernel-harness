"""CPU-only adversarial tests for the Task26 stage11-v4 production driver."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVER = (
    HERE.parent
    / "evidence"
    / "glm52_prod_26_moe_w2_decode_scoped_bm16_em8_bm16_stage11_v4"
    / "run_single_b200.sh"
)
LEASE_SENTINEL = "glm52-task26-em8-bm16-stage11-v4-flexible-gpu-lease-v1"
LEAF = "moe_w2_grouped_decode_m32_em8_bm16_stage11_v4"
REGION = "moe_w13_swiglu_w2_region_decode_m32_em8_bm16_stage11_v4"


class Task26Stage11V4DriverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.harness = self.root / "kernel-harness"
        self.sglang = self.root / "sglang"
        self.bin_dir = self.root / "bin"
        self.lock_root = self.root / "locks"
        self.cache_root = self.root / "cache"
        self.campaign_lock = self.cache_root / "run_single_b200_stage11_v4.lock"
        self.attempt = self.cache_root / "ONE_ATTEMPT_CONSUMED"
        self.gpu_lock = self.lock_root / "gpu1.lock"
        self.ready = self.root / ("8" * 64) / "READY"
        self.event_log = self.root / "events.jsonl"
        for directory in (
            self.harness / "serving_native" / "candidates",
            self.sglang,
            self.bin_dir,
            self.lock_root,
            self.cache_root,
            self.ready.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.gpu_lock.write_text("")
        self.ready.write_text('{"status":"READY"}\n')

        self.runner = self.harness / "serving_native" / "runner.py"
        self.auditor = self.harness / "serving_native" / "audit_result.py"
        self.audit_gate = (
            self.harness / "serving_native" / "validate_portfolio_audit.py"
        )
        self.candidate = (
            self.harness
            / "serving_native"
            / "candidates"
            / "moe_w2_em8_bm16_stage11_v4.py"
        )
        self.stock = self.bin_dir / "stock"
        self.nvidia_smi = self.bin_dir / "nvidia-smi"
        self.df = self.bin_dir / "df"
        self.ready_tool = self.bin_dir / "ready_bundle.py"

        self._script(
            self.runner,
            """#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
def value(flag):
    return args[args.index(flag) + 1]
with open(os.environ["TASK26_FAKE_EVENT_LOG"], "a") as stream:
    stream.write(json.dumps({
        "kind": "runner",
        "task": value("--task"),
        "mode": value("--execution-mode"),
        "warmup": value("--warmup"),
        "repeat": value("--repeat"),
        "series": value("--series"),
    }) + "\\n")
pathlib.Path(value("--output")).write_text("{}\\n")
""",
        )
        self._script(
            self.auditor,
            """#!/usr/bin/env python3
print("{}")
""",
        )
        self._script(
            self.audit_gate,
            """#!/usr/bin/env python3
import json, os, sys
with open(os.environ["TASK26_FAKE_EVENT_LOG"], "a") as stream:
    stream.write(json.dumps({"kind": "audit"}) + "\\n")
if os.environ.get("TASK26_FAKE_AUDIT_FAIL") == "1":
    raise SystemExit(17)
""",
        )
        self.candidate.write_text("# fixture candidate\n")
        (self.harness / ".gitignore").write_text("runs/\n")
        (self.sglang / "README.md").write_text("fixture SGLang checkout\n")
        self._script(
            self.stock,
            """#!/usr/bin/env python3
import json, os, subprocess, sys
with open(os.environ["TASK26_FAKE_EVENT_LOG"], "a") as stream:
    stream.write(json.dumps({"kind": "stock"}) + "\\n")
raise SystemExit(subprocess.call(sys.argv[1:]))
""",
        )
        self._script(
            self.nvidia_smi,
            """#!/usr/bin/env python3
import json, os, sys
with open(os.environ["TASK26_FAKE_EVENT_LOG"], "a") as stream:
    stream.write(json.dumps({
        "kind": "gpu",
        "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "argv": sys.argv[1:],
    }) + "\\n")
print("1, GPU-fixture-0001, NVIDIA B200, 590.00, 1800, 3000")
""",
        )
        self._script(
            self.df,
            """#!/usr/bin/env python3
import sys
if "-Pk" in sys.argv:
    print("Filesystem 1024-blocks Used Available Capacity Mounted on")
    print("fixture 100000000 1 99999999 1% /")
else:
    print("Filesystem Size Used Avail Use% Mounted on")
    print("fixture 100G 1G 99G 1% /")
""",
        )
        self._script(
            self.ready_tool,
            """#!/usr/bin/env python3
import hashlib, json, os, pathlib, sys
args = sys.argv[1:]
if not args or args[0] != "verify":
    raise SystemExit(2)
ready = pathlib.Path(args[args.index("--ready") + 1]).resolve()
with open(os.environ["TASK26_FAKE_EVENT_LOG"], "a") as stream:
    stream.write(json.dumps({"kind": "ready", "path": str(ready)}) + "\\n")
if os.environ.get("TASK26_FAKE_READY_CORRUPT") == "1":
    print("corrupt READY", file=sys.stderr)
    raise SystemExit(19)
digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
print(json.dumps({
    "ready_path": str(ready),
    "ready_sha256": hashlib.sha256(ready.read_bytes()).hexdigest(),
    "contract_sha256": digest("contract"),
    "bundle_digest": ready.parent.name,
    "manifest_path": str(ready.parent / "manifest.json"),
    "manifest_sha256": digest("manifest"),
    "source_replay_path": str(ready.parent / "source_replay.json"),
    "source_replay_sha256": digest("replay"),
    "build_provenance_path": str(ready.parent / "build_provenance.json"),
    "build_provenance_sha256": digest("provenance"),
    "stock_package_tree_sha256": digest("stock-package-tree"),
    "candidate_package_tree_sha256": digest("candidate-package-tree"),
    "stock_site": str(ready.parent / "stock/site"),
    "candidate_package": str(ready.parent / "candidate/site/deep_gemm"),
}, sort_keys=True))
""",
        )
        self._commit_repo(self.harness)
        self._commit_repo(self.sglang)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _script(path: Path, source: str) -> None:
        path.write_text(source)
        path.chmod(0o755)

    @staticmethod
    def _commit_repo(repo: Path) -> None:
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=Task26",
                "-c",
                "user.email=task26-v4@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )

    def _invoke(
        self,
        name: str,
        *,
        missing_ready: bool = False,
        corrupt_ready: bool = False,
        audit_fail: bool = False,
    ) -> tuple[subprocess.CompletedProcess[str], Path, list[dict]]:
        run_root = (
            self.harness
            / "runs"
            / "glm52_prod_26_moe_w2_decode_scoped_bm16_em8_bm16_stage11_v4"
            / name
        )
        event_log = self.root / f"{name}.events.jsonl"
        ready = self.root / "missing" / "READY" if missing_ready else self.ready
        env = os.environ.copy()
        env.update(
            {
                "TASK26_DRIVER_TEST_MODE": "1",
                "TASK26_TEST_ROOT": str(self.harness),
                "TASK26_TEST_SGLANG_ROOT": str(self.sglang),
                "TASK26_TEST_PYTHON": sys.executable,
                "TASK26_TEST_STOCK_LAUNCHER": str(self.stock),
                "TASK26_TEST_NVIDIA_SMI": str(self.nvidia_smi),
                "TASK26_TEST_DF": str(self.df),
                "TASK26_TEST_GPU_LOCK_ROOT": str(self.lock_root),
                "TASK26_TEST_CAMPAIGN_LOCK": str(self.campaign_lock),
                "TASK26_TEST_READY_TOOL": str(self.ready_tool),
                "TASK26_TEST_READY_RECORD": str(ready),
                "TASK26_STAGE11_TEST_ATTEMPT_SENTINEL": str(self.attempt),
                "TASK26_RUN_ROOT": str(run_root),
                "TASK26_FAKE_EVENT_LOG": str(event_log),
                "TASK26_FLEXIBLE_GPU_LEASE_SENTINEL": LEASE_SENTINEL,
                "CUDA_VISIBLE_DEVICES": "1",
            }
        )
        if corrupt_ready:
            env["TASK26_FAKE_READY_CORRUPT"] = "1"
        if audit_fail:
            env["TASK26_FAKE_AUDIT_FAIL"] = "1"
        lease_fd = os.open(self.gpu_lock, os.O_RDWR)
        try:
            completed = subprocess.run(
                [str(DRIVER)],
                env=env,
                text=True,
                capture_output=True,
                check=False,
                pass_fds=(lease_fd,),
            )
        finally:
            os.close(lease_fd)
        events = (
            [json.loads(line) for line in event_log.read_text().splitlines()]
            if event_log.exists()
            else []
        )
        return completed, run_root, events

    def test_success_verifies_ready_before_gpu_and_runs_four_lanes(self) -> None:
        completed, run_root, events = self._invoke("success")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(events[0]["kind"], "ready")
        self.assertTrue(any(event["kind"] == "gpu" for event in events))
        runners = [event for event in events if event["kind"] == "runner"]
        self.assertEqual(
            [(event["task"], event["mode"]) for event in runners],
            [
                (LEAF, "eager"),
                (LEAF, "cuda_graph"),
                (REGION, "eager"),
                (REGION, "cuda_graph"),
            ],
        )
        self.assertTrue(
            all(
                event["warmup"] == "3"
                and event["repeat"] == "50"
                and event["series"] == "3"
                for event in runners
            ),
            runners,
        )
        self.assertEqual(len([e for e in events if e["kind"] == "audit"]), 4)
        self.assertTrue((run_root / "TEST_COMPLETE").is_file())
        self.assertTrue((run_root / "ready_evidence.json").is_file())
        self.assertTrue((self.attempt / "TEST_COMPLETE").is_file())

    def test_missing_ready_fails_before_gpu_run_root_or_sentinel(self) -> None:
        completed, run_root, events = self._invoke(
            "missing-ready",
            missing_ready=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(events, [])
        self.assertFalse(run_root.exists())
        self.assertFalse(self.attempt.exists())

    def test_corrupt_ready_fails_before_gpu_run_root_or_sentinel(self) -> None:
        completed, run_root, events = self._invoke(
            "corrupt-ready",
            corrupt_ready=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual([event["kind"] for event in events], ["ready"])
        self.assertFalse(run_root.exists())
        self.assertFalse(self.attempt.exists())

    def test_audit_failure_is_terminal_after_first_lane(self) -> None:
        completed, run_root, events = self._invoke(
            "audit-failure",
            audit_fail=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(
            len([event for event in events if event["kind"] == "runner"]),
            1,
        )
        self.assertTrue((run_root / "FAILED").is_file())
        self.assertTrue((self.attempt / "FAILED").is_file())

    def test_driver_has_no_build_path_and_declares_exact_portfolio(self) -> None:
        source = DRIVER.read_text()
        for forbidden in (
            "build_overlay.sh",
            "build_sgl_deep_gemm",
            "nvcc",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("TASK26_TEST_READY_RECORD", source)
        self.assertIn("ready_evidence.json", source)
        self.assertEqual(source.count(f'"{LEAF}|'), 2)
        self.assertEqual(source.count(f'"{REGION}|'), 2)


if __name__ == "__main__":
    unittest.main()
