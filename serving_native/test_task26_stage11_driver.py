"""CPU-only contracts for the one-attempt Task26 stage11 portfolio."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from serving_native.validate_portfolio_audit import validate_audit_document

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = (
    REPO_ROOT
    / "evidence"
    / "glm52_prod_26_moe_w2_decode_scoped_bm16_em8_bm16_stage11_v3"
    / "run_single_b200.sh"
)
AUDIT_GATE = REPO_ROOT / "serving_native/validate_portfolio_audit.py"
LEASE_SENTINEL = "glm52-task26-em8-bm16-stage11-v3-flexible-gpu-lease-v1"
TASK_CACHE_ROOT = Path(
    "/home/qinhaiyan/glm52-v2-goal-runs/cache/"
    "26-moe_w2_decode_scoped_bm16/em8_bm16_stage11_v3"
)
EXPECTED_LANES = [
    ("moe_w2_grouped_decode_m32_em8_bm16_stage11", "eager"),
    ("moe_w2_grouped_decode_m32_em8_bm16_stage11", "cuda_graph"),
    (
        "moe_w13_swiglu_w2_region_decode_m32_em8_bm16_stage11",
        "eager",
    ),
]


class Task26Stage11DriverContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.temp = Path(self.temporary.name)
        self.harness = self.temp / "kernel-harness"
        self.sglang = self.temp / "sglang"
        self.bin_dir = self.temp / "bin"
        self.gpu_lock_root = self.temp / "gpu-locks"
        self.campaign_lock = self.temp / "cache" / "run_single_b200.lock"
        self.attempt_root = self.temp / "attempts"
        self.bin_dir.mkdir()
        self.gpu_lock_root.mkdir()
        self.campaign_lock.parent.mkdir()
        self.attempt_root.mkdir()
        for gpu in range(4):
            (self.gpu_lock_root / f"gpu{gpu}.lock").touch()

        (self.harness / ".gitignore").parent.mkdir(parents=True)
        (self.harness / ".gitignore").write_text("runs/\n")
        candidate = (
            self.harness / "serving_native/candidates/moe_w2_em8_bm16_stage11.py"
        )
        candidate.parent.mkdir(parents=True)
        candidate.write_text("# candidate fixture\n")
        self._write_fake_runner(self.harness / "serving_native/runner.py")
        self._write_fake_auditor(self.harness / "serving_native/audit_result.py")
        shutil.copyfile(
            AUDIT_GATE,
            self.harness / "serving_native/validate_portfolio_audit.py",
        )
        (self.sglang / "tracked.txt").parent.mkdir(parents=True)
        (self.sglang / "tracked.txt").write_text("clean\n")
        self._init_git(self.harness)
        self._init_git(self.sglang)

        self.fake_stock = self.bin_dir / "fake_stock"
        self.fake_nvidia_smi = self.bin_dir / "fake_nvidia_smi"
        self.fake_df = self.bin_dir / "fake_df"
        self._write_fakes()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _init_git(path: Path) -> None:
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "-c",
                "user.name=Task26 Driver Test",
                "-c",
                "user.email=task26-driver-test@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )

    @staticmethod
    def _write_executable(path: Path, contents: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(contents).lstrip())
        path.chmod(0o755)

    def _write_fake_runner(self, path: Path) -> None:
        self._write_executable(
            path,
            r"""
            import json
            import os
            import sys
            from pathlib import Path

            def option(args, name):
                return args[args.index(name) + 1]

            args = sys.argv[1:]
            task = option(args, "--task")
            mode = option(args, "--execution-mode")
            payload = {
                "kind": "runner",
                "argv": sys.argv,
                "task": task,
                "mode": mode,
                "dg": os.environ.get("DG_JIT_CACHE_DIR"),
                "sglang_dg": os.environ.get("SGLANG_DG_CACHE_DIR"),
                "triton": os.environ.get("TRITON_CACHE_DIR"),
                "torch_extensions": os.environ.get("TORCH_EXTENSIONS_DIR"),
            }
            with Path(os.environ["TASK26_FAKE_EVENT_LOG"]).open("a") as stream:
                stream.write(json.dumps(payload, sort_keys=True) + "\n")
            if os.environ.get("TASK26_FAKE_RUNNER_FAIL") == f"{task}|{mode}":
                raise SystemExit(17)
            if os.environ.get("TASK26_FAKE_MUTATE_ATTEMPT") == "1":
                sentinel = Path(
                    os.environ["TASK26_STAGE11_TEST_ATTEMPT_SENTINEL"]
                )
                (sentinel / "CLAIMED").write_text("tampered\n")
            output = Path(option(args, "--output"))
            output.write_text(json.dumps({"task": task, "mode": mode}))
            """,
        )

    def _write_fake_auditor(self, path: Path) -> None:
        self._write_executable(
            path,
            r"""
            import json
            import os
            import sys
            from pathlib import Path

            result = Path(sys.argv[-1])
            with Path(os.environ["TASK26_FAKE_EVENT_LOG"]).open("a") as stream:
                stream.write(json.dumps(
                    {"kind": "audit", "argv": sys.argv, "result": str(result)},
                    sort_keys=True,
                ) + "\n")
            if os.environ.get("TASK26_FAKE_AUDITOR_FAIL") == result.stem:
                raise SystemExit(19)
            if os.environ.get("TASK26_FAKE_AUDITOR_MALFORMED") == result.stem:
                print("{malformed")
                raise SystemExit(0)
            performance_gate_passed = (
                os.environ.get("TASK26_FAKE_AUDITOR_NON_WIN") != result.stem
            )
            print(json.dumps({
                "valid": True,
                "performance_gate_passed": performance_gate_passed,
                "result": str(result),
            }))
            """,
        )

    def _write_fakes(self) -> None:
        self._write_executable(
            self.fake_stock,
            r"""
            #!/usr/bin/env python3
            import json
            import os
            import sys
            from pathlib import Path

            with Path(os.environ["TASK26_FAKE_EVENT_LOG"]).open("a") as stream:
                stream.write(json.dumps(
                    {"kind": "stock", "argv": sys.argv[1:]},
                    sort_keys=True,
                ) + "\n")
            os.execv(sys.argv[1], sys.argv[1:])
            """,
        )
        self._write_executable(
            self.fake_nvidia_smi,
            r"""
            #!/usr/bin/env python3
            import json
            import os
            import sys
            from pathlib import Path

            state = Path(os.environ["TASK26_FAKE_GPU_STATE"])
            count = int(state.read_text()) + 1 if state.exists() else 1
            state.write_text(str(count))
            with Path(os.environ["TASK26_FAKE_EVENT_LOG"]).open("a") as stream:
                stream.write(json.dumps(
                    {
                        "kind": "gpu",
                        "argv": sys.argv[1:],
                        "count": count,
                        "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    },
                    sort_keys=True,
                ) + "\n")
            index = os.environ["CUDA_VISIBLE_DEVICES"]
            drift_at = int(os.environ.get("TASK26_FAKE_UUID_DRIFT_AT", "0"))
            uuid = "GPU-DRIFTED" if drift_at and count >= drift_at else "GPU-STABLE"
            name = os.environ.get("TASK26_FAKE_GPU_NAME", "NVIDIA B200")
            sm_clock = os.environ.get(
                "TASK26_FAKE_SM_CLOCK", str(1800 + count)
            )
            memory_clock = os.environ.get(
                "TASK26_FAKE_MEMORY_CLOCK", str(2600 + count)
            )
            print(
                f"{index}, {uuid}, {name}, 590.00, "
                f"{sm_clock}, {memory_clock}"
            )
            if os.environ.get("TASK26_FAKE_GPU_MULTIPLE") == "1":
                print(
                    f"{index}, GPU-SECOND, NVIDIA B200, 590.00, 1800, 2600"
                )
            """,
        )
        self._write_executable(
            self.fake_df,
            r"""
            #!/usr/bin/env python3
            import json
            import os
            import sys
            from pathlib import Path

            with Path(os.environ["TASK26_FAKE_EVENT_LOG"]).open("a") as stream:
                stream.write(json.dumps(
                    {"kind": "df", "argv": sys.argv[1:]},
                    sort_keys=True,
                ) + "\n")
            if "-Pk" in sys.argv:
                available = (
                    7 * 1024 * 1024
                    if os.environ.get("TASK26_FAKE_DISK_LOW") == "1"
                    else 9 * 1024 * 1024
                )
                print("Filesystem 1024-blocks Used Available Capacity Mounted on")
                print(f"fixture 20000000 1000 {available} 1% /")
            else:
                print("Filesystem Size Used Avail Use% Mounted on")
                print("fixture 20G 1G 19G 5% /")
            """,
        )

    def _run_root(self, name: str) -> Path:
        return (
            self.harness
            / "runs/glm52_prod_26_moe_w2_decode_scoped_bm16_em8_bm16_stage11_v3"
            / name
        )

    def _invoke(
        self,
        name: str,
        *,
        sentinel: str | None = LEASE_SENTINEL,
        cuda_visible_devices: str = "1",
        lease_fd: str = "correct",
        run_root_override: Path | None = None,
        attempt_sentinel_override: Path | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], Path, list[dict]]:
        run_root = run_root_override or self._run_root(name)
        event_log = self.temp / f"{name}.events.jsonl"
        gpu_state = self.temp / f"{name}.gpu-count"
        attempt_sentinel = (
            attempt_sentinel_override or self.attempt_root / f"{name}.one-attempt"
        )
        env = os.environ.copy()
        for key in (
            "TASK26_FLEXIBLE_GPU_LEASE_SENTINEL",
            "TASK26_FAKE_RUNNER_FAIL",
            "TASK26_FAKE_AUDITOR_FAIL",
            "TASK26_FAKE_AUDITOR_NON_WIN",
            "TASK26_FAKE_AUDITOR_MALFORMED",
            "TASK26_FAKE_UUID_DRIFT_AT",
            "TASK26_FAKE_GPU_MULTIPLE",
            "TASK26_FAKE_GPU_NAME",
            "TASK26_FAKE_SM_CLOCK",
            "TASK26_FAKE_MEMORY_CLOCK",
            "TASK26_FAKE_DISK_LOW",
            "TASK26_FAKE_MUTATE_ATTEMPT",
            "TASK26_STAGE11_TEST_ATTEMPT_SENTINEL",
        ):
            env.pop(key, None)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
                "TASK26_DRIVER_TEST_MODE": "1",
                "TASK26_TEST_ROOT": str(self.harness),
                "TASK26_TEST_SGLANG_ROOT": str(self.sglang),
                "TASK26_TEST_PYTHON": sys.executable,
                "TASK26_TEST_STOCK_LAUNCHER": str(self.fake_stock),
                "TASK26_TEST_NVIDIA_SMI": str(self.fake_nvidia_smi),
                "TASK26_TEST_DF": str(self.fake_df),
                "TASK26_TEST_GPU_LOCK_ROOT": str(self.gpu_lock_root),
                "TASK26_TEST_CAMPAIGN_LOCK": str(self.campaign_lock),
                "TASK26_STAGE11_TEST_ATTEMPT_SENTINEL": str(attempt_sentinel),
                "TASK26_RUN_ROOT": str(run_root),
                "TASK26_FAKE_EVENT_LOG": str(event_log),
                "TASK26_FAKE_GPU_STATE": str(gpu_state),
            }
        )
        if sentinel is not None:
            env["TASK26_FLEXIBLE_GPU_LEASE_SENTINEL"] = sentinel
        if extra_env:
            env.update(extra_env)

        inherited_fd: int | None = None
        if lease_fd != "missing":
            gpu = cuda_visible_devices if cuda_visible_devices.isdigit() else "1"
            if lease_fd == "wrong":
                gpu = "0" if gpu != "0" else "1"
            lock_path = self.gpu_lock_root / f"gpu{gpu}.lock"
            inherited_fd = os.open(lock_path, os.O_WRONLY)
            fcntl.flock(inherited_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            completed = subprocess.run(
                ["bash", str(DRIVER)],
                cwd=REPO_ROOT,
                env=env,
                pass_fds=(() if inherited_fd is None else (inherited_fd,)),
                text=True,
                capture_output=True,
                check=False,
            )
        finally:
            if inherited_fd is not None:
                os.close(inherited_fd)
        events = (
            [json.loads(line) for line in event_log.read_text().splitlines()]
            if event_log.exists()
            else []
        )
        return completed, run_root, events

    @staticmethod
    def _arg_value(argv: list[str], option: str) -> str:
        return argv[argv.index(option) + 1]

    def _assert_failed_test_artifact(self, run_root: Path) -> None:
        self.assertTrue((run_root / "TEST_ONLY").is_file())
        self.assertTrue((run_root / "IN_PROGRESS").is_file())
        self.assertTrue((run_root / "FAILED").is_file())
        self.assertFalse((run_root / "TEST_COMPLETE").exists())
        self.assertFalse((run_root / "COMPLETE").exists())
        attempt = self.attempt_root / f"{run_root.name}.one-attempt"
        self.assertTrue((attempt / "CLAIMED").is_file())
        self.assertTrue((attempt / "FAILED").is_file())
        self.assertFalse((attempt / "TEST_COMPLETE").exists())

    def test_exact_portfolio_mapping_and_guards(self) -> None:
        self.assertTrue(os.access(DRIVER, os.X_OK))
        completed, run_root, events = self._invoke("success")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(
            "TEST_ONLY task26 stage11 CPU driver simulation",
            completed.stderr,
        )
        self.assertNotIn("PASS task26", completed.stderr)

        execution_events = [
            event for event in events if event["kind"] in {"stock", "runner", "audit"}
        ]
        self.assertEqual(
            [event["kind"] for event in execution_events],
            ["stock", "runner", "audit"] * len(EXPECTED_LANES),
        )
        stocks = [event for event in events if event["kind"] == "stock"]
        runners = [event for event in events if event["kind"] == "runner"]
        audits = [event for event in events if event["kind"] == "audit"]
        self.assertEqual(
            [(event["task"], event["mode"]) for event in runners],
            EXPECTED_LANES,
        )
        self.assertEqual(len(stocks), len(EXPECTED_LANES))
        self.assertEqual(len(audits), len(EXPECTED_LANES))

        expected_runner = self.harness / "serving_native/runner.py"
        expected_candidate = (
            self.harness / "serving_native/candidates/moe_w2_em8_bm16_stage11.py"
        )
        expected_auditor = self.harness / "serving_native/audit_result.py"
        for stock, runner, audit, lane in zip(
            stocks, runners, audits, EXPECTED_LANES, strict=True
        ):
            task, mode = lane
            self.assertEqual(stock["argv"][:2], [sys.executable, str(expected_runner)])
            argv = runner["argv"]
            self.assertEqual(Path(argv[0]), expected_runner)
            self.assertEqual(self._arg_value(argv, "--task"), task)
            self.assertEqual(self._arg_value(argv, "--execution-mode"), mode)
            self.assertEqual(
                Path(self._arg_value(argv, "--candidate")),
                expected_candidate,
            )
            self.assertEqual(self._arg_value(argv, "--warmup"), "3")
            self.assertEqual(self._arg_value(argv, "--repeat"), "10")
            self.assertEqual(self._arg_value(argv, "--series"), "3")
            self.assertEqual(runner["dg"], str(TASK_CACHE_ROOT / "deepgemm"))
            self.assertEqual(runner["sglang_dg"], str(TASK_CACHE_ROOT / "deepgemm"))
            self.assertEqual(runner["triton"], str(TASK_CACHE_ROOT / "triton"))
            self.assertEqual(
                runner["torch_extensions"],
                str(TASK_CACHE_ROOT / "torch_extensions"),
            )
            self.assertEqual(Path(audit["argv"][0]), expected_auditor)
            self.assertEqual(audit["argv"][1], "--json")
            self.assertEqual(
                Path(audit["result"]),
                Path(self._arg_value(argv, "--output")),
            )

        gpu_events = [event for event in events if event["kind"] == "gpu"]
        self.assertEqual(len(gpu_events), 8)
        for event in gpu_events:
            self.assertEqual(event["visible"], "1")
            self.assertEqual(
                event["argv"],
                [
                    "-i",
                    "1",
                    (
                        "--query-gpu=index,uuid,name,driver_version,"
                        "clocks.current.sm,clocks.current.memory"
                    ),
                    "--format=csv,noheader,nounits",
                ],
            )
        disk_gates = [
            event
            for event in events
            if event["kind"] == "df" and "-Pk" in event["argv"]
        ]
        self.assertEqual(len(disk_gates), 4)

        expected_stages = ["initial"]
        for task, mode in EXPECTED_LANES:
            stem = f"{task}__{mode}"
            expected_stages.extend([f"before:{stem}", f"after:{stem}"])
        expected_stages.append("final")
        snapshot_rows = (run_root / "gpu_snapshots.tsv").read_text().splitlines()
        self.assertEqual(
            [row.split("\t")[1] for row in snapshot_rows[1:]],
            expected_stages,
        )
        self.assertTrue(all("NVIDIA B200" in row for row in snapshot_rows[1:]))
        self.assertTrue(all("590.00" in row for row in snapshot_rows[1:]))

        self.assertTrue((run_root / "TEST_ONLY").is_file())
        self.assertTrue((run_root / "TEST_COMPLETE").is_file())
        self.assertFalse((run_root / "COMPLETE").exists())
        self.assertFalse((run_root / "IN_PROGRESS").exists())
        self.assertFalse((run_root / "FAILED").exists())
        self.assertTrue((run_root / "artifact_sha256.txt").is_file())
        attempt = self.attempt_root / "success.one-attempt"
        self.assertTrue((attempt / "CLAIMED").is_file())
        self.assertTrue((attempt / "TEST_COMPLETE").is_file())
        self.assertFalse((attempt / "COMPLETE").exists())

        environment = (run_root / "environment.txt").read_text()
        self.assertIn("artifact_class=TEST_ONLY", environment)
        self.assertIn("driver_test_mode=1", environment)
        self.assertIn(f"stock_launcher={self.fake_stock}", environment)
        self.assertIn(f"runner={expected_runner}", environment)
        self.assertIn(f"candidate={expected_candidate}", environment)
        self.assertIn(f"auditor={expected_auditor}", environment)
        self.assertIn(
            f"audit_gate={self.harness / 'serving_native/validate_portfolio_audit.py'}",
            environment,
        )
        self.assertIn("warmup=3\nrepeat=10\nseries=3\n", environment)
        self.assertIn("gpu_uuid=GPU-STABLE", environment)
        self.assertIn("gpu_name=NVIDIA B200", environment)
        self.assertIn("gpu_driver_version=590.00", environment)
        self.assertIn(f"campaign_lock={self.campaign_lock}", environment)
        self.assertIn("variant=em8_bm16_stage11", environment)
        self.assertIn("variant_version=3", environment)
        self.assertIn(
            f"persistent_one_attempt_sentinel={attempt}",
            environment,
        )
        self.assertIn("predeclared_fallback=em8_bm16_stage10", environment)
        self.assertIn("fallback_eligible=0", environment)
        self.assertIn(
            f"wrapper_gpu_lock={self.gpu_lock_root / 'gpu1.lock'}",
            environment,
        )
        completion = (run_root / "completion.txt").read_text()
        self.assertIn("artifact_class=TEST_ONLY", completion)
        self.assertIn("driver_test_mode=1", completion)
        self.assertIn("gpu_final_sm_clock_mhz=", completion)
        self.assertIn("gpu_final_memory_clock_mhz=", completion)

    def test_requires_exact_wrapper_sentinel_and_one_visible_gpu(self) -> None:
        for name, sentinel, visible in (
            ("missing-sentinel", None, "1"),
            ("wrong-sentinel", "wrong", "1"),
            ("multiple-visible", LEASE_SENTINEL, "0,1"),
            ("empty-visible", LEASE_SENTINEL, ""),
        ):
            with self.subTest(name=name):
                completed, run_root, events = self._invoke(
                    name,
                    sentinel=sentinel,
                    cuda_visible_devices=visible,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertFalse(run_root.exists())
                self.assertEqual(events, [])

        source = DRIVER.read_text()
        self.assertIn(
            "/home/qinhaiyan/glm52-goal-runs/with_flexible_gpu.sh --",
            source,
        )
        self.assertIn(
            "env TASK26_FLEXIBLE_GPU_LEASE_SENTINEL="
            "glm52-task26-em8-bm16-stage11-v3-flexible-gpu-lease-v1",
            source,
        )
        self.assertIn(
            'readonly PRODUCTION_GPU_LOCK_ROOT="/home/qinhaiyan/glm52-goal-runs/locks"',
            source,
        )
        self.assertIn(
            'readonly PRODUCTION_ROOT="/home/qinhaiyan/glm52-v2-goal-runs/'
            'worktrees/26-moe-w2-decode-scoped-bm16/kernel-harness"',
            source,
        )
        self.assertIn(
            'readonly PRODUCTION_SGLANG_ROOT="/home/qinhaiyan/'
            "glm52-v2-goal-runs/worktrees/26-moe-w2-decode-scoped-bm16/"
            'sglang"',
            source,
        )
        self.assertIn(
            'readonly TASK_SHARED_CACHE_ROOT="/home/qinhaiyan/glm52-v2-goal-runs/'
            'cache/26-moe_w2_decode_scoped_bm16"',
            source,
        )
        self.assertIn(
            'readonly TASK_CACHE_ROOT="${TASK_SHARED_CACHE_ROOT}/em8_bm16_stage11_v3"',
            source,
        )
        self.assertIn(
            'third_party/deepgemm_w2_em8_bm16_stage11/run_with_exact_post1_stock.sh"',
            source,
        )
        self.assertIn(
            'CAMPAIGN_LOCK="${TASK_SHARED_CACHE_ROOT}/run_single_b200.lock"',
            source,
        )
        self.assertIn(
            'ATTEMPT_SENTINEL="${TASK_CACHE_ROOT}/ONE_ATTEMPT_CONSUMED"',
            source,
        )

    def test_requires_inherited_exact_gpu_lock_fd(self) -> None:
        for name, lease_fd in (
            ("missing-lease-fd", "missing"),
            ("wrong-lease-fd", "wrong"),
        ):
            with self.subTest(name=name):
                completed, run_root, events = self._invoke(name, lease_fd=lease_fd)
                self.assertNotEqual(completed.returncode, 0)
                self.assertFalse(run_root.exists())
                self.assertFalse(any(event["kind"] == "gpu" for event in events))
                self.assertFalse(any(event["kind"] == "runner" for event in events))

    def test_campaign_lock_blocks_a_second_bundle_before_gpu_or_root(self) -> None:
        campaign_fd = os.open(
            self.campaign_lock,
            os.O_WRONLY | os.O_CREAT,
            0o644,
        )
        fcntl.flock(campaign_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            completed, run_root, events = self._invoke("campaign-busy")
        finally:
            os.close(campaign_fd)
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(run_root.exists())
        self.assertFalse(any(event["kind"] == "gpu" for event in events))
        self.assertFalse(any(event["kind"] == "runner" for event in events))

    def test_persistent_one_attempt_sentinel_blocks_any_second_attempt(self) -> None:
        shared = self.attempt_root / "persistent-stage11-attempt"
        first, first_root, _ = self._invoke(
            "one-attempt-first",
            attempt_sentinel_override=shared,
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertTrue((first_root / "TEST_COMPLETE").is_file())
        claim_before = (shared / "CLAIMED").read_bytes()
        marker_before = (shared / "TEST_COMPLETE").read_bytes()

        second, second_root, events = self._invoke(
            "one-attempt-second",
            attempt_sentinel_override=shared,
        )
        self.assertNotEqual(second.returncode, 0)
        self.assertFalse(second_root.exists())
        self.assertFalse(any(event["kind"] == "gpu" for event in events))
        self.assertFalse(any(event["kind"] == "runner" for event in events))
        self.assertEqual((shared / "CLAIMED").read_bytes(), claim_before)
        self.assertEqual(
            (shared / "TEST_COMPLETE").read_bytes(),
            marker_before,
        )

    def test_attempt_sentinel_parent_must_preexist(self) -> None:
        missing = self.temp / "missing-parent" / "attempt"
        completed, run_root, events = self._invoke(
            "missing-attempt-parent",
            attempt_sentinel_override=missing,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(run_root.exists())
        self.assertFalse(missing.exists())
        self.assertFalse(any(event["kind"] == "gpu" for event in events))
        self.assertFalse(any(event["kind"] == "runner" for event in events))

    def test_attempt_claim_mutation_fails_after_first_lane(self) -> None:
        completed, run_root, events = self._invoke(
            "attempt-claim-mutated",
            extra_env={"TASK26_FAKE_MUTATE_ATTEMPT": "1"},
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(
            len([event for event in events if event["kind"] == "runner"]),
            1,
        )
        self.assertEqual(
            len([event for event in events if event["kind"] == "audit"]),
            1,
        )
        self._assert_failed_test_artifact(run_root)

    def test_dirty_either_repository_fails_before_gpu_query(self) -> None:
        for name, repo in (
            ("dirty-harness", self.harness),
            ("dirty-sglang", self.sglang),
        ):
            dirty = repo / f"{name}.untracked"
            dirty.write_text("dirty\n")
            completed, run_root, events = self._invoke(name)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(run_root.exists())
            self.assertFalse(any(event["kind"] == "gpu" for event in events))
            self.assertFalse(any(event["kind"] == "runner" for event in events))
            dirty.unlink()

    def test_run_root_must_be_a_fresh_direct_child(self) -> None:
        existing = self._run_root("existing")
        existing.mkdir(parents=True)
        marker = existing / "preserve-me"
        marker.write_text("original\n")
        completed, observed_root, events = self._invoke("existing")
        self.assertEqual(observed_root, existing)
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(marker.read_text(), "original\n")
        self.assertEqual(
            sorted(path.name for path in existing.iterdir()), ["preserve-me"]
        )
        self.assertFalse(any(event["kind"] == "gpu" for event in events))

        for name, invalid_root in (
            ("outside-root", self.temp / "outside-root"),
            (
                "nested-root",
                self._run_root("parent") / "nested",
            ),
        ):
            with self.subTest(name=name):
                failed, _, invalid_events = self._invoke(
                    name,
                    run_root_override=invalid_root,
                )
                self.assertNotEqual(failed.returncode, 0)
                self.assertFalse(invalid_root.exists())
                self.assertFalse(
                    any(event["kind"] == "gpu" for event in invalid_events)
                )

    def test_low_disk_fails_before_gpu_query_and_run_root_creation(self) -> None:
        completed, run_root, events = self._invoke(
            "low-disk",
            extra_env={"TASK26_FAKE_DISK_LOW": "1"},
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(run_root.exists())
        self.assertFalse(any(event["kind"] == "gpu" for event in events))
        self.assertFalse(any(event["kind"] == "runner" for event in events))

    def test_rejects_multiple_physical_gpu_rows(self) -> None:
        completed, run_root, events = self._invoke(
            "multiple-physical",
            extra_env={"TASK26_FAKE_GPU_MULTIPLE": "1"},
        )
        self.assertNotEqual(completed.returncode, 0)
        self._assert_failed_test_artifact(run_root)
        self.assertFalse(any(event["kind"] == "runner" for event in events))

    def test_requires_b200_with_numeric_current_clocks(self) -> None:
        for name, overrides in (
            ("wrong-gpu-model", {"TASK26_FAKE_GPU_NAME": "NVIDIA H100"}),
            ("na-sm-clock", {"TASK26_FAKE_SM_CLOCK": "N/A"}),
            ("na-memory-clock", {"TASK26_FAKE_MEMORY_CLOCK": "N/A"}),
        ):
            with self.subTest(name=name):
                completed, run_root, events = self._invoke(
                    name,
                    extra_env=overrides,
                )
                self.assertNotEqual(completed.returncode, 0)
                self._assert_failed_test_artifact(run_root)
                self.assertFalse(any(event["kind"] == "runner" for event in events))

    def test_uuid_is_checked_after_each_lane_and_at_completion(self) -> None:
        after_lane, lane_root, lane_events = self._invoke(
            "uuid-after-lane",
            extra_env={"TASK26_FAKE_UUID_DRIFT_AT": "3"},
        )
        self.assertNotEqual(after_lane.returncode, 0)
        self.assertEqual(
            len([event for event in lane_events if event["kind"] == "runner"]),
            1,
        )
        self.assertEqual(
            len([event for event in lane_events if event["kind"] == "audit"]),
            1,
        )
        self._assert_failed_test_artifact(lane_root)

        at_completion, final_root, final_events = self._invoke(
            "uuid-at-completion",
            extra_env={"TASK26_FAKE_UUID_DRIFT_AT": "8"},
        )
        self.assertNotEqual(at_completion.returncode, 0)
        self.assertEqual(
            len([event for event in final_events if event["kind"] == "runner"]),
            len(EXPECTED_LANES),
        )
        self.assertEqual(
            len([event for event in final_events if event["kind"] == "audit"]),
            len(EXPECTED_LANES),
        )
        self._assert_failed_test_artifact(final_root)

    def test_runner_or_auditor_failure_stops_all_later_lanes(self) -> None:
        first_task, first_mode = EXPECTED_LANES[0]
        runner_failure, runner_root, runner_events = self._invoke(
            "runner-failure",
            extra_env={
                "TASK26_FAKE_RUNNER_FAIL": f"{first_task}|{first_mode}",
            },
        )
        self.assertNotEqual(runner_failure.returncode, 0)
        self.assertEqual(
            len([event for event in runner_events if event["kind"] == "runner"]),
            1,
        )
        self.assertEqual(
            len([event for event in runner_events if event["kind"] == "audit"]),
            0,
        )
        self._assert_failed_test_artifact(runner_root)

        first_stem = f"{first_task}__{first_mode}"
        audit_failure, audit_root, audit_events = self._invoke(
            "audit-failure",
            extra_env={"TASK26_FAKE_AUDITOR_FAIL": first_stem},
        )
        self.assertNotEqual(audit_failure.returncode, 0)
        self.assertEqual(
            len([event for event in audit_events if event["kind"] == "runner"]),
            1,
        )
        self.assertEqual(
            len([event for event in audit_events if event["kind"] == "audit"]),
            1,
        )
        self._assert_failed_test_artifact(audit_root)

    def test_non_win_and_malformed_audit_json_fail_closed(self) -> None:
        first_task, first_mode = EXPECTED_LANES[0]
        first_stem = f"{first_task}__{first_mode}"
        for name, env_key in (
            ("valid-non-win", "TASK26_FAKE_AUDITOR_NON_WIN"),
            ("malformed-audit", "TASK26_FAKE_AUDITOR_MALFORMED"),
        ):
            with self.subTest(name=name):
                completed, run_root, events = self._invoke(
                    name,
                    extra_env={env_key: first_stem},
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertEqual(
                    len([event for event in events if event["kind"] == "runner"]),
                    1,
                )
                self.assertEqual(
                    len([event for event in events if event["kind"] == "audit"]),
                    1,
                )
                self._assert_failed_test_artifact(run_root)
                self.assertNotIn("PASS task26", completed.stderr)

    def test_audit_gate_requires_exact_booleans(self) -> None:
        self.assertEqual(
            validate_audit_document({"valid": True, "performance_gate_passed": True}),
            [],
        )
        for document in (
            {"valid": 1, "performance_gate_passed": True},
            {"valid": True, "performance_gate_passed": 1},
            {"valid": True, "performance_gate_passed": False},
            {"valid": False, "performance_gate_passed": True},
            [],
        ):
            with self.subTest(document=document):
                self.assertTrue(validate_audit_document(document))


if __name__ == "__main__":
    unittest.main()
