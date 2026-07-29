"""CPU-only contracts for the dedicated MoE W2 hotspot goal."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from serving_native.workloads import WORKLOADS

HERE = Path(__file__).resolve().parent
CANDIDATE_PATH = HERE / "candidates" / "moe_w2_hotspot_bm16_auto.py"
PROVIDER_PATH = HERE / "providers" / "moe_w2_decode_bm16_auto.py"


def _load_candidate():
    spec = importlib.util.spec_from_file_location(
        "test_moe_w2_hotspot_candidate",
        CANDIDATE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_portfolio_is_exactly_four_hints_times_leaf_and_region():
    workloads = [
        item
        for item in WORKLOADS.values()
        if item.params.get("hotspot_goal")
        == "moe_w2_decode_bm16_auto_v1"
    ]
    assert len(workloads) == 8
    assert {
        (item.family, item.params["decode_m"], item.params["expected_m"])
        for item in workloads
    } == {
        (family, decode_m, expected_m)
        for family in ("moe_grouped_masked", "moe_compute_region")
        for decode_m, expected_m in ((16, 4), (16, 5), (32, 8), (32, 9))
    }
    assert {
        item.params["candidate_jit_identity"] for item in workloads
    } == {"infini_kernel_glm52_moe_w2_decode_bm16_auto"}
    assert all(
        item.execution_modes == ("eager", "cuda_graph")
        for item in workloads
    )


def test_signed_cpu_identity_is_exact_and_distinct():
    module = _load_candidate()
    workload = SimpleNamespace(
        name="moe_w2_hotspot_decode_em4",
        family="moe_grouped_masked",
        params={
            "hotspot_goal": module.HOTSPOT_GOAL,
            "decode_m": 16,
            "expected_m": 4,
        },
    )
    identity = module._verify_identity(SimpleNamespace(workload=workload))
    ready = identity["ready"]
    stock = identity["stock"]
    candidate = identity["candidate"]
    assert ready["cpu_only_identity_gate"] is True
    assert ready["cuda_initialized"] is False
    assert ready["material_identities"] == 2
    assert ready["production_default"] is False
    assert stock["source_sha256"] != candidate["source_sha256"]
    assert stock["entry_symbol"] != candidate["entry_symbol"]
    assert candidate["entry_symbol"].startswith(
        "infini_kernel_glm52_moe_w2_decode"
    )
    for hint in ("4", "5", "8", "9"):
        assert (
            stock["config_by_expected_m"][hint]["block_m"],
            stock["config_by_expected_m"][hint]["num_stages"],
        ) == (128, 8)
        assert (
            candidate["config_by_expected_m"][hint]["block_m"],
            candidate["config_by_expected_m"][hint]["num_stages"],
        ) == (16, 12)
    assert stock["resources"]["spill_load_bytes"] == 0
    assert candidate["resources"]["spill_load_bytes"] == 0
    assert stock["launch"]["cluster"] == candidate["launch"]["cluster"] == [
        2,
        1,
        1,
    ]


def test_provider_selected_callback_has_one_launch_and_no_import_or_fallback():
    tree = ast.parse(PROVIDER_PATH.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "moe_w2"
    )
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        for node in ast.walk(function)
    )
    launch_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_LAUNCH"
    ]
    assert len(launch_calls) == 1
    assert not any(
        isinstance(node, ast.Try) for node in ast.walk(function)
    )
    source = ast.get_source_segment(PROVIDER_PATH.read_text(), function)
    assert source is not None
    assert "glm52_moe_w2_decode_bm16_auto=True" in source
    assert "return None" in source


def test_candidate_dispatch_decline_is_fatal_before_any_stock_route():
    module = _load_candidate()

    class Provider:
        counts = {"attempted": 0, "completed": 0, "warmup": 4}

        @classmethod
        def call_counts(cls):
            return dict(cls.counts)

    class Decline:
        @staticmethod
        def try_dispatch_moe_masked(*_args):
            return False

    module._STATE.update(
        {
            "dispatch": Decline(),
            "provider_module": Provider,
            "selected_invocations": 0,
        }
    )
    try:
        with pytest.raises(RuntimeError, match="stock fallback is forbidden"):
            module._selected_w2(None, None, None, None, 4)
        assert Provider.call_counts()["attempted"] == 0
        assert module._STATE["selected_invocations"] == 0
    finally:
        module._STATE.clear()


def test_candidate_requires_exact_one_provider_launch():
    module = _load_candidate()

    class Provider:
        counts = {"attempted": 0, "completed": 0, "warmup": 4}

        @classmethod
        def call_counts(cls):
            return dict(cls.counts)

    class Select:
        @staticmethod
        def try_dispatch_moe_masked(*_args):
            Provider.counts["attempted"] += 1
            Provider.counts["completed"] += 1
            return True

    module._STATE.update(
        {
            "dispatch": Select(),
            "provider_module": Provider,
            "selected_invocations": 0,
        }
    )
    try:
        assert module._selected_w2(None, None, None, None, 9) is None
        assert Provider.call_counts()["attempted"] == 1
        assert Provider.call_counts()["completed"] == 1
        assert module._STATE["selected_invocations"] == 1
    finally:
        module._STATE.clear()


def test_selected_hot_path_defers_counter_dict_materialization():
    tree = ast.parse(CANDIDATE_PATH.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_selected_w2"
    )
    source = ast.get_source_segment(CANDIDATE_PATH.read_text(), function)
    assert source is not None
    assert "call_counts(" not in source
    assert source.count("try_dispatch_moe_masked(") == 1


def test_manifest_signatures_name_their_content_addresses():
    module = _load_candidate()
    for root in (module._STOCK_DIR, module._CANDIDATE_DIR):
        manifest = root / "manifest.json"
        signature = (root / "manifest.json.sha256").read_text().split()
        assert signature == [root.name, "manifest.json"]
        assert json.loads(manifest.read_text())["schema"] == (
            "glm52-moe-w2-decode-binary-bundle-v1"
        )


def test_untouched_tail_gate_is_strict_for_candidate_and_models_stock_bm128():
    runner_source = (HERE / "runner.py").read_text()
    audit_source = (HERE / "audit_result.py").read_text()
    for source in (runner_source, audit_source):
        assert "count == 0 or count % 128 == 0" in source
        assert '"candidate_required": True' in source
    assert "reference_tails is not reference_tails_expected" in runner_source
    assert "or not candidate_tails" in runner_source
