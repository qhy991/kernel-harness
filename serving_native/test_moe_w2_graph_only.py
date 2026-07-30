"""CPU-only contracts for the graph-only MoE W2 hotspot dispatch policy."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from serving_native import moe_w2_graph_only_gpu_contract as contract

HERE = Path(__file__).resolve().parent
SGLANG_ROOT = Path(
    "/home/qinhaiyan/glm52-hotspot-goal-runs/worktrees/moe-w2-decode/sglang"
)
DISPATCH = SGLANG_ROOT / "python/sglang/srt/layers/glm52_opt/dispatch.py"
REGISTRY = SGLANG_ROOT / "python/sglang/srt/layers/glm52_opt/registry.py"
CONFIG = SGLANG_ROOT / "python/sglang/srt/layers/glm52_opt/config.py"


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_w2_hotspot_spec_declares_graph_only():
    source = REGISTRY.read_text()
    tree = ast.parse(source)
    hotspot = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_HOTSPOT_DECODE"
    )
    graph_only_ops = set()
    for key, value in zip(hotspot.value.keys, hotspot.value.values):
        for keyword in value.keywords:
            if keyword.arg == "graph_only" and keyword.value.value is True:
                graph_only_ops.add(key.value)
    assert graph_only_ops == {"moe_down_proj"}


def test_graph_only_decline_precedes_abi_and_miss_accounting():
    """The eager decline must not launch, hash, or take the hit/miss lock."""
    source = DISPATCH.read_text()
    dispatch_fn = _function(DISPATCH, "try_dispatch_moe_masked")
    body = ast.get_source_segment(source, dispatch_fn)
    assert body is not None
    decline = body.index("_graph_only_declines(spec)")
    assert decline < body.index("_moe_hotspot_abi_matches(")
    assert decline < body.index("run_hotspot_moe_masked(")
    # Everything between the decline and its `return False` is the return.
    tail = body[decline : decline + 120]
    assert "return False" in tail
    assert "_record_miss" not in tail

    helper = ast.get_source_segment(source, _function(DISPATCH, "_graph_only_declines"))
    assert helper is not None
    assert "spec.graph_only" in helper
    assert "config.graph_only_enabled(spec.op)" in helper
    assert "not _is_cuda_graph_capturing()" in helper


def test_graph_only_env_defaults_on_and_is_registered():
    source = CONFIG.read_text()
    assert '"SGLANG_GLM52_W2_GRAPH_ONLY",' in source
    resolver = ast.get_source_segment(source, _function(CONFIG, "graph_only_enabled"))
    assert resolver is not None
    assert 'os.environ.get(key, "1")' in resolver
    assert '"moe_down_proj": "SGLANG_GLM52_W2_GRAPH_ONLY"' in source


def test_gpu_contract_audits_every_declared_property():
    """Each lettered property in the GPU contract has an audit failure path."""
    audit = ast.get_source_segment(
        contract.__file__ and Path(contract.__file__).read_text(),
        _function(Path(contract.__file__), "audit"),
    )
    assert audit is not None
    for prefix in ("A:", "B:", "C:", "D:", "E:"):
        assert prefix in audit
    assert "provider_attempts\"] != 0" in audit or "provider_attempts'] != 0" in audit


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            [
                {"order": "AB", "implementation": "reference", "latency_ms": 2.0},
                {"order": "AB", "implementation": "candidate", "latency_ms": 1.0},
                {"order": "BA", "implementation": "candidate", "latency_ms": 2.0},
                {"order": "BA", "implementation": "reference", "latency_ms": 8.0},
            ],
            (2.0, 4.0),
        ),
    ],
)
def test_identity_lane_estimators_use_both_order_strata(raw, expected):
    estimates = contract._estimates(raw)
    assert estimates["ab_median_speedup"] == expected[0]
    assert estimates["ba_median_speedup"] == expected[1]
    assert estimates["pair_count"] == 2
    assert estimates["order_balanced_speedup"] == pytest.approx(
        (expected[0] * expected[1]) ** 0.5
    )


def test_identity_lane_rejects_malformed_pairs():
    with pytest.raises(RuntimeError, match="pair order is malformed"):
        contract._estimates(
            [
                {"order": "AB", "implementation": "reference", "latency_ms": 1.0},
                {"order": "BA", "implementation": "candidate", "latency_ms": 1.0},
            ]
        )
    with pytest.raises(RuntimeError, match="pair is incomplete"):
        contract._estimates(
            [
                {"order": "AB", "implementation": "reference", "latency_ms": 1.0},
                {"order": "AB", "implementation": "reference", "latency_ms": 1.0},
            ]
        )


def test_identity_lane_floor_is_declared_below_one():
    # Both arms run the same stock kernel, so the lane is an identity check and
    # must never be presented as a candidate speedup.
    assert 0.9 < contract.IDENTITY_LANE_FLOOR < 1.0
