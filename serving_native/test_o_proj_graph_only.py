"""CPU-only contracts for the graph-only decode o_proj fixed-N/K dispatch."""

from __future__ import annotations

import ast
from pathlib import Path

from serving_native import o_proj_graph_only_gpu_contract as contract

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


def test_o_proj_e2e_spec_declares_graph_only():
    source = REGISTRY.read_text()
    tree = ast.parse(source)
    e2e = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_E2E_DECODE"
    )
    graph_only_ops = set()
    for key, value in zip(e2e.value.keys, e2e.value.values):
        for keyword in value.keywords:
            if keyword.arg == "graph_only" and keyword.value.value is True:
                graph_only_ops.add(key.value)
    assert graph_only_ops == {"o_proj"}


def test_fp8_gemm_graph_only_decline_precedes_abi_and_miss_accounting():
    """The eager decline must not launch, hash, or take the hit/miss lock."""
    source = DISPATCH.read_text()
    dispatch_fn = _function(DISPATCH, "try_dispatch_fp8_gemm")
    body = ast.get_source_segment(source, dispatch_fn)
    assert body is not None
    decline = body.index("_graph_only_declines(spec)")
    assert decline < body.index("_fixed_nk_abi_matches(")
    assert decline < body.index("run_fp8_gemm(")
    tail = body[decline : decline + 80]
    assert "return None" in tail
    assert "_record_miss" not in tail


def test_o_proj_graph_only_env_registered_and_defaults_on():
    source = CONFIG.read_text()
    assert '"SGLANG_GLM52_O_PROJ_GRAPH_ONLY",' in source
    assert '"o_proj": "SGLANG_GLM52_O_PROJ_GRAPH_ONLY"' in source
    resolver = ast.get_source_segment(source, _function(CONFIG, "graph_only_enabled"))
    assert resolver is not None
    assert 'os.environ.get(key, "1")' in resolver


def test_gpu_contract_audits_every_declared_lane():
    audit_src = ast.get_source_segment(
        Path(contract.__file__).read_text(), _function(Path(contract.__file__), "audit")
    )
    assert audit_src is not None
    for tag in ("A:", "B:", "C:", "D:", "E:", "F:", "G:", "H:"):
        assert tag in audit_src
    # The performance gate must be the plan's 1.03 graph floor.
    assert "GRAPH_WIN_FLOOR" in audit_src


def test_contract_floors_are_the_plan_values():
    assert contract.GRAPH_WIN_FLOOR == 1.03
    assert 0.9 < contract.IDENTITY_LANE_FLOOR < 1.0
    assert contract.N == 6144 and contract.K == 16384
    assert contract.M_VALUES == (16, 32)
    assert contract.CANDIDATE_NVTX_SYMBOL == "infini_kernel_glm52_attn_o_decode_nk"


def test_audit_flags_a_sub_floor_graph_series():
    """A synthetic sub-1.03 graph-leaf series must fail the audit."""

    def series(ratio):
        raw = []
        for i in range(4):
            order = "AB" if i % 2 == 0 else "BA"
            names = ("reference", "candidate") if order == "AB" else ("candidate", "reference")
            lat = {"reference": ratio, "candidate": 1.0}
            for name in names:
                raw.append({"order": order, "implementation": name, "latency_ms": lat[name]})
        return {"performance_estimates": contract._estimates(raw)}

    good = [series(1.10) for _ in range(3)]
    bad = [series(1.10), series(1.01), series(1.10)]  # one series below floor
    ok_m = {
        "x_scale_dtype": "torch.int32",
        "A_eager_graph_only_declines": {"returned_none": True, "hit_delta": 0},
        "B_eager_diagnostic_selects": {"returned_tensor": True, "hit_delta": 1},
        "C_candidate_leaf_graph": {"kernel_node_count": 1, "forbidden_nodes": []},
        "C_candidate_differs_from_stock": True,
        "D_capture_identical_on_off": {"a": True, "b": True},
        "E_eager_max_abs_err": 0.0,
        "E_graph_max_abs_err": 0.0,
        "G_region_max_abs_err": 0.0,
        "F_graph_leaf_series": good,
        "G_graph_region_series": good,
        "H_eager_identity_series": good,
    }
    base = {
        "gpu_uuid": "GPU-test",
        "deepgemm_scale_ue8m0": True,
        "by_m": {"16": ok_m},
    }
    assert contract.audit(base) == []
    import copy

    broken = copy.deepcopy(base)
    broken["by_m"]["16"]["G_graph_region_series"] = bad
    fails = contract.audit(broken)
    assert any("G: graph-region series 1" in f for f in fails), fails
