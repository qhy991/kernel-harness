#!/usr/bin/env python3
"""Package a verified drop-in candidate as a reversible sglang source patch.

`integrate.py` proves (by monkeypatch, in a live forward) that a candidate `run()`
is a valid in-place replacement for a specific sglang dispatch symbol. `migrate.py`
turns that proven swap into the DEPLOYABLE artifact: a byte-exact-reversible edit to
the sglang module that owns the symbol, rebinding it to route through the candidate —
the permanent form of integrate.py's temporary swap.

Flow:
  1. Refuse unless `integrate.py <task>` is GREEN (only a machine-verified drop-in is
     migratable). This is the gate.
  2. Append a MARKER-delimited block to the symbol's home module that loads the
     candidate solution.py and rebinds the exact dispatch symbol through an adapter
     (the same signature reconciliation integrate.py verified).
  3. Verify: the block imports cleanly and the symbol now routes to the candidate;
     then write both the forward `.patch` and a `revert.patch` (strip the block) into
     the task's results/. Revert restores the file byte-exact.
  4. Print the remaining deployment gates this patch does NOT clear (loud, not silent).

Families whose dispatch is not a rebindable module-level symbol (bmm = global
torch.bmm; lm-head = an in-body torch.matmul; bf16-linear = a method on a class) are
reported as having no clean source site — mirroring integrate.py's honesty rather than
faking a patch.

Usage:  python bin/migrate.py <task_dir> [--solution solution.py] [--apply]
        (without --apply it writes the patch files but does not modify sglang source)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SGLANG_DIR, VENV, CUDA_HOME, resolve_sglang_dir

BIN = Path(__file__).resolve().parent

# family -> (module dotted path, symbol, adapter source). The adapter reconciles
# sglang's call signature with the candidate run(), identical to the integrate.py
# recipe that was verified. `_cand` is the loaded candidate run.
MIGRATE_MAP = {
    "fp8-linear-gemm": (
        "sglang.srt.layers.quantization.fp8_utils", "w8a8_block_fp8_matmul_deepgemm",
        "def _sgl(A, B, As, Bs, block_size, output_dtype):\n"
        "    return _cand(A, As, B, Bs)\n",
    ),
    "act-fp8-quant": (
        "sglang.srt.layers.quantization.fp8_utils", "sglang_per_token_group_quant_fp8",
        "def _sgl(x, group_size, *a, **k):\n"
        "    return _cand(x)\n",
    ),
    "rmsnorm": (
        "sglang.srt.layers.layernorm", "rmsnorm",
        "def _sgl(x, weight, eps, *a, **k):\n"
        "    return _cand(x, weight)\n",
    ),
    "fused-add-rmsnorm": (
        "sglang.srt.layers.layernorm", "fused_add_rmsnorm",
        "def _sgl(*a, **k):\n    return _cand(*a, **k)\n",   # interface-exact
    ),
    "gemma-rmsnorm": (
        "sglang.srt.layers.layernorm", "gemma_rmsnorm",
        "def _sgl(*a, **k):\n    return _cand(*a, **k)\n",   # interface-exact
    ),
    "gemma-fused-add-rmsnorm": (
        "sglang.srt.layers.layernorm", "gemma_fused_add_rmsnorm",
        "def _sgl(*a, **k):\n    return _cand(*a, **k)\n",   # interface-exact
    ),
    "swiglu": (
        "sglang.srt.layers.activation", "silu_and_mul",
        "def _sgl(inp, out):\n    out.copy_(_cand(inp))\n",
    ),
    "moe-combine": (
        "sgl_kernel", "moe_sum",
        "def _sgl(*a, **k):\n    return _cand(*a, **k)\n",   # interface-exact (inp, out)
    ),
    "grouped-moe": (
        "sglang.srt.layers.deep_gemm_wrapper.entrypoint", "grouped_gemm_nt_f8f8bf16_masked",
        "def _sgl(lhs, rhs, out, masked_m, expected_m, *a, **k):\n"
        "    return _cand(lhs[0], lhs[1], rhs[0], rhs[1], out, masked_m, expected_m)\n",
    ),
    "grouped-moe-contiguous": (
        "sglang.srt.layers.deep_gemm_wrapper.entrypoint", "grouped_gemm_nt_f8f8bf16_contig",
        "def _sgl(lhs, rhs, out, m_indices, *a, **k):\n"
        "    return _cand(lhs[0], lhs[1], rhs[0], rhs[1], out, m_indices)\n",
    ),
    "router-gemm": (
        "sgl_kernel", "dsv3_router_gemm",
        "def _sgl(a, b, *args, **k):\n    return _cand(a, b)\n",
    ),
    "moe-gate": (
        "sgl_kernel", "kimi_k2_moe_fused_gate",
        "def _sgl(inp, bias, *a, **k):\n"
        "    idx = _cand(inp, bias)\n"
        "    return None, idx\n",   # (weights, indices) tuple; index oracle
    ),
    # DSA (sglang-m3) interface-exact JIT ops — rebind the jit_kernel module symbol.
    "dsa-qknorm-rope": (
        "sglang.jit_kernel.minimax_qknorm_rope", "minimax_qknorm_rope",
        "def _sgl(*a, **k):\n    return _cand(*a, **k)\n",
    ),
    "dsa-decode-topk": (
        "sglang.jit_kernel.minimax_decode_topk", "minimax_decode_topk",
        "def _sgl(*a, **k):\n    return _cand(*a, **k)\n",
    ),
    "dsa-store-kv-index": (
        "sglang.jit_kernel.minimax_store_kv_index", "store_kv_index",
        "def _sgl(*a, **k):\n    return _cand(*a, **k)\n",
    ),
}

# families with no rebindable module-level symbol to patch in source
NO_SOURCE_SITE = {
    "bmm": "sglang dispatches global torch.bmm at the MLA absorb call sites; no "
           "module-level symbol to rebind (patch the call sites in forward_mla.py by "
           "hand, or add a scoped wrapper first).",
    "lm-head": "logits use an in-body torch.matmul inside LogitsProcessor._compute_lm_head; "
               "no module-level dispatch symbol — edit the method directly.",
    "bf16-linear": "dispatch is UnquantizedLinearMethod.apply (a method); rebind at the "
                   "method level in unquant.py, not a module symbol.",
    "embedding": "dispatch is UnquantizedEmbeddingMethod.embedding (a method); rebind at "
                 "the method level, not a module symbol.",
    "rope": "rope is apply_rope_with_cos_sin_cache_inplace in rotary_embedding.base; "
            "module-level but in-place with a keyword-only tail — patch by hand.",
}

MARK_BEGIN = "# === KERSOR-MIGRATE BEGIN: {task} ({family}) ==="
MARK_END = "# === KERSOR-MIGRATE END: {task} ==="

# Per-(family, model) overrides mirroring integrate.py's MODEL_RECIPES: some families
# dispatch a different sglang kernel per model. MiniMax-M3's gate is topk_sigmoid (a DPS
# kernel), not kimi_k2_moe_fused_gate; its router is a fp32 Linear method (no module
# symbol), so it's a NO_SOURCE_SITE for migration.
MODEL_MIGRATE = {
    ("moe-gate", "minimax_m3"): (
        "sgl_kernel", "topk_sigmoid",
        "def _sgl(topk_weights, topk_ids, gating_output, *a, **k):\n"
        "    topk_ids.copy_(_cand(gating_output, k.get('correction_bias')))\n",
    ),
}
MODEL_NO_SOURCE = {
    ("router-gemm", "minimax_m3"):
        "MiniMax-M3 router is a fp32 ReplicatedLinear (UnquantizedLinearMethod.apply, a "
        "method), not dsv3_router_gemm — rebind at the method level, not a module symbol.",
}


def _resolve_migrate(family, model):
    """(entry, no_source_reason). entry is (module, symbol, adapter) or None."""
    if (family, model) in MODEL_NO_SOURCE:
        return None, MODEL_NO_SOURCE[(family, model)]
    if (family, model) in MODEL_MIGRATE:
        return MODEL_MIGRATE[(family, model)], None
    if family in NO_SOURCE_SITE:
        return None, NO_SOURCE_SITE[family]
    if family in MIGRATE_MAP:
        return MIGRATE_MAP[family], None
    return None, None


def _module_file(module: str, sglang_dir: str) -> Path:
    """Resolve a dotted module to its source .py (in the given sglang build)."""
    out = subprocess.run(
        [str(VENV / "bin" / "python"), "-c",
         f"import importlib; m=importlib.import_module('{module}'); print(m.__file__)"],
        capture_output=True, text=True,
        env={"PYTHONPATH": f"{sglang_dir}/python", "PATH": f"{VENV}/bin"},
    )
    if out.returncode != 0:
        raise RuntimeError(f"cannot resolve module {module}: {out.stderr.strip()}")
    return Path(out.stdout.strip())


def _evaluate_win(task_dir: Path, solution: str, repeat: int) -> dict:
    """Run evaluate.py and return its VERDICT_JSON; used to gate migration on a real
    performance WIN (correct on every shape AND faster on every shape), not just a
    drop-in that might be correct-but-slower."""
    out = subprocess.run(
        [str(VENV / "bin" / "python"), str(BIN / "evaluate.py"),
         str(task_dir), "--solution", solution, "--repeat", str(repeat)],
        capture_output=True, text=True,
        env={"PYTHONPATH": f"{SGLANG_DIR}/python", "PATH": f"{VENV}/bin",
             "CUDA_HOME": str(CUDA_HOME)},
    )
    txt = out.stdout
    if "VERDICT_JSON_BEGIN" not in txt:
        raise RuntimeError(f"evaluate.py produced no verdict:\n{txt[-800:]}\n{out.stderr[-400:]}")
    return json.loads(txt.split("VERDICT_JSON_BEGIN")[1].split("VERDICT_JSON_END")[0])


def _integrate_green(task_dir: Path, solution: str) -> dict:
    """Run integrate.py and return its verdict JSON; raise if not drop-in verified."""
    out = subprocess.run(
        [str(VENV / "bin" / "python"), str(BIN / "integrate.py"),
         str(task_dir), "--solution", solution],
        capture_output=True, text=True,
        env={"PYTHONPATH": f"{SGLANG_DIR}/python", "PATH": f"{VENV}/bin",
             "CUDA_HOME": str(CUDA_HOME)},
    )
    txt = out.stdout
    if "INTEGRATION_JSON_BEGIN" not in txt:
        raise RuntimeError(f"integrate.py produced no verdict:\n{txt[-800:]}\n{out.stderr[-400:]}")
    verdict = json.loads(txt.split("INTEGRATION_JSON_BEGIN")[1].split("INTEGRATION_JSON_END")[0])
    return verdict


def _block(task, family, module, symbol, adapter, sol_path):
    """The marker-delimited source block that rebinds the dispatch symbol."""
    begin = MARK_BEGIN.format(task=task, family=family)
    end = MARK_END.format(task=task)
    return (
        f"\n\n{begin}\n"
        f"# Verified drop-in from kernel-harness (integrate.py green). Reversible: strip\n"
        f"# this block (bin/migrate.py writes results/revert.patch) to restore the kernel.\n"
        f"import importlib.util as _ilu\n"
        f"_spec = _ilu.spec_from_file_location('_kernel_harness_{task}', r'{sol_path}')\n"
        f"_mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)\n"
        f"_cand = _mod.run\n"
        f"{adapter}"
        f"{symbol} = _sgl\n"
        f"{end}\n"
    )


REMAINING_GATES = [
    "full shape-space coverage (this proves the sweep only, not sglang's whole dispatch space)",
    "real-model end-to-end accuracy (loose per-op tolerance is not model quality)",
    "sglang's own unit tests (emit with bin/emit_sglang.py, run under sgl-kernel/tests/)",
    "CUDA-graph capture safety (candidate must be capturable at the dispatch site)",
    "AOT/JIT build + non-Blackwell fallback (this targets B200 only)",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir", type=Path)
    ap.add_argument("--solution", default="solution.py")
    ap.add_argument("--apply", action="store_true",
                    help="actually append the block to the sglang source (default: patch files only)")
    ap.add_argument("--repeat", type=int, default=3,
                    help="evaluate.py --repeat for the WIN gate (default 3, noise-robust)")
    ap.add_argument("--skip-win-gate", action="store_true",
                    help="skip the evaluate WIN gate (integrate-green only; NOT recommended)")
    args = ap.parse_args()

    task_dir = args.task_dir.resolve()
    meta = json.loads((task_dir / "task.json").read_text())
    family = meta.get("family")
    model = meta.get("model") or task_dir.parent.name   # dir is the reliable model signal
    task = meta.get("name", task_dir.name)
    sol_path = task_dir / args.solution
    sglang_dir = resolve_sglang_dir(meta.get("sglang_dir"))

    entry, no_source = _resolve_migrate(family, model)
    if entry is None and no_source is not None:
        print(f"NO CLEAN SOURCE SITE for family '{family}' (model={model}):\n  {no_source}")
        sys.exit(2)
    if entry is None:
        print(f"no migrate recipe for family '{family}' (model={model}).")
        sys.exit(2)

    print(f"== gate 1/2: evaluate.py must be a WIN for {task} ==")
    if args.skip_win_gate:
        print("  SKIPPED (--skip-win-gate): migrating a not-necessarily-faster drop-in.")
    else:
        wv = _evaluate_win(task_dir, args.solution, args.repeat)
        if not (wv.get("correct") and wv.get("win")):
            print(f"REFUSED: not a performance WIN (correct={wv.get('correct')}, "
                  f"win={wv.get('win')}, min_speedup_conservative="
                  f"{wv.get('min_speedup_conservative')}). Only a candidate that is correct "
                  f"AND faster on every shape is worth migrating.")
            sys.exit(1)
        print(f"  evaluate.py WIN (geomean={wv.get('geomean_speedup')}, "
              f"min_conservative={wv.get('min_speedup_conservative')})")

    print(f"== gate 2/2: integrate.py must be green for {task} ==")
    verdict = _integrate_green(task_dir, args.solution)
    if not verdict.get("drop_in_ok"):
        print(f"REFUSED: integrate.py is not green (verdict={verdict}). "
              f"A candidate must be a verified drop-in before migration.")
        sys.exit(1)
    print(f"  integrate.py GREEN (invoked={verdict.get('invoked')}, "
          f"match={verdict.get('match_ratio')}, restored={verdict.get('restored')})")

    module, symbol, adapter = entry
    src = _module_file(module, sglang_dir)
    original = src.read_text()
    block = _block(task, family, module, symbol, adapter, sol_path)
    patched = original + block

    results = task_dir / "results"
    results.mkdir(exist_ok=True)
    fwd = results / "migrate.patch"
    rev = results / "revert.patch"

    def _diff(a_text, b_text, path):
        import difflib
        # repo-relative path so `git apply` works from the sglang root
        try:
            rel = str(Path(path).resolve().relative_to(Path(sglang_dir).resolve()))
        except ValueError:
            rel = str(path)
        return "".join(difflib.unified_diff(
            a_text.splitlines(keepends=True), b_text.splitlines(keepends=True),
            fromfile=f"a/{rel}", tofile=f"b/{rel}"))

    fwd.write_text(_diff(original, patched, src))
    rev.write_text(_diff(patched, original, src))

    # Verify round-trip: apply -> symbol routes to candidate -> revert byte-exact.
    if args.apply:
        src.write_text(patched)
        check = subprocess.run(
            [str(VENV / "bin" / "python"), "-c",
             f"import importlib; m=importlib.import_module('{module}'); "
             f"print(getattr(m,'{symbol}').__name__)"],
            capture_output=True, text=True,
            env={"PYTHONPATH": f"{sglang_dir}/python", "PATH": f"{VENV}/bin"},
        )
        routed = "_sgl" in check.stdout
        src.write_text(original)   # revert
        restored_ok = src.read_text() == original
        print(f"  applied+verified: symbol routes to candidate={routed}; "
              f"reverted byte-exact={restored_ok}")
        if not (routed and restored_ok):
            print("MIGRATE FAILED (apply/verify/revert round-trip)")
            sys.exit(1)

    print(f"\nwrote {fwd}\nwrote {rev}")
    print(f"\nApply with:   (cd {sglang_dir} && git apply {fwd})")
    print(f"Revert with:  (cd {sglang_dir} && git apply {rev})")
    print("\nMIGRATE READY — remaining deployment gates this patch does NOT clear:")
    for g in REMAINING_GATES:
        print(f"  [ ] {g}")
    sys.exit(0)


if __name__ == "__main__":
    main()
