#!/usr/bin/env python3
"""Warm-start a kernel-optimization session: everything worth knowing before you
touch the kernel, in one call. Stdlib-only, fast, GPU-free.

    python3 testbench/bin/brief.py o_proj_decode

Prints, in order:
  1. individually runnable production experiment recipes;
  2. the best audited prior measured result for this task (from runs/), so you start from the
     current frontier instead of rediscovering it;
  3. the knowledge union — internal recipes (what we tried + why) + the library-kernel
     ledger + KernelWiki prior-art (delegates to `knowledge.py brief`);
  4. the task's roofline / bound (best-effort `run.sh --describe`; skipped if the heavy
     import stack or venv isn't ready — never blocks the fast sections above).

Retrieval was the discretionary step sessions skipped; this makes it one command.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

BIN = Path(__file__).resolve().parent
TASKS = BIN.parent / "tasks" / "glm52"
RUNS = BIN.parent.parent / "runs" / "glm52"
RECIPE_INDEX = BIN.parent / "knowledge" / "recipes" / "index.json"


def _load_auditor():
    path = BIN / "audit_result.py"
    spec = importlib.util.spec_from_file_location("_brief_audit_result", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_rank(r: dict) -> tuple:
    verdict = r.get("verdict") or {}
    agg = r.get("aggregate") or {}
    if isinstance(verdict, dict):
        perf_ok = bool(verdict.get("performance_ok"))
        correct = bool(verdict.get("correct"))
        exit_code = verdict.get("exit_code")
    else:
        perf_ok = False
        correct = str(verdict).upper() == "CORRECT"
        exit_code = None
    speed = (agg.get("geomean_speedup") or
             agg.get("min_speedup_conservative") or
             agg.get("speedup") or 0)
    reward = agg.get("best_reward") or 0
    # Correctness outranks speed: an incorrect run with a bogus high speedup must
    # never be the warm-start recommendation.
    return (perf_ok, correct, exit_code == 0, speed, reward)


def _schema_rank(value) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in str(value).split("."))
    except ValueError:
        return (0,)


def _best_prior_run(task: str) -> None:
    d = RUNS / task
    results = sorted(d.glob("*/result.json")) if d.is_dir() else []
    print(f"== best prior run ({len(results)} recorded) — runs/glm52/{task} ==")
    if not results:
        print("  (no prior runs)")
        return
    try:
        auditor = _load_auditor()
    except Exception as exc:
        print(f"  (cannot audit prior runs: {exc}; no frontier selected)")
        return
    best = None
    invalid = 0
    for f in results:
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        audit = auditor.audit(f.resolve())
        if audit.get("errors"):
            invalid += 1
            continue
        # Official evidence outranks provisional evidence before speed. This
        # prevents an old-schema/noisy result from becoming the warm-start frontier.
        if audit.get("official"):
            key = (2,) + _run_rank(r)
        else:
            # There is no defensible provisional "frontier": show the most
            # recent correct context from the newest schema, not the largest
            # stale headline speedup. A schema-1.0 strawman win must not become
            # the next agent's starting candidate.
            timestamp = ((r.get("run") or {}).get("finished_utc") or f.parent.name)
            key = (
                1,
                bool((r.get("verdict") or {}).get("correct"))
                if isinstance(r.get("verdict"), dict)
                else str(r.get("verdict")).upper() == "CORRECT",
                _schema_rank(r.get("schema_version")),
                str(timestamp),
                -len(audit.get("warnings") or []),
            )
        if best is None or key > best[0]:
            best = (key, r, f, audit)
    if best is None:
        print("  (prior runs unreadable)")
        return
    _, r, f, audit = best
    agg, verdict = r.get("aggregate") or {}, r.get("verdict")
    if isinstance(verdict, dict):
        status = (f"correct={bool(verdict.get('correct'))} "
                  f"performance_ok={bool(verdict.get('performance_ok'))} "
                  f"exit={verdict.get('exit_code')}")
    else:
        status = str(verdict)
    audit_label = "OFFICIAL" if audit.get("official") else "PROVISIONAL"
    if not audit.get("official"):
        print("  no official frontier; showing most recent admissible provisional context")
    print(f"  audit={audit_label}  verdict={status}  {json.dumps(agg)[:300]}")
    print(f"  from {f.relative_to(BIN.parent.parent)}")
    for warning in (audit.get("warnings") or [])[:2]:
        print(f"  caveat: {warning}")
    if invalid:
        print(f"  excluded {invalid} invalid prior result(s) from ranking")


def _atomic_recipes(task: str) -> None:
    print("== atomic experiment recipes (select and finish one at a time) ==")
    try:
        catalog = json.loads(RECIPE_INDEX.read_text())
    except Exception as exc:
        print(f"  (recipe catalog unavailable: {exc})")
        return
    entries = catalog.get("recipes") or []
    by_id = {entry.get("id"): entry for entry in entries}
    selected_ids = [
        "00-pin-production-contract",
        "01-capture-kv-context-profile",
        "02-measure-target-headroom",
        "03-adversarial-correctness",
        "04-paired-leaf-timing",
    ]
    selected_ids.append(
        "06-incremental-prefill-kv-matrix"
        if "prefill" in task else "05-decode-kv-context-matrix"
    )
    lower = task.lower()
    if "moe" in lower or "routed" in lower:
        selected_ids.append("23-stop-roofline-bound-moe-retuning")
    if "index" in lower:
        selected_ids.append("21-tma-split-k-small-m")
    if "q_b" in lower:
        selected_ids.append("20-fuse-scale-packing")
    if "o_proj" in lower:
        selected_ids.append("22-bounded-compiled-dimension")

    relative = RECIPE_INDEX.relative_to(BIN.parent.parent)
    print(f"  {catalog.get('title')} — {len(entries)} recipes  [{relative}]")
    for recipe_id in selected_ids:
        entry = by_id.get(recipe_id)
        if entry:
            print(f"  {recipe_id}: python3 testbench/bin/recipe.py show {recipe_id}")
    print("  catalog: python3 testbench/bin/recipe.py list")


def _knowledge_brief(task: str, limit: int, no_external: bool) -> None:
    cmd = [sys.executable, str(BIN / "knowledge.py"), "brief", "--task", task,
           "--limit", str(limit)]
    if no_external:
        cmd.append("--no-external")
    print()
    try:
        subprocess.run(cmd, check=False)
    except Exception as e:
        print(f"  (knowledge brief failed: {e})")


def _roofline(task: str, timeout: int) -> None:
    run = TASKS / task / "run.sh"
    print(f"\n== roofline / bound (best-effort `run.sh --describe`) ==")
    if not run.is_file():
        print("  (no run.sh for this task)")
        return
    try:
        r = subprocess.run([str(run), "--describe"], capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  (--describe exceeded {timeout}s; run it directly: {run} --describe)")
        return
    except Exception as e:
        print(f"  (--describe unavailable: {e}; venv may not be set up)")
        return
    out = (r.stdout or r.stderr).strip()
    # Keep it compact: the per-shape roofline table and the `ref baseline` rows.
    kept = [ln for ln in out.splitlines()
            if any(t in ln for t in ("bound", "reward", "AI", "MFU", "BW", "baseline",
                                     "ridge", "roof"))]
    print("\n".join("  " + ln for ln in (kept or out.splitlines())[:24]) or "  (empty)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task", help="task dir name, e.g. o_proj_decode")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--no-external", action="store_true", help="skip KernelWiki bridge")
    ap.add_argument("--describe-timeout", type=int, default=90)
    ap.add_argument("--no-describe", action="store_true", help="skip the heavy roofline")
    args = ap.parse_args()

    try:  # keep our prints ordered with subprocess output when piped
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    task = args.task.split("/")[-1]
    if not (TASKS / task).is_dir():
        avail = ", ".join(sorted(p.name for p in TASKS.iterdir() if p.is_dir())[:6])
        print(f"warning: no task dir testbench/tasks/glm52/{task} (e.g. {avail}, ...)",
              file=sys.stderr)

    print(f"# warm-start brief — {task}\n")
    _atomic_recipes(task)
    print()
    _best_prior_run(task)
    _knowledge_brief(task, args.limit, args.no_external)
    if not args.no_describe:
        _roofline(task, args.describe_timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
