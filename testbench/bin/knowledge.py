#!/usr/bin/env python3
"""Optimization-recipe knowledge base — add / lint / query (stdlib only, no GPU).

Agents write one entry per completed optimization session so later sessions (any
model family, any GPU) start from prior diagnoses, wins, and dead ends. The schema
and honesty rules are the contract in testbench/knowledge/README.md; this tool
enforces them.

    python3 testbench/bin/knowledge.py add <entry.json> [--check]   # validate (+ install)
    python3 testbench/bin/knowledge.py lint                          # validate all entries
    python3 testbench/bin/knowledge.py query [filters] [--json]      # find relevant recipes

Entries are append-only: `add` refuses to overwrite and nothing here mutates an
existing entry. Exit 0 = ok / matches found, 1 = validation problems, 2 = usage error.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

BOTTLENECKS = {"memory-bandwidth", "compute", "launch-overhead", "kernel-count",
               "quantization-overhead", "occupancy", "synchronization",
               "none-identified", "other"}
OUTCOMES = {"win", "partial", "slower", "incorrect", "error", "abandoned"}
STATUSES = {"win", "no-win", "failed"}
INTEGRATE = {"pass", "fail", "not-run", "no-recipe"}
TOP_KEYS = {"schema_version", "id", "date", "model", "task", "op", "family", "phase",
            "shapes", "hardware", "stack", "baseline_kernel", "bottleneck",
            "approaches", "result", "lesson", "transfers_to", "caveats", "agent"}
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _s(v) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate(entry, expected_id=None) -> list[str]:
    """Return a list of problems (empty = valid)."""
    if not isinstance(entry, dict):
        return ["entry is not a JSON object"]
    p = []
    unknown = set(entry) - TOP_KEYS
    if unknown:
        p.append(f"unknown top-level keys {sorted(unknown)}")
    if entry.get("schema_version") != 1:
        p.append("schema_version must be 1")

    eid = entry.get("id")
    if not (_s(eid) and _ID_RE.match(eid)):
        p.append("id must be a lowercase [a-z0-9._-] slug")
    elif expected_id is not None and eid != expected_id:
        p.append(f"id {eid!r} != filename stem {expected_id!r}")
    try:
        datetime.date.fromisoformat(entry.get("date", ""))
    except (TypeError, ValueError):
        p.append("date must be YYYY-MM-DD")

    for k in ("model", "op", "family", "baseline_kernel", "lesson"):
        if not _s(entry.get(k)):
            p.append(f"{k} must be a non-empty string")
    if entry.get("phase") not in ("prefill", "decode"):
        p.append("phase must be prefill|decode")
    task = entry.get("task")
    if not (_s(task) and "/" in task):
        p.append("task must be '<model>/<task_dir>'")
    elif _s(entry.get("model")) and task.split("/", 1)[0] != entry["model"]:
        p.append("task must start with '<model>/'")
    if "agent" in entry and not _s(entry["agent"]):
        p.append("agent, when present, must be a non-empty string")

    if not (isinstance(entry.get("shapes"), dict) and entry["shapes"]):
        p.append("shapes must be a non-empty object")
    hw = entry.get("hardware")
    if not (isinstance(hw, dict) and _s(hw.get("gpu")) and _s(hw.get("sm"))):
        p.append("hardware must have non-empty gpu and sm")
    stack = entry.get("stack")
    if not (isinstance(stack, dict) and _s(stack.get("sglang_commit"))):
        p.append("stack must have non-empty sglang_commit")

    bn = entry.get("bottleneck")
    if not isinstance(bn, dict) or bn.get("kind") not in BOTTLENECKS:
        p.append(f"bottleneck.kind must be one of {sorted(BOTTLENECKS)}")
    if not (isinstance(bn, dict) and _s(bn.get("evidence"))):
        p.append("bottleneck.evidence must be a non-empty string (the measurement)")

    approaches = entry.get("approaches")
    if not (isinstance(approaches, list) and approaches):
        p.append("approaches must be a non-empty array")
        approaches = []
    for i, a in enumerate(approaches, 1):
        if not isinstance(a, dict):
            p.append(f"approaches[{i}] is not an object")
            continue
        for k in ("technique", "summary", "why"):
            if not _s(a.get(k)):
                p.append(f"approaches[{i}].{k} must be a non-empty string")
        out = a.get("outcome")
        if out not in OUTCOMES:
            p.append(f"approaches[{i}].outcome must be one of {sorted(OUTCOMES)}")
        gs = a.get("geomean_speedup")
        if out in ("win", "partial"):
            if not _num(gs):
                p.append(f"approaches[{i}].geomean_speedup required (a number) for outcome {out!r}")
        elif gs is not None and not _num(gs):
            p.append(f"approaches[{i}].geomean_speedup must be a number or null")

    res = entry.get("result")
    if not isinstance(res, dict):
        p.append("result must be an object")
        res = {}
    if res.get("status") not in STATUSES:
        p.append(f"result.status must be one of {sorted(STATUSES)}")
    if res.get("integrate") not in INTEGRATE:
        p.append(f"result.integrate must be one of {sorted(INTEGRATE)}")
    for k in ("geomean_speedup", "min_speedup_conservative"):
        if res.get(k) is not None and not _num(res.get(k)):
            p.append(f"result.{k} must be a number or null")
    if res.get("repeat") is not None and not isinstance(res.get("repeat"), int):
        p.append("result.repeat must be an integer or null")
    if res.get("status") == "win":
        msc = res.get("min_speedup_conservative")
        if not (_num(msc) and msc > 1.0):
            p.append("result.status 'win' requires min_speedup_conservative > 1.0 "
                     "(from the final VERDICT_JSON)")
        if not any(isinstance(a, dict) and a.get("outcome") == "win" for a in approaches):
            p.append("result.status 'win' requires at least one approach with outcome 'win'")

    for k in ("transfers_to", "caveats"):
        v = entry.get(k)
        if not (isinstance(v, list) and all(_s(x) for x in v)):
            p.append(f"{k} must be an array of non-empty strings (may be empty)")
    return p


def _entries_dir(root: Path) -> Path:
    return root / "entries"


def _load_all(root: Path):
    out = []
    for f in sorted(_entries_dir(root).glob("*.json")):
        try:
            out.append((f, json.loads(f.read_text())))
        except Exception as e:
            out.append((f, {"__parse_error__": str(e)}))
    return out


def cmd_add(args, root: Path) -> int:
    try:
        entry = json.loads(Path(args.entry).read_text())
    except Exception as e:
        print(f"error: cannot read entry: {e}", file=sys.stderr)
        return 2
    problems = validate(entry)
    if problems:
        for pr in problems:
            print(f"invalid: {pr}")
        return 1
    if args.check:
        print("valid (not installed; --check)")
        return 0
    dest = _entries_dir(root) / f"{entry['id']}.json"
    if dest.exists():
        print(f"error: {dest} already exists — the knowledge base is append-only; "
              "pick a new id to supersede it", file=sys.stderr)
        return 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(entry, indent=2) + "\n")
    print(f"installed {dest}")
    return 0


def cmd_lint(args, root: Path) -> int:
    total = 0
    entries = _load_all(root)
    for f, entry in entries:
        if "__parse_error__" in entry:
            print(f"{f.name}: does not parse: {entry['__parse_error__']}")
            total += 1
            continue
        for pr in validate(entry, expected_id=f.stem):
            print(f"{f.name}: {pr}")
            total += 1
    print(f"knowledge lint: {len(entries)} entries, {total} problems")
    return 1 if total else 0


def _matches(entry, args) -> bool:
    exact = {"model": entry.get("model"), "family": entry.get("family"),
             "op": entry.get("op"), "phase": entry.get("phase"),
             "task": entry.get("task"),
             "status": (entry.get("result") or {}).get("status"),
             "bottleneck": (entry.get("bottleneck") or {}).get("kind")}
    for k, have in exact.items():
        want = getattr(args, k)
        if want and (have or "").lower() != want.lower():
            return False
    hw = entry.get("hardware") or {}
    for k, have in (("gpu", hw.get("gpu")), ("sm", hw.get("sm"))):
        want = getattr(args, k)
        if want and want.lower() not in (have or "").lower():
            return False
    if args.technique:
        techs = " ".join(a.get("technique", "") for a in entry.get("approaches", [])
                         if isinstance(a, dict))
        if args.technique.lower() not in techs.lower():
            return False
    return True


def cmd_query(args, root: Path) -> int:
    entries = [e for _, e in _load_all(root) if "__parse_error__" not in e]
    hits = [e for e in entries if _matches(e, args)]
    hits.sort(key=lambda e: (e.get("date", ""), e.get("id", "")), reverse=True)
    if args.json:
        print(json.dumps(hits, indent=2))
        return 0
    if not hits:
        print(f"no matching entries ({len(entries)} total)")
        return 0
    for e in hits:
        res = e.get("result") or {}
        print(f"{e.get('id')}  [{res.get('status')} geo={res.get('geomean_speedup')} "
              f"minc={res.get('min_speedup_conservative')}]  "
              f"{e.get('family')}/{e.get('op')} {e.get('phase')}  "
              f"{(e.get('hardware') or {}).get('gpu')} "
              f"{(e.get('hardware') or {}).get('sm')}  "
              f"bottleneck={(e.get('bottleneck') or {}).get('kind')}")
        print(f"    lesson: {e.get('lesson')}")
        for a in e.get("approaches", []):
            if isinstance(a, dict):
                print(f"    - {a.get('technique')} [{a.get('outcome')}"
                      f"{' ' + str(a.get('geomean_speedup')) if a.get('geomean_speedup') is not None else ''}]"
                      f": {a.get('why')}")
    print(f"{len(hits)} matching entries ({len(entries)} total)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).resolve().parent.parent / "knowledge",
                    help="knowledge-base root (default: testbench/knowledge)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="validate an entry file and install it")
    a.add_argument("entry")
    a.add_argument("--check", action="store_true", help="validate only, do not install")
    sub.add_parser("lint", help="validate every installed entry")
    q = sub.add_parser("query", help="filter entries and print recipes")
    for flag in ("model", "family", "op", "phase", "task", "status", "bottleneck"):
        q.add_argument(f"--{flag}")
    q.add_argument("--gpu", help="substring match, e.g. B200")
    q.add_argument("--sm", help="substring match, e.g. sm_100")
    q.add_argument("--technique", help="substring match over approach techniques")
    q.add_argument("--json", action="store_true", help="print matches as JSON")
    args = ap.parse_args()
    return {"add": cmd_add, "lint": cmd_lint, "query": cmd_query}[args.cmd](args, args.root)


if __name__ == "__main__":
    raise SystemExit(main())
