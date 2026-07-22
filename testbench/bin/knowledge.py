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
import os
import re
import sys
from pathlib import Path

try:  # optional: prior-art bridge (stdlib-only sibling; absent-safe)
    import kwiki_bridge
except Exception:  # pragma: no cover - bridge is best-effort
    kwiki_bridge = None

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
        legacy_win = _num(msc) and msc > 1.0
        if str(entry.get("task", "")).startswith("glm52/"):
            # GLM-5.2 stopped gating on min_speedup_conservative: its runner judges each
            # shape win/neutral/regress and passes on >=1 win with 0 regressions, so a
            # legitimate win that falls back to the reference on one shape has msc < 1.0
            # by construction and the legacy rule alone would reject it. Either form is
            # accepted, because the legacy one implies the new one — msc > 1.0 means
            # every shape won, hence >=1 win and no regression. That also keeps the
            # pre-consolidation entries (which predate these fields) valid, as an
            # append-only log requires.
            won, regressed = res.get("shapes_won"), res.get("shapes_regressed")
            shape_win = isinstance(won, int) and won >= 1 and regressed == 0
            if not (shape_win or legacy_win):
                p.append("result.status 'win' for glm52 requires either "
                         "result.shapes_won >= 1 with result.shapes_regressed == 0 "
                         "(from result.json's aggregate), or the legacy "
                         "min_speedup_conservative > 1.0")
        elif not legacy_win:
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


def _default_root() -> Path:
    """Knowledge-base root. $KH_KNOWLEDGE_ROOT lets a whole fleet of worktrees share
    one bank (recipes learned in one worktree are visible to the others); otherwise
    the in-repo testbench/knowledge is used."""
    env = os.environ.get("KH_KNOWLEDGE_ROOT")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent.parent / "knowledge"


def _load_candidates(root: Path) -> dict:
    """Library-kernel-first ledger: per-op current best library drop-in. JSON (not
    yaml) to keep this tool stdlib-only. Empty dict if absent/unreadable."""
    f = root / "candidates.json"
    try:
        data = json.loads(f.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


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


def _fmt_entry(e) -> str:
    res = e.get("result") or {}
    lines = [f"{e.get('id')}  [{res.get('status')} geo={res.get('geomean_speedup')} "
             f"minc={res.get('min_speedup_conservative')}]  "
             f"bottleneck={(e.get('bottleneck') or {}).get('kind')}",
             f"    lesson: {e.get('lesson')}"]
    for a in e.get("approaches", []):
        if isinstance(a, dict):
            gs = a.get("geomean_speedup")
            lines.append(f"    - {a.get('technique')} [{a.get('outcome')}"
                         f"{' ' + str(gs) if gs is not None else ''}]: {a.get('why')}")
    return "\n".join(lines)


def _op_slug(task_token: str | None) -> str:
    """o_proj_decode -> o_proj ; glm52/o_proj_prefill -> o_proj."""
    t = (task_token or "").split("/")[-1]
    for suf in ("_decode", "_prefill"):
        if t.endswith(suf):
            return t[:-len(suf)]
    return t


def cmd_brief(args, root: Path) -> int:
    """Warm-start digest for a task: internal recipes + library ledger + KernelWiki
    prior-art. The one call a session runs before touching a kernel, so retrieval is
    load-bearing instead of discretionary. Keyed on the task-dir token (e.g.
    o_proj_decode), which matches entries by task/op substring — entries store `op`
    as a descriptive name, so exact --op matching is unreliable."""
    token = args.task or args.op
    tok = (token or "").lower()
    slug = _op_slug(token)
    entries = [e for _, e in _load_all(root) if "__parse_error__" not in e]

    def _match(e) -> bool:
        if tok and tok not in (e.get("task") or "").lower() \
                and slug not in (e.get("task") or "").lower():
            return False
        if args.phase and (e.get("phase") or "") != args.phase:
            return False
        if args.bottleneck and (e.get("bottleneck") or {}).get("kind", "") != args.bottleneck:
            return False
        return True

    hits = sorted((e for e in entries if _match(e)),
                  key=lambda e: (e.get("date", ""), e.get("id", "")), reverse=True)
    # Infer the bottleneck from the newest hit when the caller didn't pass one, so the
    # KernelWiki query still targets the right pattern.
    bottleneck = args.bottleneck or ((hits[0].get("bottleneck") or {}).get("kind") if hits else None)

    print(f"== internal recipes ({len(hits)}/{len(entries)})  "
          f"task~{token or '*'} phase={args.phase or '*'} bottleneck={bottleneck or '*'} ==")
    for e in hits[:args.limit]:
        print(_fmt_entry(e))
    if not hits:
        print("  (none — this is new ground; write one at session end)")

    entry = _load_candidates(root).get(slug)
    if entry:
        print(f"\n== library-kernel-first ledger: {slug} ==")
        for k in ("best_library", "call", "dtype", "layout", "wins_where",
                  "handwrite_where", "source"):
            if entry.get(k):
                print(f"    {k}: {entry[k]}")

    # Recurring dead-ends for this bottleneck (from `distill`), so the feedback loop
    # closes back at warm-start: don't re-try what already failed across sessions.
    dist = _load_json(root, "distilled.json")
    if dist:
        pit = [m for m in dist.get("mistakes", [])
               if m.get("recurring") and (not bottleneck or m["bottleneck"] == bottleneck)]
        if pit:
            print("\n== recurring pitfalls (already-tried dead ends — avoid) ==")
            for m in pit[:args.limit]:
                flag = "[promoted]" if m.get("promoted") else f"[promote→{m['suggested_owner']}]"
                why = m["whys"][0] if m.get("whys") else ""
                print(f"    {m['technique']} ×{m['count']} ({m['bottleneck']}) {flag}: {why}")

    if not args.no_external and kwiki_bridge is not None:
        techs = [a.get("technique") for e in hits[:2]
                 for a in e.get("approaches", []) if isinstance(a, dict)]
        ext = kwiki_bridge.query(op=slug, bottleneck=bottleneck, techniques=techs,
                                 limit=args.limit)
        print("\n== KernelWiki prior-art ==")
        print(ext if ext else "  (KernelWiki unavailable or no match — internal-only)")
    return 0


def _group_index(entries, key_fn) -> str:
    groups: dict = {}
    for e in entries:
        for key in key_fn(e):
            groups.setdefault(key, []).append(e)
    out = []
    for key in sorted(groups):
        out.append(f"## {key}\n")
        for e in sorted(groups[key], key=lambda x: (x.get("date", ""), x.get("id", "")),
                        reverse=True):
            res = e.get("result") or {}
            out.append(f"- `{e.get('id')}` [{res.get('status')}] "
                       f"{e.get('op')}/{e.get('phase')} — {e.get('lesson')}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def cmd_index(args, root: Path) -> int:
    """Generate cross-reference indices (mirrors KernelWiki's queries/). --check
    verifies they are up to date (for CI) without writing."""
    entries = [e for _, e in _load_all(root) if "__parse_error__" not in e]
    header = "<!-- generated by knowledge.py index; do not edit by hand -->\n\n"
    indices = {
        "by-op.md": _group_index(entries, lambda e: [e.get("op") or "unknown"]),
        "by-bottleneck.md": _group_index(
            entries, lambda e: [(e.get("bottleneck") or {}).get("kind") or "unknown"]),
        "by-technique.md": _group_index(
            entries, lambda e: sorted({a.get("technique") for a in e.get("approaches", [])
                                       if isinstance(a, dict) and a.get("technique")}) or ["unknown"]),
    }
    qdir = root / "queries"
    stale = 0
    for name, body in indices.items():
        content = header + f"# {name[:-3]}\n\n" + body
        dest = qdir / name
        if args.check:
            cur = dest.read_text() if dest.exists() else ""
            if cur != content:
                print(f"stale: {dest}")
                stale += 1
        else:
            qdir.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)
            print(f"wrote {dest}")
    if args.check:
        print(f"index --check: {stale} stale")
        return 1 if stale else 0
    return 0


FAIL_OUTCOMES = {"slower", "incorrect", "error", "abandoned"}
PROMOTION_OWNERS = {"prompt", "doc", "reviewer", "diagnostic", "typed-boundary", "lint"}


def _norm_tech(t) -> str:
    t = re.sub(r"[^a-z0-9]+", "-", (t or "").strip().lower()).strip("-")
    return t or "unknown"


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if not n:
        return None
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _load_json(root: Path, name: str):
    try:
        return json.loads((root / name).read_text())
    except Exception:
        return None


def _suggest_owner(outcomes: set) -> str:
    # A wrong-result class wants to become unavailable/loud (typed boundary or an
    # actionable gate diagnostic); a strategy dead-end wants to warn the next session.
    if outcomes & {"incorrect", "error"}:
        return "typed-boundary|diagnostic"
    return "doc|reviewer"


def _distill(root: Path) -> dict:
    """Aggregate the bank into recurring failure-classes (mistakes) and proven
    techniques (learnings), keyed by bottleneck::technique. This is the sensor half
    of turn-feedback-into-infrastructure: it surfaces the governing failure class so
    a lesson can be promoted to its smallest durable owner instead of re-learned."""
    entries = [e for _, e in _load_all(root) if "__parse_error__" not in e]
    mistakes: dict = {}
    learnings: dict = {}
    for e in entries:
        bk = (e.get("bottleneck") or {}).get("kind") or "unknown"
        for a in e.get("approaches", []):
            if not isinstance(a, dict):
                continue
            key = f"{bk}::{_norm_tech(a.get('technique'))}"
            out = a.get("outcome")
            if out in FAIL_OUTCOMES:
                m = mistakes.setdefault(key, {"bottleneck": bk, "technique": _norm_tech(a.get("technique")),
                                              "count": 0, "entries": set(), "ops": set(),
                                              "outcomes": set(), "whys": []})
                m["count"] += 1
                m["entries"].add(e.get("id"))
                m["ops"].add(e.get("op"))
                m["outcomes"].add(out)
                if a.get("why") and len(m["whys"]) < 3:
                    m["whys"].append(a["why"])
            elif out in ("win", "partial"):
                lr = learnings.setdefault(key, {"bottleneck": bk, "technique": _norm_tech(a.get("technique")),
                                                "count": 0, "ops": set(), "geos": []})
                lr["count"] += 1
                lr["ops"].add(e.get("op"))
                if _num(a.get("geomean_speedup")):
                    lr["geos"].append(a["geomean_speedup"])
    promoted = {p.get("class") for p in (_load_json(root, "promotions.json") or []) if isinstance(p, dict)}

    def mrec(k, m):
        return {"class": k, "bottleneck": m["bottleneck"], "technique": m["technique"],
                "count": m["count"], "distinct_entries": len(m["entries"]),
                "ops": sorted(x for x in m["ops"] if x), "outcomes": sorted(m["outcomes"]),
                "whys": m["whys"], "suggested_owner": _suggest_owner(m["outcomes"]),
                "promoted": k in promoted,
                "recurring": len(m["entries"]) >= 2 or m["count"] >= 2}

    def lrec(k, lr):
        return {"class": k, "bottleneck": lr["bottleneck"], "technique": lr["technique"],
                "count": lr["count"], "ops": sorted(x for x in lr["ops"] if x),
                "median_geomean": (round(_median(lr["geos"]), 4) if lr["geos"] else None)}

    return {
        "generated_from_entries": len(entries),
        "mistakes": sorted((mrec(k, m) for k, m in mistakes.items()),
                           key=lambda r: (r["recurring"], r["count"], r["class"]), reverse=True),
        "learnings": sorted((lrec(k, lr) for k, lr in learnings.items()),
                            key=lambda r: (r["count"], r["class"]), reverse=True),
    }


def _distilled_md(data: dict) -> str:
    out = ["<!-- generated by knowledge.py distill; do not edit by hand -->\n",
           f"# distilled — recurring failure-classes & proven techniques "
           f"({data['generated_from_entries']} entries)\n",
           "## Recurring mistakes (promotion candidates)\n"]
    rec = [m for m in data["mistakes"] if m["recurring"]]
    for m in rec or [None]:
        if m is None:
            out.append("- (none recurring yet)")
            break
        flag = "promoted" if m["promoted"] else f"PROMOTE → {m['suggested_owner']}"
        out.append(f"- **{m['technique']}** ({m['bottleneck']}) ×{m['count']} over "
                   f"{m['distinct_entries']} entries {m['ops']} — {flag}\n"
                   f"  - {' '.join(m['whys'][:2])}")
    out.append("\n## Proven techniques (by bottleneck)\n")
    for lr in data["learnings"] or []:
        out.append(f"- **{lr['technique']}** ({lr['bottleneck']}) ×{lr['count']} "
                   f"geo~{lr['median_geomean']} {lr['ops']}")
    return "\n".join(out).rstrip() + "\n"


def cmd_distill(args, root: Path) -> int:
    data = _distill(root)
    payload = json.dumps(data, indent=2) + "\n"
    md_payload = _distilled_md(data)
    if args.json:
        print(payload, end="")
        return 0
    jdst, mdst = root / "distilled.json", root / "distilled.md"
    if args.check:
        stale = []
        if (jdst.read_text() if jdst.exists() else "") != payload:
            stale.append("distilled.json")
        if (mdst.read_text() if mdst.exists() else "") != md_payload:
            stale.append("distilled.md")
        if stale:
            print(f"stale: {', '.join(stale)} (run `knowledge.py distill`)")
            return 1
        print("distill --check: up to date")
        return 0
    jdst.write_text(payload)
    mdst.write_text(md_payload)
    rec = sum(1 for m in data["mistakes"] if m["recurring"] and not m["promoted"])
    print(f"wrote {jdst} and {mdst}: {len(data['mistakes'])} mistake-classes "
          f"({rec} recurring & un-promoted), {len(data['learnings'])} proven techniques")
    return 0


def cmd_promote(args, root: Path) -> int:
    """Record that a recurring lesson has been promoted to its smallest durable owner
    (the promotion ladder: prompt → doc → reviewer → diagnostic → typed-boundary →
    lint). Append-only ledger; distill/brief read it to stop re-surfacing solved gaps."""
    if args.to not in PROMOTION_OWNERS:
        print(f"error: --to must be one of {sorted(PROMOTION_OWNERS)}", file=sys.stderr)
        return 2
    promos = _load_json(root, "promotions.json") or []
    if not isinstance(promos, list):
        promos = []
    promos.append({"class": args.klass, "to": args.to, "owner_path": args.owner_path or "",
                   "note": args.note or "", "date": datetime.date.today().isoformat()})
    (root / "promotions.json").write_text(json.dumps(promos, indent=2) + "\n")
    print(f"promoted {args.klass!r} -> {args.to}"
          f"{f' ({args.owner_path})' if args.owner_path else ''}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=_default_root(),
                    help="knowledge-base root (default: $KH_KNOWLEDGE_ROOT or testbench/knowledge)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="validate an entry file and install it")
    a.add_argument("entry")
    a.add_argument("--check", action="store_true", help="validate only, do not install")
    sub.add_parser("lint", help="validate every installed entry")
    q = sub.add_parser("query", help="filter entries and print recipes")
    for flag in ("model", "family", "op", "phase", "task", "status", "bottleneck"):
        q.add_argument(f"--{flag}")
    q.add_argument("--gpu", help="substring match, e.g. MI300X")
    q.add_argument("--sm", help="substring match, e.g. sm_100")
    q.add_argument("--technique", help="substring match over approach techniques")
    q.add_argument("--json", action="store_true", help="print matches as JSON")
    b = sub.add_parser("brief", help="warm-start: internal recipes + ledger + KernelWiki")
    for flag in ("op", "phase", "task", "bottleneck"):
        b.add_argument(f"--{flag}")
    b.add_argument("--limit", type=int, default=6)
    b.add_argument("--no-external", action="store_true", help="skip the KernelWiki bridge")
    ix = sub.add_parser("index", help="(re)generate queries/*.md cross-reference indices")
    ix.add_argument("--check", action="store_true", help="verify up-to-date, do not write")
    d = sub.add_parser("distill", help="mine recurring failure-classes + proven techniques")
    d.add_argument("--json", action="store_true", help="print to stdout, do not write files")
    d.add_argument("--check", action="store_true", help="verify distilled.json/md up-to-date")
    pr = sub.add_parser("promote", help="record a lesson promoted to its smallest durable owner")
    pr.add_argument("--class", dest="klass", required=True, help="failure-class key from distill")
    pr.add_argument("--to", required=True, help="|".join(sorted(PROMOTION_OWNERS)))
    pr.add_argument("--owner-path", help="the file/check that now owns it")
    pr.add_argument("--note")
    args = ap.parse_args()
    return {"add": cmd_add, "lint": cmd_lint, "query": cmd_query,
            "brief": cmd_brief, "index": cmd_index,
            "distill": cmd_distill, "promote": cmd_promote}[args.cmd](args, args.root)


if __name__ == "__main__":
    raise SystemExit(main())
