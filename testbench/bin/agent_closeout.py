#!/usr/bin/env python3
"""Agent session closeout: integrate check + structured CLOSEOUT_JSON.

Run after evaluate.py on a completed optimization session. On WIN, optionally
runs integrate.py and reports whether the solution is a real SGLang drop-in.

  .venv/bin/python testbench/bin/agent_closeout.py testbench/tasks/glm52/foo
  .venv/bin/python testbench/bin/agent_closeout.py glm52/foo --skip-integrate
  .venv/bin/python testbench/bin/agent_closeout.py glm52/foo --owner qinhaiyan \\
      --record-tokens  # append a row under token-records/<owner>/

Exit codes mirror evaluate semantics where possible:
  0 = WIN (+ drop-in verified when contract requires it, or fused-only noted)
  1 = correct but not faster, or WIN but drop-in failed
  2 = incorrect / usage error
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_BIN = Path(__file__).resolve().parent
_REPO = _BIN.parent.parent
_TASKS = _BIN.parent / "tasks"
_TOKEN_ROOT = _REPO / "token-records"


def _resolve_task(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir() and (p / "task.json").exists():
        return p.resolve()
    if "/" in arg:
        td = _TASKS / arg
        if (td / "task.json").exists():
            return td.resolve()
    raise SystemExit(f"task not found: {arg}")


def _parse_verdict_json(text: str) -> dict | None:
    m = re.search(r"VERDICT_JSON_BEGIN\s*(\{.*?\})\s*VERDICT_JSON_END", text, re.S)
    if not m:
        return None
    return json.loads(m.group(1))


def _parse_integration_json(text: str) -> dict | None:
    m = re.search(r"INTEGRATION_JSON_BEGIN\s*(\{.*?\})\s*INTEGRATION_JSON_END", text, re.S)
    if not m:
        return None
    return json.loads(m.group(1))


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def _append_token_row(owner: str, row: dict) -> Path:
    owner_dir = _TOKEN_ROOT / owner
    owner_dir.mkdir(parents=True, exist_ok=True)
    path = owner_dir / "experiments_tokens.csv"
    fieldnames = [
        "experiment_id", "batch", "owner", "task", "result", "evaluate_result",
        "integrate_status", "drop_in_verified", "integration_contract",
        "geomean_speedup", "min_speedup_conservative", "model", "reasoning_effort",
        "started_at", "notes",
    ]
    write_header = not path.exists() or path.stat().st_size == 0
    # If an existing richer CSV is present, append using its header when possible.
    if path.exists() and path.stat().st_size > 0:
        with path.open() as f:
            existing = next(csv.reader(f), None)
        if existing:
            fieldnames = existing
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", help="task dir or model/task")
    ap.add_argument("--skip-integrate", action="store_true")
    ap.add_argument("--repeat", type=int, default=0,
                    help="if >0, run evaluate.py --repeat N instead of reading prior verdict")
    ap.add_argument("--owner", default=os.environ.get("KH_TOKEN_OWNER", "qinhaiyan"),
                    help="token-records/<owner>/ subdirectory")
    ap.add_argument("--record-tokens", action="store_true",
                    help="append a summary row to token-records/<owner>/experiments_tokens.csv")
    ap.add_argument("--model-name", default=os.environ.get("KH_AGENT_MODEL", "gpt-5.5"))
    ap.add_argument("--effort", default=os.environ.get("KH_AGENT_EFFORT", "xhigh"))
    args = ap.parse_args()

    task_dir = _resolve_task(args.task)
    model = task_dir.parent.name
    rel = f"{model}/{task_dir.name}"
    py = _REPO / ".venv/bin/python"
    if not py.exists():
        py = Path(sys.executable)

    _, status_out = _run(
        [str(py), str(_BIN / "integration_status.py"), "--json", rel],
        _REPO,
    )
    contract = json.loads(status_out)[0]

    verdict: dict
    if args.repeat > 0:
        _, eval_out = _run(
            [str(py), str(_BIN / "evaluate.py"), str(task_dir.relative_to(_REPO)),
             "--repeat", str(args.repeat)],
            _REPO,
        )
        parsed = _parse_verdict_json(eval_out)
        if parsed is None:
            print(eval_out)
            raise SystemExit(2)
        verdict = parsed
    else:
        verdict = {"note": "no evaluate run; pass --repeat 3 to gate here",
                   "correct": None, "win": None}

    win = bool(verdict.get("win")) if verdict.get("win") is not None else False
    correct = bool(verdict.get("correct")) if verdict.get("correct") is not None else False
    gated = args.repeat > 0

    integration = None
    drop_in_ok = None
    integrate_status = "not-run"
    if win and not args.skip_integrate:
        _, integ_out = _run(
            [str(py), str(_BIN / "integrate.py"), str(task_dir)],
            _REPO,
        )
        integration = _parse_integration_json(integ_out)
        if integration is None:
            print(integ_out)
            integrate_status = "fail"
        elif integration.get("integration_contract") == "fused-only":
            drop_in_ok = None
            integrate_status = "no-recipe"
        else:
            drop_in_ok = integration.get("drop_in_ok")
            integrate_status = "pass" if drop_in_ok else "fail"
    elif contract["integration_contract"] == "fused-only":
        integrate_status = "no-recipe"
    elif not win:
        integrate_status = "not-run"

    closeout = {
        "task": rel,
        "contract": contract,
        "verdict": verdict,
        "integrate": integration,
        "drop_in_verified": drop_in_ok,
        "recommendation": _recommendation(contract, verdict, drop_in_ok, gated),
    }

    token_path = None
    if args.record_tokens:
        if not gated and verdict.get("win") is None:
            print("warning: --record-tokens without --repeat skips evaluate; "
                  "row will have unknown evaluate_result", file=sys.stderr)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        eval_result = (
            "win" if win else
            ("no-win" if correct else
             ("unknown" if not gated else "failed"))
        )
        row = {
            "experiment_id": f"closeout-{stamp}",
            "batch": "agent-closeout",
            "owner": args.owner,
            "task": rel,
            "result": eval_result,
            "evaluate_result": eval_result,
            "integrate_status": integrate_status,
            "drop_in_verified": "" if drop_in_ok is None else str(drop_in_ok),
            "integration_contract": contract.get("integration_contract", ""),
            "geomean_speedup": verdict.get("geomean_speedup", ""),
            "min_speedup_conservative": verdict.get("min_speedup_conservative", ""),
            "model": args.model_name,
            "reasoning_effort": args.effort,
            "started_at": stamp,
            "notes": "appended by agent_closeout.py --record-tokens",
        }
        token_path = _append_token_row(args.owner, row)
        closeout["token_record"] = str(token_path)

    print("CLOSEOUT_JSON_BEGIN")
    print(json.dumps(closeout, indent=2))
    print("CLOSEOUT_JSON_END")

    if not gated:
        # Contract-only / integrate-only mode: do not invent evaluate failure.
        raise SystemExit(0 if (drop_in_ok is not False) else 1)
    if not correct:
        raise SystemExit(2)
    if not win:
        raise SystemExit(1)
    if contract["integration_contract"] == "drop-in" and drop_in_ok is False:
        raise SystemExit(1)
    raise SystemExit(0)


def _recommendation(contract: dict, verdict: dict, drop_in_ok, gated: bool) -> str:
    if gated and not verdict.get("correct"):
        return "Fix correctness before further perf work."
    if gated and not verdict.get("win"):
        return "Record no-win knowledge entry; try a different bottleneck or task."
    ic = contract["integration_contract"]
    if ic == "fused-only":
        return ("WIN accepted (or fused-only). No isolated drop-in symbol — document "
                "interface contract in knowledge; do not fake integrate.py.")
    if drop_in_ok is True:
        return "WIN + drop-in verified. Record knowledge with integrate=pass."
    if drop_in_ok is False:
        return ("WIN on harness but drop-in failed — solution may not be safe in "
                "real sglang forward; fix before claiming production-ready.")
    if not gated:
        return "Run with --repeat 3 to gate; then integrate when contract is drop-in."
    return "WIN. Run integrate.py when contract is drop-in."


if __name__ == "__main__":
    main()
