#!/usr/bin/env python3
"""Fail closed unless one serving-native audit is a valid performance win."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def validate_audit_document(document: Any) -> list[str]:
    if not isinstance(document, dict):
        return ["audit JSON must be an object"]
    problems: list[str] = []
    if document.get("valid") is not True:
        problems.append("audit valid must be the exact boolean true")
    if document.get("performance_gate_passed") is not True:
        problems.append(
            "audit performance_gate_passed must be the exact boolean true"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_json", type=Path)
    args = parser.parse_args(argv)
    try:
        document = json.loads(args.audit_json.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"invalid audit JSON: {exc}", file=sys.stderr)
        return 1
    problems = validate_audit_document(document)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
