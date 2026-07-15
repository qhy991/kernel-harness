#!/usr/bin/env bash
# Self-contained entrypoint: evaluate THIS task folder against the sglang
# baseline. Forwards args to bin/evaluate.py (e.g. ./run.sh --repeat 3).
#
# This is the AUTHORITATIVE test (CUPTI cold-L2 + correctness) — the WIN/lose
# gate. For a fast advisory probe from the repo root:
#   PYTHONPATH=testbench .venv/bin/python -m harness.profile "$HERE" --shape M
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HERE/../../../bin/evaluate.py" "$HERE" "$@"
