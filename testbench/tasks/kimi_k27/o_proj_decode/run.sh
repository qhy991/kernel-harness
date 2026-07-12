#!/usr/bin/env bash
# Self-contained entrypoint: evaluate THIS task folder against the sglang
# baseline. Forwards args to bin/evaluate.py (e.g. ./run.sh --repeat 3).
#
# This is the AUTHORITATIVE test (CUPTI cold-L2 100-rep + correctness) — the
# WIN/lose gate. For a fast, no-verdict probe while iterating, use the harness
# profiler:  python3 -m harness.profile <this-dir> --shape M   (advisory only).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HERE/../../../bin/evaluate.py" "$HERE" "$@"
