#!/usr/bin/env bash
# Self-contained entrypoint: evaluate THIS task folder against the sglang
# baseline. Forwards args to bin/evaluate.py (e.g. ./run.sh --repeat 3).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HERE/../../../bin/evaluate.py" "$HERE" "$@"
