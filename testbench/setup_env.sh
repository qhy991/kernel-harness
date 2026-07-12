#!/usr/bin/env bash
# Create the one supported test environment: <repo>/.venv managed by uv.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
eval "$(python3 "$HERE/bin/config.py")"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required. Install it from https://docs.astral.sh/uv/" >&2
  exit 2
fi
if [[ ! -f "$SGLANG_DIR/python/pyproject.toml" ]]; then
  echo "error: SGLANG_DIR does not point to an SGLang checkout: $SGLANG_DIR" >&2
  echo "Set SGLANG_DIR or copy testbench/harness.env.example to testbench/harness.env." >&2
  exit 2
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python "${PYTHON_VERSION:-3.12}" "$VENV"
fi
uv pip install --python "$VENV/bin/python" -e "$SGLANG_DIR/python"
uv pip install --python "$VENV/bin/python" -r "$HERE/requirements.txt"

echo
echo "Environment ready: $VENV"
"$VENV/bin/python" "$HERE/bin/check_env.py"
echo "Activate with: source $ROOT/activate_env.sh"
