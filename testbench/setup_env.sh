#!/usr/bin/env bash
# Create the one supported test environment: <repo>/.venv managed by uv.
set -euo pipefail

DRY_RUN=0
PRINT_CONFIG=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --print-config) PRINT_CONFIG=1 ;;
    -h|--help)
      cat <<'EOF'
usage: testbench/setup_env.sh [--dry-run] [--print-config]

Create or refresh the supported testbench environment. --dry-run resolves the
effective paths and runs the environment check if the venv already exists, but
does not install packages. --print-config only prints the resolved paths.
EOF
      exit 0
      ;;
    *)
      echo "error: unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
SGLANG_DIR="${SGLANG_DIR:-$ROOT/../sglang}"
if [[ -f "$HERE/harness.env" ]]; then
  # shellcheck disable=SC1091
  source "$HERE/harness.env"
fi

resolve_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    realpath -m "$p"
  else
    realpath -m "$ROOT/$p"
  fi
}

VENV="$(resolve_path "$VENV")"
CUDA_HOME="$(resolve_path "$CUDA_HOME")"
SGLANG_DIR="$(resolve_path "$SGLANG_DIR")"
export VENV CUDA_HOME SGLANG_DIR

print_config() {
  printf 'ROOT=%s\n' "$ROOT"
  printf 'VENV=%s\n' "$VENV"
  printf 'CUDA_HOME=%s\n' "$CUDA_HOME"
  printf 'SGLANG_DIR=%s\n' "$SGLANG_DIR"
  printf 'PYTHON_VERSION=%s\n' "${PYTHON_VERSION:-3.12}"
}

if [[ "$PRINT_CONFIG" -eq 1 ]]; then
  print_config
  exit 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  print_config
  if [[ -x "$VENV/bin/python" ]]; then
    "$VENV/bin/python" "$HERE/bin/check_env.py"
  else
    echo "warning: venv does not exist yet; dry-run skipped check_env.py: $VENV" >&2
  fi
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required. Install it from https://docs.astral.sh/uv/" >&2
  exit 2
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python "${PYTHON_VERSION:-3.12}" "$VENV"
fi
if [[ -f "$SGLANG_DIR/python/pyproject.toml" ]]; then
  uv pip install --python "$VENV/bin/python" -e "$SGLANG_DIR/python"
else
  echo "warning: SGLANG_DIR is not a checkout ($SGLANG_DIR); using installed sglang package" >&2
fi
uv pip install --python "$VENV/bin/python" -r "$HERE/requirements.txt"

echo
echo "Environment ready: $VENV"
"$VENV/bin/python" "$HERE/bin/check_env.py"
echo "Activate with: source $ROOT/activate_env.sh"
