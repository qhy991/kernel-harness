#!/usr/bin/env bash
# Apply the vendored winning candidate to the Kernel-Harness task path, so the
# harness gate (evaluate.py / integrate.py / agent_closeout.py) runs it.
#
# The optimized kernel lives in the sibling Kernel-Harness checkout, reached via
# the repo-root `Kernel-Harness/` symlink which is NOT tracked by this repo.
# This repo therefore vendors the exact solution.py here so the executable change
# is committed and reproducible; run this script to place it at the gate path.
#
# Usage:
#   ./apply.sh            # copy vendored solution.py -> harness task path, verify
#   KDA_HARNESS_ROOT=/path ./apply.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
HARNESS_ROOT="${KDA_HARNESS_ROOT:-/home/qinhaiyan/Kernel-Harness}"
DEST="$HARNESS_ROOT/testbench/tasks/glm52/o_proj_decode/solution.py"
SRC="$HERE/solution.py"

if [[ ! -f "$SRC" ]]; then
    echo "error: vendored source not found: $SRC" >&2
    exit 1
fi
if [[ ! -d "$(dirname "$DEST")" ]]; then
    echo "error: harness task dir not found: $(dirname "$DEST")" >&2
    echo "       set KDA_HARNESS_ROOT to your Kernel-Harness checkout." >&2
    exit 1
fi

cp "$SRC" "$DEST"

# Verify byte-identical
if cmp -s "$SRC" "$DEST"; then
    echo "applied + verified byte-identical:"
    echo "  src : $SRC"
    echo "  dest: $DEST"
    sha256sum "$SRC" "$DEST"
else
    echo "error: copy mismatch after apply" >&2
    exit 1
fi

# Print reproduction commands with the ALREADY-RESOLVED harness root expanded,
# so custom-root users (KDA_HARNESS_ROOT=/path ./apply.sh) reproduce against the
# exact checkout this script just modified — not the default. (Unquoted heredoc
# expands $HARNESS_ROOT; there are no other shell metacharacters to guard.)
cat <<EOF

Reproduce the WIN (harness .venv, idle B200 GPU 0) against the checkout just modified:
  cd "$HARNESS_ROOT"
  export REMOTE_GPU_ID=0 CUDA_VISIBLE_DEVICES=0
  .venv/bin/python testbench/bin/evaluate.py testbench/tasks/glm52/o_proj_decode --repeat 3     # exit 0 = WIN
  .venv/bin/python testbench/bin/integrate.py testbench/tasks/glm52/o_proj_decode              # DROP-IN VERIFIED
  .venv/bin/python testbench/bin/agent_closeout.py glm52/o_proj_decode --repeat 3 --owner qinhaiyan
EOF
