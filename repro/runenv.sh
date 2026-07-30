# Portable environment for reproducing the GLM-5.2 AMD op-level GATE-1 results.
#
#   source repro/runenv.sh
#
# It sets three things:
#   (1) the ROCm / MI300X (gfx942) runtime knobs,
#   (2) the FROZEN gate identity the wins were measured under, and
#   (3) the venv + source-tree locations (aiter, sglang).
#
# Everything is overridable via environment variables. The only values you MUST
# provide for your machine are the three locations in section 1 — either edit them
# here, or `export` them before sourcing:
#
#   export ROCM_TORCH_VENV=/opt/rocm-venvs/rocm-torch \
#          AITER_ROOT=/src/aiter SGLANG_ROOT=/src/sglang
#   source repro/runenv.sh
#
# This file contains NO credentials and sources no personal profile. (The original
# author's private env additionally exported LLM-gateway tokens; those are irrelevant
# to reproduction and are deliberately excluded here.)

# ---------------------------------------------------------------------------
# 1. Machine locations — SET THESE for your box (edit, or export before sourcing).
# ---------------------------------------------------------------------------
# ROCm install (7.0.0 was used for the reference numbers; must target gfx942 / MI300X).
export ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
# Python venv holding torch 2.10.0+rocm7.0 and triton-rocm 3.6.0 (see REPRODUCE.md).
export ROCM_TORCH_VENV="${ROCM_TORCH_VENV:-/path/to/rocm-torch-venv}"
# Source checkouts added to PYTHONPATH (reference commits: aiter 2ca7878e2, sglang 20fc529ab).
export AITER_ROOT="${AITER_ROOT:-/path/to/aiter}"
export SGLANG_ROOT="${SGLANG_ROOT:-/path/to/sglang}"

if [ ! -x "${ROCM_TORCH_VENV}/bin/python" ]; then
  echo "[repro] ERROR: no python at '${ROCM_TORCH_VENV}/bin/python'." >&2
  echo "[repro]        Set ROCM_TORCH_VENV to your rocm-torch venv (see repro/REPRODUCE.md)," >&2
  echo "[repro]        e.g.  export ROCM_TORCH_VENV=/your/venv  then re-source this file." >&2
  return 1 2>/dev/null || exit 1
fi
for _d in "$AITER_ROOT" "$SGLANG_ROOT"; do
  [ -d "$_d" ] || echo "[repro] WARNING: source tree not found: '$_d' (import may fail)." >&2
done

# ---------------------------------------------------------------------------
# 2. ROCm runtime (gfx942 / MI300X).
# ---------------------------------------------------------------------------
export PATH="${ROCM_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_HOME}/lib:${ROCM_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_ROCM_ARCH=gfx942
export GPU_ARCHS=gfx942
export HIP_CLANG_PATH="${ROCM_HOME}/lib/llvm/bin"

# Per-machine caches (default to scratch; override ROCM_REPRO_CACHE to a fast local disk).
export ROCM_REPRO_CACHE="${ROCM_REPRO_CACHE:-${TMPDIR:-/tmp}/glm52-repro-cache}"
export TMPDIR="${ROCM_REPRO_CACHE}/tmp"
export AITER_CONFIG_DIR="${ROCM_REPRO_CACHE}/aiter_configs"
export TRITON_CACHE_DIR="${ROCM_REPRO_CACHE}/triton"
export XDG_CACHE_HOME="${ROCM_REPRO_CACHE}/xdg"
export SGLANG_CACHE_DIR="${ROCM_REPRO_CACHE}/sglang"
mkdir -p "$TMPDIR" "$AITER_CONFIG_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$SGLANG_CACHE_DIR" 2>/dev/null || true

# aiter + sglang are used from source (not pip-installed) — put them on PYTHONPATH.
export PYTHONPATH="${AITER_ROOT}:${AITER_ROOT}/aiter/jit/utils:${SGLANG_ROOT}/python:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# 3. Frozen gate identity — DO NOT change; the wins were measured under exactly this.
# ---------------------------------------------------------------------------
export KERNEL_HARNESS_PLATFORM="${KERNEL_HARNESS_PLATFORM:-rocm}"
export KERNEL_HARNESS_PROFILE="${KERNEL_HARNESS_PROFILE:-amd-mi300x}"
export KERNEL_HARNESS_PROVIDER="${KERNEL_HARNESS_PROVIDER:-aiter-torch-reference}"
export KERNEL_HARNESS_TIMER="${KERNEL_HARNESS_TIMER:-event}"
# aiter path ON — the dsa ASM ~1.7 ms baseline and the fp8_mqa_logits dispatch need it.
export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
export AITER_TRITON_ONLY="${AITER_TRITON_ONLY:-0}"

# ---------------------------------------------------------------------------
# 4. GPU selection — pin to ONE healthy card.
# ---------------------------------------------------------------------------
# NOTE: HIP_VISIBLE_DEVICES orders by PCI bus and does NOT match rocm-smi's GPU
# numbering. Confirm the chosen card is healthy (a tiny torch matmul, or rocminfo)
# before trusting timings — a degraded card silently inflates and destabilises them.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

echo "[repro] venv=$ROCM_TORCH_VENV"
echo "[repro] arch=gfx942 platform=$KERNEL_HARNESS_PLATFORM provider=$KERNEL_HARNESS_PROVIDER timer=$KERNEL_HARNESS_TIMER"
echo "[repro] SGLANG_USE_AITER=$SGLANG_USE_AITER AITER_TRITON_ONLY=$AITER_TRITON_ONLY HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES cache=$ROCM_REPRO_CACHE"
