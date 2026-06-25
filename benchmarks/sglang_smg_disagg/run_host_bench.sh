#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/tmp/connorc_sglang_smg_disagg_bench}"

mkdir -p "$ROOT"
"$SCRIPT_DIR/setup_host_env.sh"

export ROOT
export VENV="${VENV:-$ROOT/venv}"
export HOME="${HOME:-$ROOT/home}"
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/pip_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/uv_cache}"
export CARGO_HOME="${CARGO_HOME:-$ROOT/cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$ROOT/rustup}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/cargo-target}"
export PATH="$VENV/bin:$CARGO_HOME/bin:$ROOT/bin:$PATH"

# shellcheck disable=SC1091
source "$VENV/bin/activate"

"$SCRIPT_DIR/sweep_h100_mini.sh"
