#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DYNAMO_SRC="${DYNAMO_SRC:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
ROOT="${ROOT:-/tmp/connorc_sglang_smg_disagg_bench}"
VENV="${VENV:-$ROOT/venv}"

export HOME="${HOME:-$ROOT/home}"
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/pip_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/uv_cache}"
export CARGO_HOME="${CARGO_HOME:-$ROOT/cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$ROOT/rustup}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/cargo-target}"
export PATH="$CARGO_HOME/bin:$VENV/bin:$PATH"

mkdir -p "$HOME" "$HF_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" \
    "$CARGO_HOME" "$RUSTUP_HOME" "$CARGO_TARGET_DIR" "$ROOT/bin"

if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
else
    python3 -m venv --system-site-packages "$VENV"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

python -m pip install --upgrade pip wheel "setuptools<81.0.0,>=77.0.3"
python -m pip install "maturin[patchelf]>=1.0,<2.0" uv

if ! command -v cargo >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable
    # shellcheck disable=SC1091
    source "$CARGO_HOME/env"
fi

rustc --version
cargo --version

uv pip install --prerelease=allow \
    aiperf \
    "smg-grpc-servicer[sglang]>=0.5.2"

cd "$DYNAMO_SRC/lib/bindings/python"
maturin develop --uv

cd "$DYNAMO_SRC"
uv pip install --prerelease=allow -e ".[sglang]"
uv pip install --prerelease=allow -e lib/gpu_memory_service

cargo build --locked -p dynamo-sglang-smg-sidecar --bin dynamo-sglang-smg-sidecar
cp -f "$CARGO_TARGET_DIR/debug/dynamo-sglang-smg-sidecar" "$ROOT/bin/dynamo-sglang-smg-sidecar"
cp -f "$ROOT/bin/dynamo-sglang-smg-sidecar" "$VENV/bin/dynamo-sglang-smg-sidecar"
chmod +x "$VENV/bin/dynamo-sglang-smg-sidecar"

python - <<'PY'
import importlib.metadata as md

for name in [
    "ai-dynamo",
    "ai-dynamo-runtime",
    "gpu-memory-service",
    "sglang",
    "smg-grpc-servicer",
    "smg-grpc-proto",
    "aiperf",
]:
    try:
        print(f"{name}={md.version(name)}")
    except Exception as exc:
        print(f"{name}=MISSING {exc}")
PY

command -v dynamo-sglang-smg-sidecar
echo "SGLang disagg benchmark host environment ready: $ROOT"
