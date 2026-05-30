#!/usr/bin/env bash
# CPU end-to-end test for the sidecar OpenEngine client (no dynamo, no vllm,
# no GPU). pytest collection would otherwise walk up the dynamo package chain
# and import dynamo/__init__ (needs the built Rust runtime), so we copy the
# test files to a neutral temp dir and point the bootstrap at the real sidecar
# source via OE_SIDECAR_DIR_OVERRIDE.
#
# Usage: ./run_cpu_tests.sh [/path/to/python]
#   default python: <repo>/.devvenv/bin/python
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIDECAR_DIR="$(cd "$HERE/.." && pwd)"
# repo root = vllm-sidecar (sidecar dir is .../dynamo/components/src/dynamo/vllm/sidecar)
REPO_ROOT="$(cd "$SIDECAR_DIR/../../../../../.." && pwd)"
PY="${1:-$REPO_ROOT/.devvenv/bin/python}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cp "$HERE"/_bootstrap.py "$HERE"/fake_servicer.py "$HERE"/test_client_e2e.py "$TMP"/

cd "$TMP"
OE_SIDECAR_DIR_OVERRIDE="$SIDECAR_DIR" "$PY" -m pytest test_client_e2e.py -q
