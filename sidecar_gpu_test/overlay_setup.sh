#!/usr/bin/env bash
# Overlay the vLLM fork's OpenEngine server module into the container's
# installed vLLM package. The module is self-contained and uses only stable
# vLLM APIs, so it works against the stock image without a rebuild.
set -euo pipefail

SRC="${1:?usage: overlay_setup.sh <openengine_overlay_dir>}"

VLLM_PKG="$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
echo "Installed vllm package: $VLLM_PKG"

DST="$VLLM_PKG/entrypoints/openengine"

echo "Overlaying $SRC -> $DST"
rm -rf "$DST"
cp -r "$SRC" "$DST"
# Drop tests + pycache from the overlay (not needed at runtime).
rm -rf "$DST/tests" "$DST/__pycache__" "$DST/_openengine/__pycache__"

echo "=== verify import ==="
python -c "from vllm.entrypoints.openengine.server import OpenEngineServer; print('OpenEngineServer import OK')"
python -c "from vllm.entrypoints.openengine._openengine import openengine_pb2_grpc; print('stubs import OK')"
echo "=== overlay done ==="
