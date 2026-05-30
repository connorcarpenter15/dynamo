#!/usr/bin/env bash
# Self-contained aggregated sidecar e2e driver. Runs INSIDE the dynamo vLLM
# container (single srun, single localhost network). Brings up etcd + NATS,
# the OpenEngine vLLM engine, the Dynamo frontend, and the sidecar worker,
# then sends a chat completion and asserts tokens stream with finish stop.
set -uo pipefail

GPUTEST=/tmp/custom_deps/vllm/sidecar_gpu_test
SRC=/tmp/custom_deps/vllm/components/src
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
LOG=/tmp/connorc/agg_e2e
mkdir -p "$LOG"

export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/connorc/hf
export VLLM_NO_USAGE_STATS=1
export DYN_LOG=info
# Home is a tiny 5GB NFS quota (often 100% full); keep every cache off it.
# Dynamo's MDC cache root is hardcoded to $HOME/.cache/dynamo/mdc (model_card.rs),
# so redirect HOME itself, not just the framework cache vars.
export HOME=/tmp/connorc/dynhome
export TRITON_CACHE_DIR=/tmp/connorc/cache/triton
export XDG_CACHE_HOME=/tmp/connorc/cache/xdg
export VLLM_CACHE_ROOT=/tmp/connorc/cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/tmp/connorc/cache/inductor
export TORCH_HOME=/tmp/connorc/cache/torch
mkdir -p "$HOME" "$HF_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_HOME"

echo "=== env ==="
echo "python: $(command -v python)"
echo "PYTHONPATH=$PYTHONPATH"

cleanup() {
  echo "=== cleanup ==="
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  sleep 2
  for p in "${PIDS[@]:-}"; do kill -9 "$p" 2>/dev/null || true; done
}
trap cleanup EXIT
PIDS=()

echo "=== overlay openengine module into installed vllm ==="
bash "$GPUTEST/overlay_setup.sh" "$GPUTEST/openengine_overlay" >"$LOG/overlay.log" 2>&1
if ! grep -q "overlay done" "$LOG/overlay.log"; then
  echo "FAIL: overlay setup failed"; tail -30 "$LOG/overlay.log"; exit 1
fi
echo "overlay OK"

echo "=== start etcd ==="
ETCD_BIN=$(command -v etcd || echo /usr/local/bin/etcd/etcd)
"$ETCD_BIN" --data-dir /tmp/connorc/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 \
  >"$LOG/etcd.log" 2>&1 &
PIDS+=($!)

echo "=== start nats ==="
NATS_BIN=$(command -v nats-server || echo /usr/bin/nats-server)
"$NATS_BIN" -js >"$LOG/nats.log" 2>&1 &
PIDS+=($!)
sleep 3

echo "=== start OpenEngine vLLM engine ==="
python "$GPUTEST/engine_launcher.py" \
  --model "$MODEL" --openengine-port 50051 --enforce-eager \
  >"$LOG/engine.log" 2>&1 &
PIDS+=($!)

echo "=== wait for engine ready ==="
for i in $(seq 1 120); do
  if grep -q "OpenEngine server ready" "$LOG/engine.log" 2>/dev/null; then
    echo "engine ready after ${i}0s-ish"; break
  fi
  if ! kill -0 "${PIDS[-1]}" 2>/dev/null; then
    echo "FAIL: engine process died"; tail -40 "$LOG/engine.log"; exit 1
  fi
  sleep 3
done
if ! grep -q "OpenEngine server ready" "$LOG/engine.log"; then
  echo "FAIL: engine never became ready"; tail -40 "$LOG/engine.log"; exit 1
fi

echo "=== direct gRPC smoke against engine ==="
python "$GPUTEST/smoke_client.py" --endpoint 127.0.0.1:50051 \
  >"$LOG/smoke.log" 2>&1 || true
tail -20 "$LOG/smoke.log"

echo "=== start dynamo frontend ==="
python -m dynamo.frontend --http-port 8000 >"$LOG/frontend.log" 2>&1 &
PIDS+=($!)
sleep 5

echo "=== start sidecar worker ==="
python -m dynamo.vllm.sidecar \
  --model "$MODEL" --openengine-endpoint 127.0.0.1:50051 \
  >"$LOG/sidecar.log" 2>&1 &
PIDS+=($!)

echo "=== wait for model to register with frontend ==="
REGISTERED=0
for i in $(seq 1 40); do
  if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "$MODEL"; then
    REGISTERED=1; echo "model registered after ~$((i*3))s"; break
  fi
  sleep 3
done
if [ "$REGISTERED" != "1" ]; then
  echo "FAIL: model never registered"
  echo "--- sidecar.log ---"; tail -40 "$LOG/sidecar.log"
  echo "--- frontend.log ---"; tail -40 "$LOG/frontend.log"
  exit 1
fi

echo "=== send chat completion ==="
curl -sS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one short sentence.\"}],\"max_tokens\":32,\"stream\":false}" \
  >"$LOG/chat.json" 2>"$LOG/chat.err"
echo "--- chat response ---"
cat "$LOG/chat.json"
echo

if grep -q '"finish_reason"' "$LOG/chat.json" && grep -q '"content"' "$LOG/chat.json"; then
  echo "RESULT: AGG_E2E_PASS"
else
  echo "RESULT: AGG_E2E_FAIL"
  echo "--- sidecar.log tail ---"; tail -30 "$LOG/sidecar.log"
  echo "--- engine.log tail ---"; tail -30 "$LOG/engine.log"
fi

echo "=== streaming chat completion ==="
curl -sS -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count to three.\"}],\"max_tokens\":32,\"stream\":true}" \
  >"$LOG/chat_stream.txt" 2>&1
echo "--- stream tail ---"
tail -8 "$LOG/chat_stream.txt"
if grep -q 'finish_reason' "$LOG/chat_stream.txt"; then
  echo "RESULT: AGG_STREAM_PASS"
else
  echo "RESULT: AGG_STREAM_FAIL"
fi

echo "=== DONE ==="
