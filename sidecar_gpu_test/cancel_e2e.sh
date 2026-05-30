#!/usr/bin/env bash
# Cancellation e2e: client disconnects mid-stream -> Dynamo aborts -> sidecar
# OpenEngine Abort -> engine releases the slot. Asserts the engine's in-flight
# count (GetLoad.running_requests, backed by the servicer _active set) returns
# to 0 after the disconnect, and that a fresh request then completes normally.
# Runs INSIDE the dynamo vLLM container (single srun, single localhost net).
set -uo pipefail

GPUTEST=/tmp/custom_deps/vllm/sidecar_gpu_test
SRC=/tmp/custom_deps/vllm/components/src
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
LOG=/tmp/connorc/cancel_e2e
mkdir -p "$LOG"

export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/connorc/hf
export VLLM_NO_USAGE_STATS=1
export DYN_LOG=info
export HOME=/tmp/connorc/dynhome
export TRITON_CACHE_DIR=/tmp/connorc/cache/triton
export XDG_CACHE_HOME=/tmp/connorc/cache/xdg
export VLLM_CACHE_ROOT=/tmp/connorc/cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/tmp/connorc/cache/inductor
export TORCH_HOME=/tmp/connorc/cache/torch
mkdir -p "$HOME" "$HF_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_HOME"

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
"$ETCD_BIN" --data-dir /tmp/connorc/etcd-data-cancel \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 \
  >"$LOG/etcd.log" 2>&1 &
PIDS+=($!)

echo "=== start nats ==="
NATS_BIN=$(command -v nats-server || echo /usr/bin/nats-server)
"$NATS_BIN" -js >"$LOG/nats.log" 2>&1 &
PIDS+=($!)
sleep 3

echo "=== start OpenEngine vLLM engine (max-num-seqs=1) ==="
# Single slot makes the cancellation differential sharp: if abort fails, the
# long stream keeps the only slot and the follow-up request would stall.
python "$GPUTEST/engine_launcher.py" \
  --model "$MODEL" --openengine-port 50051 --enforce-eager --max-num-seqs 1 \
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

load_running() {
  python "$GPUTEST/load_probe.py" --endpoint 127.0.0.1:50051 2>/dev/null \
    | sed -n 's/^running_requests=//p'
}

echo "=== baseline load ==="
echo "running_requests(before)=$(load_running)"

echo "=== fire long stream, then disconnect client mid-stream ==="
curl -sS -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a very long detailed essay about the history of computing.\"}],\"max_tokens\":2000,\"stream\":true}" \
  >"$LOG/long_stream.txt" 2>&1 &
CURL_PID=$!
sleep 4
echo "running_requests(mid-stream)=$(load_running)"
echo "--- killing client (disconnect) ---"
kill -9 "$CURL_PID" 2>/dev/null || true

echo "=== poll for engine slot release ==="
RELEASED=0
for i in $(seq 1 20); do
  r=$(load_running)
  if [ "$r" = "0" ]; then
    RELEASED=1; echo "running_requests back to 0 after ~$((i))s"; break
  fi
  sleep 1
done
if [ "$RELEASED" = "1" ]; then
  echo "RESULT: CANCEL_ABORT_PASS"
else
  echo "RESULT: CANCEL_ABORT_FAIL (running_requests=$(load_running))"
  echo "--- sidecar.log tail ---"; tail -30 "$LOG/sidecar.log"
  echo "--- engine.log tail ---"; tail -30 "$LOG/engine.log"
fi

echo "=== follow-up request must complete on the freed slot ==="
START=$(date +%s)
curl -sS --max-time 30 http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":8,\"stream\":false}" \
  >"$LOG/followup.json" 2>"$LOG/followup.err"
ELAPSED=$(( $(date +%s) - START ))
echo "follow-up took ${ELAPSED}s"
cat "$LOG/followup.json"; echo
if grep -q '"finish_reason"' "$LOG/followup.json" && grep -q '"content"' "$LOG/followup.json"; then
  echo "RESULT: CANCEL_RECOVER_PASS"
else
  echo "RESULT: CANCEL_RECOVER_FAIL"
fi

echo "=== DONE ==="
