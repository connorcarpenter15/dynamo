#!/usr/bin/env bash
# Migration e2e: two aggregated sidecar replicas behind the Dynamo frontend.
# Fire a long stream, then KILL the engine serving it mid-stream. The sidecar's
# gRPC stream to the dead engine breaks -> it raises a typed EngineShutdown ->
# the frontend rebuilds the request from accumulated tokens and reroutes to the
# surviving replica (token-replay migration). Asserts the stream still finishes
# with a finish_reason (no error surfaced to the client). This is existing
# Dynamo behavior; the test verifies the sidecar does not break it.
# Runs INSIDE the dynamo vLLM container.
set -uo pipefail

GPUTEST=/tmp/custom_deps/vllm/sidecar_gpu_test
SRC=/tmp/custom_deps/vllm/components/src
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
LOG=/tmp/connorc/migration_e2e
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

wait_ready() {
  # $1=logfile $2=label $3=pid
  for i in $(seq 1 120); do
    if grep -q "OpenEngine server ready" "$1" 2>/dev/null; then
      echo "$2 engine ready after ${i}0s-ish"; return 0
    fi
    if ! kill -0 "$3" 2>/dev/null; then
      echo "FAIL: $2 engine process died"; tail -40 "$1"; return 1
    fi
    sleep 3
  done
  echo "FAIL: $2 engine never became ready"; tail -40 "$1"; return 1
}

load_running() {
  python "$GPUTEST/load_probe.py" --endpoint "$1" 2>/dev/null \
    | sed -n 's/^running_requests=//p'
}

echo "=== overlay openengine module into installed vllm ==="
bash "$GPUTEST/overlay_setup.sh" "$GPUTEST/openengine_overlay" >"$LOG/overlay.log" 2>&1
if ! grep -q "overlay done" "$LOG/overlay.log"; then
  echo "FAIL: overlay setup failed"; tail -30 "$LOG/overlay.log"; exit 1
fi
echo "overlay OK"

echo "=== start etcd ==="
ETCD_BIN=$(command -v etcd || echo /usr/local/bin/etcd/etcd)
"$ETCD_BIN" --data-dir /tmp/connorc/etcd-data-migration \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 \
  >"$LOG/etcd.log" 2>&1 &
PIDS+=($!)

echo "=== start nats ==="
NATS_BIN=$(command -v nats-server || echo /usr/bin/nats-server)
"$NATS_BIN" -js >"$LOG/nats.log" 2>&1 &
PIDS+=($!)
sleep 3

echo "=== start engine A (:50051) ==="
CUDA_VISIBLE_DEVICES=0 python "$GPUTEST/engine_launcher.py" \
  --model "$MODEL" --openengine-port 50051 --enforce-eager \
  --gpu-memory-utilization 0.4 --max-num-seqs 4 \
  >"$LOG/engine_a.log" 2>&1 &
ENGINE_A_PID=$!; PIDS+=($ENGINE_A_PID)

echo "=== start engine B (:50052) ==="
CUDA_VISIBLE_DEVICES=0 python "$GPUTEST/engine_launcher.py" \
  --model "$MODEL" --openengine-port 50052 --enforce-eager \
  --gpu-memory-utilization 0.4 --max-num-seqs 4 \
  >"$LOG/engine_b.log" 2>&1 &
ENGINE_B_PID=$!; PIDS+=($ENGINE_B_PID)

echo "=== wait for engines ready ==="
wait_ready "$LOG/engine_a.log" "A" "$ENGINE_A_PID" || exit 1
wait_ready "$LOG/engine_b.log" "B" "$ENGINE_B_PID" || exit 1

echo "=== start dynamo frontend (migration enabled) ==="
python -m dynamo.frontend --http-port 8000 --migration-limit 3 \
  >"$LOG/frontend.log" 2>&1 &
PIDS+=($!)
sleep 5

echo "=== start sidecar A (-> :50051) ==="
DYN_SYSTEM_PORT=8081 python -m dynamo.vllm.sidecar \
  --model "$MODEL" --openengine-endpoint 127.0.0.1:50051 \
  >"$LOG/sidecar_a.log" 2>&1 &
PIDS+=($!)

echo "=== start sidecar B (-> :50052) ==="
DYN_SYSTEM_PORT=8082 python -m dynamo.vllm.sidecar \
  --model "$MODEL" --openengine-endpoint 127.0.0.1:50052 \
  >"$LOG/sidecar_b.log" 2>&1 &
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
  echo "--- sidecar_a.log ---"; tail -40 "$LOG/sidecar_a.log"
  echo "--- sidecar_b.log ---"; tail -40 "$LOG/sidecar_b.log"
  echo "--- frontend.log ---"; tail -40 "$LOG/frontend.log"
  exit 1
fi
# Give discovery a moment so BOTH replicas are routable before we start.
sleep 5

echo "=== fire long stream ==="
curl -sS -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a very long detailed essay about the history of computing.\"}],\"max_tokens\":1500,\"stream\":true}" \
  >"$LOG/stream.txt" 2>&1 &
CURL_PID=$!
PIDS+=($CURL_PID)

echo "=== detect which engine is serving, then kill it ==="
sleep 5
RA=$(load_running 127.0.0.1:50051); RB=$(load_running 127.0.0.1:50052)
echo "running A=$RA B=$RB"
VICTIM=""
if [ "${RA:-0}" != "0" ] && [ -n "${RA:-}" ]; then
  VICTIM="$ENGINE_A_PID"; echo "killing engine A ($ENGINE_A_PID, :50051)"
elif [ "${RB:-0}" != "0" ] && [ -n "${RB:-}" ]; then
  VICTIM="$ENGINE_B_PID"; echo "killing engine B ($ENGINE_B_PID, :50052)"
else
  echo "WARN: could not detect serving engine; killing A by default"
  VICTIM="$ENGINE_A_PID"
fi
kill -9 "$VICTIM" 2>/dev/null || true

echo "=== wait for stream to finish ==="
WAITED=0
while kill -0 "$CURL_PID" 2>/dev/null; do
  sleep 1; WAITED=$((WAITED+1))
  if [ "$WAITED" -ge 60 ]; then echo "WARN: stream did not end in 60s"; break; fi
done
echo "stream ended after ~${WAITED}s"

echo "--- stream tail ---"; tail -8 "$LOG/stream.txt"
if grep -q 'finish_reason' "$LOG/stream.txt" \
   && ! grep -qi 'Internal Server Error\|"code":5' "$LOG/stream.txt"; then
  echo "RESULT: MIGRATION_PASS"
else
  echo "RESULT: MIGRATION_FAIL"
  echo "--- frontend.log tail ---"; tail -40 "$LOG/frontend.log"
  echo "--- sidecar_a.log tail ---"; tail -20 "$LOG/sidecar_a.log"
  echo "--- sidecar_b.log tail ---"; tail -20 "$LOG/sidecar_b.log"
fi

echo "=== DONE ==="
