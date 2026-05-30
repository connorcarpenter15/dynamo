#!/usr/bin/env bash
# 1P1D disaggregated sidecar e2e. Runs INSIDE the dynamo vLLM container.
# Two vLLM OpenEngine engines colocated on a single GPU (reduced memory util):
#   - decode  engine (kv_consumer) on :50051
#   - prefill engine (kv_producer) on :50052
# plus a decode sidecar, a prefill sidecar, and the Dynamo frontend. KV moves
# prefill->decode over NIXL; the sidecars relay KvSessionRef<->kv_transfer_params
# across the Dynamo boundary. Asserts a chat completion streams with a finish.
set -uo pipefail

GPUTEST=/tmp/custom_deps/vllm/sidecar_gpu_test
SRC=/tmp/custom_deps/vllm/components/src
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
LOG=/tmp/connorc/disagg_e2e
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

echo "=== overlay openengine module into installed vllm ==="
bash "$GPUTEST/overlay_setup.sh" "$GPUTEST/openengine_overlay" >"$LOG/overlay.log" 2>&1
if ! grep -q "overlay done" "$LOG/overlay.log"; then
  echo "FAIL: overlay setup failed"; tail -30 "$LOG/overlay.log"; exit 1
fi
echo "overlay OK"

echo "=== start etcd ==="
ETCD_BIN=$(command -v etcd || echo /usr/local/bin/etcd/etcd)
"$ETCD_BIN" --data-dir /tmp/connorc/etcd-data-disagg \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 \
  >"$LOG/etcd.log" 2>&1 &
PIDS+=($!)

echo "=== start nats ==="
NATS_BIN=$(command -v nats-server || echo /usr/bin/nats-server)
"$NATS_BIN" -js >"$LOG/nats.log" 2>&1 &
PIDS+=($!)
sleep 3

echo "=== start DECODE engine (kv_consumer, :50051) ==="
CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT=20098 \
  python "$GPUTEST/engine_launcher.py" \
  --model "$MODEL" --openengine-port 50051 --enforce-eager \
  --gpu-memory-utilization 0.4 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' \
  >"$LOG/decode_engine.log" 2>&1 &
DECODE_PID=$!; PIDS+=($DECODE_PID)

echo "=== start PREFILL engine (kv_producer, :50052) ==="
CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
  python "$GPUTEST/engine_launcher.py" \
  --model "$MODEL" --openengine-port 50052 --enforce-eager \
  --gpu-memory-utilization 0.4 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
  >"$LOG/prefill_engine.log" 2>&1 &
PREFILL_PID=$!; PIDS+=($PREFILL_PID)

echo "=== wait for engines ready ==="
wait_ready "$LOG/decode_engine.log" "decode" "$DECODE_PID" || exit 1
wait_ready "$LOG/prefill_engine.log" "prefill" "$PREFILL_PID" || exit 1

echo "=== start dynamo frontend ==="
python -m dynamo.frontend --http-port 8000 >"$LOG/frontend.log" 2>&1 &
PIDS+=($!)
sleep 5

echo "=== start DECODE sidecar (backend) ==="
DYN_SYSTEM_PORT=8081 python -m dynamo.vllm.sidecar \
  --model "$MODEL" --disaggregation-mode decode \
  --openengine-endpoint 127.0.0.1:50051 \
  >"$LOG/decode_sidecar.log" 2>&1 &
PIDS+=($!)

echo "=== start PREFILL sidecar ==="
DYN_SYSTEM_PORT=8082 python -m dynamo.vllm.sidecar \
  --model "$MODEL" --disaggregation-mode prefill \
  --openengine-endpoint 127.0.0.1:50052 \
  >"$LOG/prefill_sidecar.log" 2>&1 &
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
  echo "--- decode_sidecar.log ---"; tail -40 "$LOG/decode_sidecar.log"
  echo "--- prefill_sidecar.log ---"; tail -40 "$LOG/prefill_sidecar.log"
  echo "--- frontend.log ---"; tail -40 "$LOG/frontend.log"
  exit 1
fi

echo "=== send chat completion (disagg) ==="
curl -sS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one short sentence.\"}],\"max_tokens\":32,\"stream\":false}" \
  >"$LOG/chat.json" 2>"$LOG/chat.err"
echo "--- chat response ---"; cat "$LOG/chat.json"; echo

if grep -q '"finish_reason"' "$LOG/chat.json" && grep -q '"content"' "$LOG/chat.json"; then
  echo "RESULT: DISAGG_E2E_PASS"
else
  echo "RESULT: DISAGG_E2E_FAIL"
  echo "--- decode_sidecar.log tail ---"; tail -30 "$LOG/decode_sidecar.log"
  echo "--- prefill_sidecar.log tail ---"; tail -30 "$LOG/prefill_sidecar.log"
  echo "--- decode_engine.log tail ---"; tail -30 "$LOG/decode_engine.log"
  echo "--- prefill_engine.log tail ---"; tail -30 "$LOG/prefill_engine.log"
fi

echo "=== streaming chat completion (disagg) ==="
curl -sS -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count to three.\"}],\"max_tokens\":32,\"stream\":true}" \
  >"$LOG/chat_stream.txt" 2>&1
echo "--- stream tail ---"; tail -6 "$LOG/chat_stream.txt"
if grep -q 'finish_reason' "$LOG/chat_stream.txt"; then
  echo "RESULT: DISAGG_STREAM_PASS"
else
  echo "RESULT: DISAGG_STREAM_FAIL"
fi

echo "=== DONE ==="
