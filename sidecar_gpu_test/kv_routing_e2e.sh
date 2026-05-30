#!/usr/bin/env bash
# KV-aware routing e2e: two aggregated sidecar replicas behind a KV-routing
# Dynamo frontend. Each native vLLM engine has prefix caching on and a ZMQ KV
# event publisher (distinct ports). The sidecar advertises that publisher via
# OpenEngine GetKvEventSources -> the Dynamo worker subscribes to the engine's
# ZMQ socket and republishes BlockStored/BlockRemoved onto NATS -> the
# frontend KV router applies them to its indexer.
#
# The test warms a long shared prefix, waits for KV-event propagation, then
# re-sends the same prefix and scrapes the frontend's /metrics. PASS requires:
#   * dynamo_component_kv_cache_events_applied > 0  -- unambiguous proof the
#     sidecar advertised its ZMQ source AND Dynamo consumed the engine events
#     (this counter is fed ONLY by the event subscriber, never by the router's
#     own routing-decision tracking).
#   * dynamo_component_router_kv_hit_rate_sum > 0   -- proof the router used the
#     resulting overlap when selecting a worker.
# Runs INSIDE the dynamo vLLM container.
set -uo pipefail

GPUTEST=/tmp/custom_deps/vllm/sidecar_gpu_test
SRC=/tmp/custom_deps/vllm/components/src
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
LOG=/tmp/connorc/kv_routing_e2e
mkdir -p "$LOG"

export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/tmp/connorc/hf
export VLLM_NO_USAGE_STATS=1
export DYN_LOG=info
# REQUIRED for KV routing: vLLM seeds its block-hash chain (NONE_HASH) from
# PYTHONHASHSEED (random per-process if unset). Both engines AND the frontend
# router must share a fixed seed so block hashes are uniform across workers and
# reproducible by the router's indexer. Every Dynamo vLLM router launch script
# exports this; engine_launcher.py runs native vLLM directly so we set it here.
export PYTHONHASHSEED=0
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

# Two vLLM engines share ONE GPU. vLLM's gpu_memory_utilization is a device-wide
# ceiling, not a private budget: the engine computes available_kv = total*util -
# (device memory already used by ALL processes). If a prior run's engines haven't
# fully released GPU memory yet, the new engine A profiles against a dirty device
# and gets "No available memory for the cache blocks". Pre-flight: kill any of our
# leftovers and wait until the GPU is actually free before launching.
echo "=== preflight: clear leftovers, wait for free GPU ==="
pkill -9 -f 'engine_launcher.py' 2>/dev/null || true
pkill -9 -f 'dynamo.vllm.sidecar' 2>/dev/null || true
pkill -9 -f 'dynamo.frontend' 2>/dev/null || true
sleep 3
for i in $(seq 1 30); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "${FREE:-0}" -gt 70000 ]; then
    echo "GPU free=${FREE} MiB -- ok to launch"; break
  fi
  echo "waiting for GPU to free up (free=${FREE} MiB)"; sleep 3
done

echo "=== overlay openengine module into installed vllm ==="
bash "$GPUTEST/overlay_setup.sh" "$GPUTEST/openengine_overlay" >"$LOG/overlay.log" 2>&1
if ! grep -q "overlay done" "$LOG/overlay.log"; then
  echo "FAIL: overlay setup failed"; tail -30 "$LOG/overlay.log"; exit 1
fi
echo "overlay OK"

echo "=== start etcd ==="
ETCD_BIN=$(command -v etcd || echo /usr/local/bin/etcd/etcd)
"$ETCD_BIN" --data-dir /tmp/connorc/etcd-data-kvrouting \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 \
  >"$LOG/etcd.log" 2>&1 &
PIDS+=($!)

echo "=== start nats ==="
NATS_BIN=$(command -v nats-server || echo /usr/bin/nats-server)
"$NATS_BIN" -js >"$LOG/nats.log" 2>&1 &
PIDS+=($!)
sleep 3

# Each engine: prefix caching (default on) + a distinct ZMQ KV-event publisher.
# Both engines share ONE GPU and must be launched SEQUENTIALLY, not concurrently.
# vLLM v1's gpu_memory_utilization check requires FREE memory at startup to be
# >= util * total_device_mem (see v1/worker/utils.py:request_memory). On an 80 GiB
# H100 that means each engine demands util*79 GiB *free* when it boots. So the two
# engines must use a MODEST, EQUAL util that fits twice: A (util 0.4) reserves
# ~32 GiB, leaving ~47 GiB free; B (util 0.4) then needs only ~32 GiB free, which
# fits. (A higher util on B fails: e.g. 0.8 demands 63 GiB free but only ~47
# remains.) Staggering — start A, wait until fully up, then start B — avoids the
# concurrent-profiling race where both grab the device at once. Peak ~64/80 GiB.
echo "=== start engine A (:50051, kv-events tcp://*:5557, util 0.4) ==="
CUDA_VISIBLE_DEVICES=0 python "$GPUTEST/engine_launcher.py" \
  --model "$MODEL" --openengine-port 50051 --enforce-eager \
  --gpu-memory-utilization 0.4 --max-num-seqs 8 \
  --kv-events-config '{"enable_kv_cache_events": true, "endpoint": "tcp://*:5557"}' \
  >"$LOG/engine_a.log" 2>&1 &
ENGINE_A_PID=$!; PIDS+=($ENGINE_A_PID)
echo "=== wait for engine A ready ==="
wait_ready "$LOG/engine_a.log" "A" "$ENGINE_A_PID" || exit 1

echo "=== start engine B (:50052, kv-events tcp://*:5657, util 0.4) ==="
CUDA_VISIBLE_DEVICES=0 python "$GPUTEST/engine_launcher.py" \
  --model "$MODEL" --openengine-port 50052 --enforce-eager \
  --gpu-memory-utilization 0.4 --max-num-seqs 8 \
  --kv-events-config '{"enable_kv_cache_events": true, "endpoint": "tcp://*:5657"}' \
  >"$LOG/engine_b.log" 2>&1 &
ENGINE_B_PID=$!; PIDS+=($ENGINE_B_PID)
echo "=== wait for engine B ready ==="
wait_ready "$LOG/engine_b.log" "B" "$ENGINE_B_PID" || exit 1

echo "=== start dynamo frontend (KV routing; /metrics served on :8000) ==="
# The frontend deliberately ignores DYN_SYSTEM_PORT (no system metrics server);
# its Prometheus /metrics is exposed by the LLM HTTP service on --http-port and
# includes the DRT/component registries (router + KV-indexer metrics).
python -m dynamo.frontend --http-port 8000 --router-mode kv \
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
# Give discovery + KV-event subscription a moment to wire up on both replicas.
sleep 8

# Build a long shared prefix so the cached prefix spans many KV blocks
# (block_size 16 -> a few hundred tokens guarantees a strong overlap signal).
PREFIX=$(python - <<'PYEOF'
para = ("The history of computing spans many centuries of human ingenuity, "
        "from the abacus and mechanical calculators through Babbage's analytical "
        "engine, Turing's theoretical machines, the vacuum-tube mainframes of the "
        "1940s, transistors, integrated circuits, microprocessors, personal "
        "computers, the internet, mobile devices, and modern accelerated AI. ")
print((para * 8).strip())
PYEOF
)

req_body() {  # $1 = unique suffix
  python - "$PREFIX" "$1" <<'PYEOF'
import json, sys
prefix, suffix = sys.argv[1], sys.argv[2]
content = prefix + " " + suffix
print(json.dumps({
    "model": "%MODEL%",
    "messages": [{"role": "user", "content": content}],
    "max_tokens": 20,
    "temperature": 0.0,
    "stream": False,
}))
PYEOF
}
# inject model name (kept out of the heredoc to avoid quoting headaches)
send() {  # $1 = unique suffix
  local body
  body=$(req_body "$1" | sed "s#%MODEL%#$MODEL#")
  curl -sS http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' -d "$body" >/dev/null 2>&1
}

echo "=== warm the shared prefix (request 1) ==="
send "warm-up question one please answer briefly"
echo "warmed; waiting for KV events to propagate to the frontend indexer"
sleep 12

echo "=== re-send the same prefix several times (routing should hit warm worker) ==="
for n in 1 2 3 4 5 6; do
  send "repeat number $n answer briefly"
done
sleep 5

echo "=== scrape frontend /metrics (:8000) ==="
curl -sS http://localhost:8000/metrics >"$LOG/metrics.txt" 2>&1
echo "--- relevant metric lines ---"
grep -E 'kv_cache_events_applied|router_kv_hit_rate|router_shared_cache_hit_rate|router_requests_total' \
  "$LOG/metrics.txt" | grep -v '^#' | head -40

python - "$LOG/metrics.txt" <<'PYEOF'
import sys
txt = open(sys.argv[1]).read()

def total(substr, label_must=None):
    s, found = 0.0, False
    for line in txt.splitlines():
        if line.startswith('#') or substr not in line:
            continue
        if label_must and label_must not in line:
            continue
        parts = line.rsplit(' ', 1)
        if len(parts) != 2:
            continue
        try:
            s += float(parts[1]); found = True
        except ValueError:
            pass
    return s, found

applied, f_applied = total('kv_cache_events_applied')
applied_stored, _ = total('kv_cache_events_applied', 'event_type="stored"')
hit_sum, f_hit = total('router_kv_hit_rate_sum')
hit_cnt, _ = total('router_kv_hit_rate_count')

print(f"kv_cache_events_applied total={applied} stored={applied_stored} present={f_applied}")
print(f"router_kv_hit_rate sum={hit_sum} count={hit_cnt} present={f_hit}")

events_ok = applied > 0
routing_ok = hit_sum > 0
if events_ok and routing_ok:
    print("RESULT: KV_ROUTING_PASS")
elif events_ok:
    print("RESULT: KV_ROUTING_PARTIAL (events consumed but no nonzero kv_hit_rate)")
else:
    print("RESULT: KV_ROUTING_FAIL")
PYEOF

echo "--- frontend KV-event log lines ---"
grep -iE 'kv.?event|indexer|subscrib|zmq|block.?stored' "$LOG/frontend.log" | tail -20
echo "--- sidecar_a KV-event log lines ---"
grep -iE 'kv.?event|source|zmq|subscrib' "$LOG/sidecar_a.log" | tail -15

echo "=== DONE ==="
