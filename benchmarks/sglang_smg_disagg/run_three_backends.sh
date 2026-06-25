#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/tmp/connorc_sglang_smg_disagg_bench}"
VENV="${VENV:-$ROOT/venv}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
ISL="${ISL:-128}"
OSL="${OSL:-64}"
REQUESTS="${REQUESTS:-100}"
CONCURRENCY="${CONCURRENCY:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
PORT_OFFSET="${PORT_OFFSET:-0}"
SMG_CONNECTIONS="${SMG_CONNECTIONS:-8}"
DISAGG_BOOTSTRAP_PORT="${DYN_DISAGG_BOOTSTRAP_PORT:-12345}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export HOME="${HOME:-$ROOT/home}"
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/pip_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/uv_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export SGLANG_FORCE_SHUTDOWN="${SGLANG_FORCE_SHUTDOWN:-1}"
export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-false}"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/.nixl_cu12.mesonpy.libs:${LD_LIBRARY_PATH:-}"

if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

mkdir -p "$ROOT/logs" "$ROOT/artifacts" "$ROOT/file_kv"

cleanup_pids=()
cleanup() {
    set +e
    for pid in "${cleanup_pids[@]:-}"; do
        kill -TERM -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    done
    sleep 5
    for pid in "${cleanup_pids[@]:-}"; do
        kill -KILL -- "-$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    cleanup_pids=()
}
trap cleanup EXIT

wait_http() {
    local url="$1"
    local log_file="$2"
    for _ in $(seq 1 360); do
        if "$PYTHON_BIN" - "$url" "$MODEL" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

url = sys.argv[1]
model = sys.argv[2]
with urllib.request.urlopen(url, timeout=2) as r:
    body = r.read().decode("utf-8", "ignore")
    sys.exit(0 if r.status == 200 and model in body else 1)
PY
        then
            return 0
        fi
        if grep -E "Traceback|ERROR|Error|Exception|No module named|unrecognized arguments|failed|CUDA out of memory" "$log_file" >/dev/null 2>&1; then
            tail -120 "$log_file"
        fi
        sleep 5
    done
    echo "Timed out waiting for $url"
    tail -240 "$log_file" || true
    return 1
}

run_aiperf() {
    local name="$1"
    local port="$2"
    local artifact_dir="$ROOT/artifacts/$name"
    mkdir -p "$artifact_dir"
    aiperf profile \
        --artifact-dir "$artifact_dir" \
        --model "$MODEL" \
        --endpoint-type chat \
        --endpoint /v1/chat/completions \
        --streaming \
        --url "http://127.0.0.1:$port" \
        --synthetic-input-tokens-mean "$ISL" \
        --synthetic-input-tokens-stddev 0 \
        --output-tokens-mean "$OSL" \
        --output-tokens-stddev 0 \
        --extra-inputs max_tokens:"$OSL" \
        --extra-inputs min_tokens:"$OSL" \
        --extra-inputs ignore_eos:true \
        --extra-inputs repetition_penalty:1.0 \
        --extra-inputs temperature:0.0 \
        --concurrency "$CONCURRENCY" \
        --request-count "$REQUESTS" \
        --warmup-request-count 5 \
        --workers-max "$CONCURRENCY" \
        --record-processors 32 \
        --ui none \
        >"$ROOT/logs/${name}_aiperf.log" 2>&1
}

sglang_common_args() {
    printf '%s\0' \
        --model-path "$MODEL" \
        --served-model-name "$MODEL" \
        --host 0.0.0.0 \
        --tp 1 \
        --trust-remote-code \
        --context-length "$MAX_MODEL_LEN" \
        --max-running-requests "$MAX_NUM_SEQS" \
        --disaggregation-transfer-backend nixl \
        --disaggregation-bootstrap-port "$DISAGG_BOOTSTRAP_PORT"
}

start_frontend() {
    local server_log="$1"
    setsid "$PYTHON_BIN" -m dynamo.frontend \
        --discovery-backend file \
        >>"$server_log" 2>&1 &
    cleanup_pids+=("$!")
}

run_smg_disagg() {
    local server_log="$1"
    local prefill_http="$((30001 + PORT_OFFSET))"
    local decode_http="$((30002 + PORT_OFFSET))"
    local prefill_smg="$((40001 + PORT_OFFSET))"
    local decode_smg="$((40002 + PORT_OFFSET))"
    local prefill_system="$((8081 + PORT_OFFSET))"
    local decode_system="$((8082 + PORT_OFFSET))"

    local common=()
    while IFS= read -r -d '' arg; do common+=("$arg"); done < <(sglang_common_args)

    CUDA_VISIBLE_DEVICES=0 SGLANG_GRPC_PORT="$prefill_smg" \
        setsid "$PYTHON_BIN" -m sglang.launch_server \
        --grpc-mode \
        "${common[@]}" \
        --port "$prefill_http" \
        --disaggregation-mode prefill \
        --disable-cuda-graph \
        >>"$server_log" 2>&1 &
    cleanup_pids+=("$!")

    DYN_SYSTEM_PORT="$prefill_system" \
        setsid dynamo-sglang-smg-sidecar \
        --smg-endpoint "127.0.0.1:${prefill_smg}" \
        --smg-connections "$SMG_CONNECTIONS" \
        >>"$server_log" 2>&1 &
    cleanup_pids+=("$!")

    CUDA_VISIBLE_DEVICES=1 SGLANG_GRPC_PORT="$decode_smg" \
        setsid "$PYTHON_BIN" -m sglang.launch_server \
        --grpc-mode \
        "${common[@]}" \
        --port "$decode_http" \
        --disaggregation-mode decode \
        --disable-cuda-graph \
        >>"$server_log" 2>&1 &
    cleanup_pids+=("$!")

    DYN_SYSTEM_PORT="$decode_system" \
        setsid dynamo-sglang-smg-sidecar \
        --smg-endpoint "127.0.0.1:${decode_smg}" \
        --smg-connections "$SMG_CONNECTIONS" \
        >>"$server_log" 2>&1 &
    cleanup_pids+=("$!")
}

run_python_disagg() {
    local module="$1"
    local server_log="$2"
    local prefill_http="$((30001 + PORT_OFFSET))"
    local decode_http="$((30002 + PORT_OFFSET))"
    local prefill_system="$((8081 + PORT_OFFSET))"
    local decode_system="$((8082 + PORT_OFFSET))"

    local common=()
    while IFS= read -r -d '' arg; do common+=("$arg"); done < <(sglang_common_args)

    DYN_SYSTEM_PORT="$prefill_system" CUDA_VISIBLE_DEVICES=0 \
        setsid "$PYTHON_BIN" -m "$module" \
        "${common[@]}" \
        --port "$prefill_http" \
        --page-size 16 \
        --disaggregation-mode prefill \
        --disable-piecewise-cuda-graph \
        >>"$server_log" 2>&1 &
    cleanup_pids+=("$!")

    DYN_SYSTEM_PORT="$decode_system" CUDA_VISIBLE_DEVICES=1 \
        setsid "$PYTHON_BIN" -m "$module" \
        "${common[@]}" \
        --port "$decode_http" \
        --page-size 16 \
        --disaggregation-mode decode \
        --disable-piecewise-cuda-graph \
        >>"$server_log" 2>&1 &
    cleanup_pids+=("$!")
}

run_backend() {
    local name="$1"
    local port="$2"
    local server_log="$ROOT/logs/${name}_server.log"
    local aiperf_log="$ROOT/logs/${name}_aiperf.log"
    : >"$server_log"
    : >"$aiperf_log"
    cleanup

    echo "=== starting $name disagg on port $port ==="
    export DYN_HTTP_PORT="$port"
    export DYN_FILE_KV="$ROOT/file_kv/$name"
    export DYN_DISCOVERY_BACKEND=file
    export DYN_REQUEST_PLANE=tcp
    rm -rf "$DYN_FILE_KV"
    mkdir -p "$DYN_FILE_KV"

    start_frontend "$server_log"

    case "$name" in
        smg)
            run_smg_disagg "$server_log"
            ;;
        legacy)
            run_python_disagg dynamo.sglang "$server_log"
            ;;
        unified)
            run_python_disagg dynamo.sglang.unified_main "$server_log"
            ;;
        *)
            echo "unknown backend: $name" >&2
            exit 2
            ;;
    esac

    wait_http "http://127.0.0.1:${port}/v1/models" "$server_log"
    echo "=== benchmarking $name disagg ==="
    run_aiperf "$name" "$port"
    echo "=== completed $name disagg ==="
    cleanup
}

"$PYTHON_BIN" - <<'PY'
import importlib.metadata as md

for name in ["ai-dynamo", "ai-dynamo-runtime", "sglang", "smg-grpc-servicer", "smg-grpc-proto", "aiperf"]:
    try:
        print(f"{name}={md.version(name)}")
    except Exception as exc:
        print(f"{name}=MISSING {exc}")
PY

BACKENDS="${BACKENDS:-smg legacy unified}"
for backend in $BACKENDS; do
    case "$backend" in
        smg)
            run_backend smg "$((8000 + PORT_OFFSET))"
            ;;
        legacy)
            run_backend legacy "$((8001 + PORT_OFFSET))"
            ;;
        unified)
            run_backend unified "$((8002 + PORT_OFFSET))"
            ;;
        *)
            echo "unknown backend in BACKENDS: $backend" >&2
            exit 2
            ;;
    esac
done

echo "all disagg backends complete"
