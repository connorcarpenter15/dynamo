#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/tmp/connorc_sglang_smg_disagg_bench}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/sglang_disagg_h100_mini_sweep_$(date +%Y-%m-%d_%H%M%S)}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
REQUESTS="${REQUESTS:-512}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

mkdir -p "$OUT_DIR"

run_point() {
    local name="$1"
    local isl="$2"
    local osl="$3"
    local concurrency="$4"
    local point="${name}_isl${isl}_osl${osl}_c${concurrency}"
    local point_dir="$OUT_DIR/$point"

    rm -rf "$ROOT/artifacts" "$ROOT/logs" "$ROOT/file_kv"
    mkdir -p "$ROOT/artifacts" "$ROOT/logs" "$point_dir"

    cat >"$point_dir/metadata.json" <<JSON
{
  "point": "$point",
  "model": "$MODEL",
  "isl": $isl,
  "osl": $osl,
  "concurrency": $concurrency,
  "requests": $REQUESTS,
  "max_model_len": $MAX_MODEL_LEN,
  "max_num_seqs": $MAX_NUM_SEQS,
  "mode": "disaggregated",
  "backends": ["smg", "legacy", "unified"]
}
JSON

    echo "=== point $point ==="
    if MODEL="$MODEL" \
        ISL="$isl" \
        OSL="$osl" \
        REQUESTS="$REQUESTS" \
        CONCURRENCY="$concurrency" \
        MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        MAX_NUM_SEQS="$MAX_NUM_SEQS" \
        BACKENDS="smg legacy unified" \
        "$SCRIPT_DIR/run_three_backends.sh" \
        >"$point_dir/run.log" 2>&1; then
        cp -a "$ROOT/artifacts" "$point_dir/artifacts"
        cp -a "$ROOT/logs" "$point_dir/logs"
        echo "=== completed $point ==="
    else
        local status=$?
        cp -a "$ROOT/artifacts" "$point_dir/artifacts" 2>/dev/null || true
        cp -a "$ROOT/logs" "$point_dir/logs" 2>/dev/null || true
        echo "=== failed $point status=$status ===" >&2
        exit "$status"
    fi
}

run_point short_decode 128 64 4
run_point short_decode 128 64 16
run_point short_decode 128 64 64
run_point long_decode_control_plane 32 512 16
run_point long_decode_control_plane 32 512 64
run_point long_decode_control_plane 32 512 256

python "$SCRIPT_DIR/summarize_h100_mini_sweep.py" "$OUT_DIR"
echo "SGLang disagg H100 mini sweep complete: $OUT_DIR"
