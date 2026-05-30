#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving via the OpenEngine sidecar (1 GPU).
#
# Unlike agg.sh (which runs vLLM in-process inside the Dynamo worker), this
# launches three processes:
#   1. Dynamo frontend (HTTP ingress)
#   2. A native `vllm serve` engine exposing the OpenEngine gRPC server
#   3. The Dynamo vLLM sidecar worker, which talks to (2) over OpenEngine
#
# The sidecar never imports vllm; it only speaks the OpenEngine v1 contract.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

# Default model
MODEL="Qwen/Qwen3-0.6B"

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <name>       Specify model (default: $MODEL)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Additional options are passed through to the vllm serve engine."
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ---- Tunable (override via env vars) ----
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"
OPENENGINE_HOST="${OPENENGINE_HOST:-127.0.0.1}"
OPENENGINE_PORT="${OPENENGINE_PORT:-50051}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Sidecar Aggregated Serving (1 GPU)" "$MODEL" "$HTTP_PORT"

# 1. Dynamo frontend (HTTP ingress)
python -m dynamo.frontend &

# 2. Native vLLM engine with the OpenEngine gRPC server.
# --enforce-eager is for quick startup; drop it for production.
vllm serve "$MODEL" \
    --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    --openengine-host "$OPENENGINE_HOST" \
    --openengine-port "$OPENENGINE_PORT" \
    "${EXTRA_ARGS[@]}" &

# 3. Dynamo sidecar worker (no vllm import; OpenEngine client only).
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm.sidecar \
    --model "$MODEL" \
    --openengine-endpoint "${OPENENGINE_HOST}:${OPENENGINE_PORT}" &

# Exit on first process failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
