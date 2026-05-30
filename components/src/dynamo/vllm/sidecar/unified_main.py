# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified entry point for the vLLM OpenEngine sidecar backend.

Usage:
    python -m dynamo.vllm.sidecar <sidecar args>

The sidecar talks to a separate native ``vllm serve`` process over OpenEngine
v1 gRPC; it never imports vllm. See sidecar/CLAUDE.md for the design.
"""

from dynamo.common.backend.run import run
from dynamo.vllm.sidecar.llm_engine import VllmSidecarLLMEngine


def main():
    run(VllmSidecarLLMEngine)


if __name__ == "__main__":
    main()
