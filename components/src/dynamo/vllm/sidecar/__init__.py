# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo vLLM sidecar: an OpenEngine v1 gRPC client `LLMEngine`.

Drives a separate native ``vllm serve`` process; never imports vllm.
"""

from dynamo.vllm.sidecar.client import OpenEngineClient
from dynamo.vllm.sidecar.llm_engine import VllmSidecarLLMEngine

__all__ = ["OpenEngineClient", "VllmSidecarLLMEngine"]
