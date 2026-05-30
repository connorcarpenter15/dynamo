# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI args for the vLLM OpenEngine sidecar worker.

**vllm-free on purpose.** Unlike ``dynamo.vllm.args`` (which parses
``AsyncEngineArgs``), the sidecar never imports vLLM — the engine runs in a
separate ``vllm serve`` process. This module only parses Dynamo runtime args
plus the OpenEngine endpoint + sidecar knobs, and resolves the
component/endpoint by disaggregation role exactly like the in-process path.
"""

from __future__ import annotations

import argparse
import logging
from typing import Optional

from dynamo.common.configuration.groups.runtime_args import (
    DynamoRuntimeArgGroup,
    DynamoRuntimeConfig,
)
from dynamo.common.configuration.utils import add_argument
from dynamo.common.constants import DisaggregationMode
from dynamo.common.utils.runtime import parse_endpoint

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_OPENENGINE_ENDPOINT = "127.0.0.1:50051"


class SidecarConfig(DynamoRuntimeConfig):
    """Runtime config for the sidecar worker.

    Inherits all Dynamo runtime fields and adds the OpenEngine endpoint, the
    served model identity (which the sidecar validates against the engine's
    ``GetEngineInfo``/``GetModelInfo``), the disaggregation role, and the
    connection/health-poll knobs.
    """

    component: str = "backend"
    model: str = DEFAULT_MODEL
    served_model_name: Optional[str] = None
    disaggregation_mode: DisaggregationMode = DisaggregationMode.AGGREGATED

    openengine_endpoint: str = DEFAULT_OPENENGINE_ENDPOINT
    connect_timeout: float = 30.0
    health_poll_interval: float = 1.0
    health_deadline: float = 300.0

    def validate(self) -> None:
        DynamoRuntimeConfig.validate(self)
        if isinstance(self.disaggregation_mode, str):
            self.disaggregation_mode = DisaggregationMode(self.disaggregation_mode)


class _SidecarArgGroup:
    """Sidecar-specific CLI args (OpenEngine endpoint, model, role, timeouts)."""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("vLLM Sidecar (OpenEngine) Options")
        add_argument(
            g,
            flag_name="--model",
            env_var="DYN_VLLM_SIDECAR_MODEL",
            default=DEFAULT_MODEL,
            help="Model id the engine serves. Must match the engine's "
            "GetModelInfo; the sidecar refuses to register on mismatch.",
        )
        add_argument(
            g,
            flag_name="--served-model-name",
            env_var="DYN_VLLM_SIDECAR_SERVED_MODEL_NAME",
            default=None,
            help="Public model name. Defaults to --model.",
        )
        add_argument(
            g,
            flag_name="--openengine-endpoint",
            env_var="DYN_VLLM_SIDECAR_OPENENGINE_ENDPOINT",
            default=DEFAULT_OPENENGINE_ENDPOINT,
            help="host:port of the vLLM OpenEngine gRPC server "
            "(`vllm serve --openengine-port ...`).",
        )
        add_argument(
            g,
            flag_name="--disaggregation-mode",
            env_var="DYN_VLLM_SIDECAR_DISAGGREGATION_MODE",
            default=DisaggregationMode.AGGREGATED.value,
            choices=[
                DisaggregationMode.AGGREGATED.value,
                DisaggregationMode.PREFILL.value,
                DisaggregationMode.DECODE.value,
            ],
            help="Worker role: agg, prefill, or decode.",
        )
        add_argument(
            g,
            flag_name="--connect-timeout",
            env_var="DYN_VLLM_SIDECAR_CONNECT_TIMEOUT",
            default=30.0,
            arg_type=float,
            help="Seconds to wait for the gRPC channel to connect.",
        )
        add_argument(
            g,
            flag_name="--health-poll-interval",
            env_var="DYN_VLLM_SIDECAR_HEALTH_POLL_INTERVAL",
            default=1.0,
            arg_type=float,
            help="Seconds between Health polls while waiting for READY.",
        )
        add_argument(
            g,
            flag_name="--health-deadline",
            env_var="DYN_VLLM_SIDECAR_HEALTH_DEADLINE",
            default=300.0,
            arg_type=float,
            help="Total seconds to wait for the engine to report READY.",
        )


def _resolve_component_endpoint(config: SidecarConfig, user_endpoint: Optional[str]):
    """Mirror the in-process vLLM resolution: role decides component/endpoint,
    an explicit ``--endpoint`` overrides namespace/component/endpoint."""
    if config.disaggregation_mode == DisaggregationMode.PREFILL:
        config.component = "prefill"
        config.endpoint = "generate"
    else:
        config.component = "backend"
        config.endpoint = "generate"

    if user_endpoint is not None:
        ns, comp, ep = parse_endpoint(user_endpoint)
        config.namespace = ns
        config.component = comp
        config.endpoint = ep


def parse_args(argv: list[str] | None = None) -> SidecarConfig:
    parser = argparse.ArgumentParser(
        description="Dynamo vLLM OpenEngine sidecar worker configuration",
        formatter_class=argparse.RawTextHelpFormatter,
        allow_abbrev=False,
    )
    DynamoRuntimeArgGroup().add_arguments(parser)
    _SidecarArgGroup().add_arguments(parser)

    args = parser.parse_args(argv)
    config = SidecarConfig.from_cli_args(args)

    user_endpoint = config.endpoint
    config.validate()

    if not config.served_model_name:
        config.served_model_name = config.model

    _resolve_component_endpoint(config, user_endpoint)
    return config
