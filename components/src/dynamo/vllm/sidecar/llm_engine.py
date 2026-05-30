# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``VllmSidecarLLMEngine`` — drives a native ``vllm serve`` process over the
OpenEngine v1 gRPC contract.

Sibling of the in-process ``dynamo.vllm.llm_engine:VllmLLMEngine``. It mirrors
that engine method-for-method but sources every value from an OpenEngine RPC
instead of an in-process vLLM object. **This module must never import vllm.**
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any, Optional

import grpc

from dynamo._core import Context
from dynamo.common.backend import telemetry
from dynamo.common.backend.disagg import (
    enforce_prefill_max_tokens,
    require_prefill_result,
)
from dynamo.common.backend.engine import (
    EngineConfig,
    GenerateChunk,
    GenerateRequest,
    LLMEngine,
)
from dynamo.common.backend.health_check import build_health_check_payload
from dynamo.common.backend.publisher import KvEventSource, ZmqSource
from dynamo.common.backend.worker import WorkerConfig
from dynamo.common.constants import DisaggregationMode
from dynamo.llm import ModelInput
from dynamo.llm.exceptions import (
    Cancelled,
    CannotConnect,
    EngineShutdown,
    InvalidArgument,
    Unknown,
)

from ._openengine import openengine_pb2 as pb
from .args import parse_args
from .client import OpenEngineClient

logger = logging.getLogger(__name__)

# Wire contract with the vLLM servicer: the KvSessionRef attribute map carries
# vLLM's `kv_transfer_params` JSON blob under this key. Kept in sync with
# `vllm/vllm/entrypoints/openengine/servicer.py::KV_TRANSFER_PARAMS_ATTR`.
_KV_TRANSFER_PARAMS_ATTR = "kv_transfer_params"

_MODE_TO_ROLE = {
    DisaggregationMode.AGGREGATED: pb.ENGINE_ROLE_AGGREGATED,
    DisaggregationMode.PREFILL: pb.ENGINE_ROLE_PREFILL,
    DisaggregationMode.DECODE: pb.ENGINE_ROLE_DECODE,
}

_FINISH_REASON_TO_STR = {
    pb.FINISH_REASON_STOP: "stop",
    pb.FINISH_REASON_LENGTH: "length",
    pb.FINISH_REASON_CANCELLED: "cancelled",
    pb.FINISH_REASON_ERROR: "error",
}

_ERROR_CODE_TO_EXC = {
    pb.ERROR_CODE_INVALID_ARGUMENT: InvalidArgument,
    pb.ERROR_CODE_CANCELLED: Cancelled,
    pb.ERROR_CODE_DRAINING: EngineShutdown,
}

# Map sampling_options keys -> proto SamplingParams scalar fields.
_SAMPLING_SCALAR_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "seed",
)


class VllmSidecarLLMEngine(LLMEngine):
    def __init__(
        self,
        *,
        openengine_endpoint: str,
        model: str,
        served_model_name: str,
        disaggregation_mode: DisaggregationMode,
        connect_timeout: float = 30.0,
        health_poll_interval: float = 1.0,
        health_deadline: float = 300.0,
    ) -> None:
        self._model = model
        self._served_model_name = served_model_name
        self.disaggregation_mode = disaggregation_mode
        self._connect_timeout = connect_timeout
        self._health_poll_interval = health_poll_interval
        self._health_deadline = health_deadline
        self._client: Optional[OpenEngineClient] = OpenEngineClient(openengine_endpoint)
        self._transfer_backend: str = "nixl"
        self._dp_range: tuple[int, int] = (0, 1)

    @classmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple["VllmSidecarLLMEngine", WorkerConfig]:
        config = parse_args(argv)
        engine = cls(
            openengine_endpoint=config.openengine_endpoint,
            model=config.model,
            served_model_name=config.served_model_name or config.model,
            disaggregation_mode=config.disaggregation_mode,
            connect_timeout=config.connect_timeout,
            health_poll_interval=config.health_poll_interval,
            health_deadline=config.health_deadline,
        )
        worker_config = WorkerConfig.from_runtime_config(
            config,
            model_name=config.model,
            served_model_name=config.served_model_name,
            model_input=ModelInput.Tokens,
        )
        return engine, worker_config

    async def start(self, worker_id: int) -> EngineConfig:
        del worker_id  # vLLM's NixlConnector handles its own per-worker IDs.
        assert self._client is not None
        await self._client.connect()
        try:
            await self._client.wait_until_channel_ready(timeout=self._connect_timeout)
        except (asyncio.TimeoutError, grpc.aio.AioRpcError) as e:
            raise CannotConnect(
                f"sidecar could not reach OpenEngine server at "
                f"{self._client.endpoint}: {e}"
            ) from e

        await self._poll_health_until_ready()

        info = await self._client.get_engine_info()
        minfo = await self._client.get_model_info()
        self._validate_role(info.role)
        self._validate_model(minfo)

        if info.kv_connector.transfer_backend:
            self._transfer_backend = info.kv_connector.transfer_backend
        dp_start = info.parallelism.data_parallel_start_rank
        dp_size = max(1, info.parallelism.data_parallel_size)
        self._dp_range = (dp_start, dp_size)

        logger.info(
            "vLLM sidecar connected: engine=%s version=%s role=%s model=%s",
            info.engine_name,
            info.engine_version,
            info.role,
            minfo.model_id,
        )

        return EngineConfig(
            model=self._model,
            served_model_name=self._served_model_name,
            context_length=minfo.max_context_length or None,
            kv_cache_block_size=minfo.kv_block_size or None,
            total_kv_blocks=minfo.total_kv_blocks or None,
            max_num_seqs=minfo.max_running_requests or None,
            max_num_batched_tokens=minfo.max_batched_tokens or None,
            data_parallel_start_rank=dp_start,
            data_parallel_size=dp_size,
        )

    async def _poll_health_until_ready(self) -> None:
        assert self._client is not None
        deadline = time.monotonic() + self._health_deadline
        last_state = None
        while True:
            try:
                resp = await self._client.health()
                last_state = resp.state
                if resp.state == pb.HEALTH_STATE_READY:
                    return
            except grpc.aio.AioRpcError as e:
                last_state = f"rpc-error:{e.code()}"
            if time.monotonic() >= deadline:
                raise CannotConnect(
                    f"OpenEngine server at {self._client.endpoint} did not become "
                    f"READY within {self._health_deadline}s (last state {last_state})"
                )
            await asyncio.sleep(self._health_poll_interval)

    def _validate_role(self, role: int) -> None:
        expected = _MODE_TO_ROLE[self.disaggregation_mode]
        if role != expected:
            raise InvalidArgument(
                f"engine role {role} does not match sidecar "
                f"disaggregation_mode {self.disaggregation_mode.value} "
                f"(expected role {expected})"
            )

    def _validate_model(self, minfo: pb.ModelInfo) -> None:
        engine_name = minfo.served_model_name or minfo.model_id
        if (
            minfo.served_model_name
            and self._served_model_name
            and minfo.served_model_name != self._served_model_name
        ):
            raise InvalidArgument(
                f"engine serves '{engine_name}' but sidecar configured for "
                f"'{self._served_model_name}'; names must match"
            )

    def _build_sampling(self, request: GenerateRequest) -> pb.SamplingParams:
        sampling_options = request.get("sampling_options", {}) or {}
        stop_conditions = request.get("stop_conditions", {}) or {}
        kwargs: dict[str, Any] = {}
        for key in _SAMPLING_SCALAR_KEYS:
            value = sampling_options.get(key)
            if value is not None:
                kwargs[key] = value
        max_tokens = stop_conditions.get("max_tokens")
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return pb.SamplingParams(**kwargs)

    def _build_stop(self, request: GenerateRequest) -> list[pb.StopCondition]:
        stop_conditions = request.get("stop_conditions", {}) or {}
        out: list[pb.StopCondition] = []
        for text in stop_conditions.get("stop", []) or []:
            out.append(pb.StopCondition(stop_text=text))
        for token_id in stop_conditions.get("stop_token_ids", []) or []:
            out.append(pb.StopCondition(stop_token_id=token_id))
        return out

    def _build_decode_kv_session(
        self, request: GenerateRequest, request_id: str
    ) -> pb.KvSessionRef:
        prefill_result = require_prefill_result(request, self.disaggregation_mode)
        kv_params = prefill_result.get("disaggregated_params", {}).get(
            _KV_TRANSFER_PARAMS_ATTR
        )
        if kv_params is None:
            raise InvalidArgument(
                "decode worker received prefill_result without "
                "kv_transfer_params; the prefill peer must populate it for "
                "vLLM's NixlConnector to pull KV blocks"
            )
        return pb.KvSessionRef(
            session_id=request_id,
            transfer_backend=self._transfer_backend,
            attributes={_KV_TRANSFER_PARAMS_ATTR: json.dumps(kv_params)},
        )

    def _build_proto_request(
        self, request: GenerateRequest, request_id: str
    ) -> pb.GenerateRequest:
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            enforce_prefill_max_tokens(request)

        req = pb.GenerateRequest(
            request_id=request_id,
            model=self._model,
            token_ids=pb.TokenIds(ids=list(request.get("token_ids", []))),
            sampling=self._build_sampling(request),
            stop=self._build_stop(request),
            stream=True,
        )
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            req.kv_session.CopyFrom(
                self._build_decode_kv_session(request, request_id)
            )
        return req

    def _decode_kv_session(self, kv_session: pb.KvSessionRef) -> Optional[dict]:
        blob = kv_session.attributes.get(_KV_TRANSFER_PARAMS_ATTR)
        if not blob:
            return None
        return json.loads(blob)

    def _map_error(self, error: pb.EngineError) -> Exception:
        exc_cls = _ERROR_CODE_TO_EXC.get(error.code, Unknown)
        return exc_cls(error.message or "OpenEngine generation error")

    def _map_grpc_error(self, e: grpc.aio.AioRpcError) -> Exception:
        code = e.code()
        if code == grpc.StatusCode.CANCELLED:
            return Cancelled(e.details() or "request cancelled")
        if code in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.ABORTED):
            # Channel/engine loss → typed shutdown so the frontend can migrate.
            return EngineShutdown(e.details() or "OpenEngine server unavailable")
        return Unknown(f"OpenEngine RPC failed ({code}): {e.details()}")

    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        if self._client is None:
            raise RuntimeError("Engine not initialized")

        request_id = context.id()
        req = self._build_proto_request(request, request_id)
        is_prefill = self.disaggregation_mode == DisaggregationMode.PREFILL

        for key, value in telemetry.engine_trace_kwargs(context).get(
            "trace_headers", {}
        ).items():
            req.metadata[key] = value

        call = self._client.generate(req)
        kv_session: Optional[pb.KvSessionRef] = None
        # Prefill emits token / prefill_ready / finished as separate OpenEngine
        # events, but Dynamo's PrefillRouter reads disaggregated_params off the
        # FIRST response chunk. Buffer the (single, capped) prefill token and
        # fold it into one terminal chunk so the first chunk carries the handle
        # — matching the in-process VllmLLMEngine which yields a combined chunk.
        prefill_token_ids: list[int] = []
        try:
            async for resp in call:
                event = resp.WhichOneof("event")
                if event == "token":
                    if is_prefill:
                        prefill_token_ids.extend(resp.token.token_ids)
                    else:
                        yield {"index": 0, "token_ids": list(resp.token.token_ids)}
                elif event == "prefill_ready":
                    kv_session = resp.prefill_ready.kv_session
                elif event == "finished":
                    chunk: GenerateChunk = {
                        "index": 0,
                        "token_ids": prefill_token_ids if is_prefill else [],
                        "finish_reason": _FINISH_REASON_TO_STR.get(
                            resp.finished.reason, "stop"
                        ),
                        "completion_usage": {
                            "prompt_tokens": resp.usage.prompt_tokens,
                            "completion_tokens": resp.usage.completion_tokens,
                            "total_tokens": resp.usage.total_tokens,
                        },
                    }
                    if is_prefill and kv_session is not None:
                        kv_params = self._decode_kv_session(kv_session)
                        if kv_params is not None:
                            chunk["disaggregated_params"] = {
                                _KV_TRANSFER_PARAMS_ATTR: kv_params
                            }
                    yield chunk
                elif event == "error":
                    raise self._map_error(resp.error)
        except asyncio.CancelledError:
            call.cancel()
            raise
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.CANCELLED:
                logger.debug("Generate stream cancelled for %s", request_id)
                return
            raise self._map_grpc_error(e) from e

    async def abort(self, context: Context) -> None:
        if self._client is None:
            return
        request_id = context.id()
        if request_id is None:
            return
        try:
            await self._client.abort(request_id=request_id)
            logger.debug("Aborted request %s", request_id)
        except grpc.aio.AioRpcError as e:
            logger.debug("Abort RPC for %s failed (ignored): %s", request_id, e)

    async def drain(self) -> None:
        if self._client is None:
            return
        try:
            call = self._client.drain(stop_accepting_new_requests=True)
            async for resp in call:
                if resp.state in (pb.DRAIN_STATE_COMPLETE, pb.DRAIN_STATE_FAILED):
                    break
        except Exception as e:
            logger.warning("vLLM sidecar drain failed (ignored): %s", e)

    async def cleanup(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            finally:
                self._client = None
                logger.info("vLLM sidecar shutdown")

    async def kv_event_sources(self) -> list[KvEventSource]:
        if self._client is None:
            return []
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            return []
        dp_start, dp_size = self._dp_range
        resp = await self._client.get_kv_event_sources(
            data_parallel_ranks=list(range(dp_start, dp_start + dp_size))
        )
        sources: list[KvEventSource] = []
        for s in resp.sources:
            if s.transport != "zmq":
                continue
            sources.append(
                ZmqSource(
                    endpoint=s.endpoint,
                    topic=s.topic,
                    dp_rank=s.data_parallel_rank,
                )
            )
        return sources

    async def health_check_payload(self) -> Optional[dict[str, Any]]:
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            logger.warning(
                "DECODE sidecar: health-check canary disabled — NixlConnector "
                "has no verified local-only bypass. Readiness relies on traffic."
            )
            return None
        # The sidecar has no tokenizer; fall back to BOS=1 (overridable via
        # --health-check-payload / DYN_HEALTH_CHECK_PAYLOAD).
        return build_health_check_payload(bos_token_id=1)
