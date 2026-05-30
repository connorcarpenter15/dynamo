# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin async OpenEngine v1 gRPC client.

Intentionally **dynamo-free and vllm-free**: this module depends only on
``grpc`` and the generated stubs, so it can be exercised in CPU tests against a
real gRPC server backed by a fake servicer — no dynamo runtime, no GPU. All
higher-level translation (TypedDict <-> proto, error mapping, disagg handshake)
lives in :mod:`llm_engine`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import grpc

from ._openengine import openengine_pb2 as pb
from ._openengine import openengine_pb2_grpc as pb_grpc

logger = logging.getLogger(__name__)


class OpenEngineClient:
    """Async wrapper over the generated ``OpenEngineStub``.

    Lifecycle: :meth:`connect` opens the channel, :meth:`close` tears it down.
    Both are idempotent. Unary RPCs return the proto reply directly; streaming
    RPCs return the live ``grpc.aio`` call object so the caller can iterate it
    and ``cancel()`` it.
    """

    def __init__(self, endpoint: str, *, max_message_mb: int = 64) -> None:
        self._endpoint = endpoint
        self._max_message_bytes = max_message_mb * 1024 * 1024
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[pb_grpc.OpenEngineStub] = None

    @property
    def endpoint(self) -> str:
        return self._endpoint

    async def connect(self) -> None:
        if self._channel is not None:
            return
        options = [
            ("grpc.max_send_message_length", self._max_message_bytes),
            ("grpc.max_receive_message_length", self._max_message_bytes),
        ]
        self._channel = grpc.aio.insecure_channel(self._endpoint, options=options)
        self._stub = pb_grpc.OpenEngineStub(self._channel)

    async def wait_until_channel_ready(self, *, timeout: float) -> None:
        """Block until the underlying HTTP/2 channel connects, or raise
        ``asyncio.TimeoutError``. This only checks transport reachability;
        engine readiness is a separate ``Health`` poll."""
        await self.connect()
        assert self._channel is not None
        await asyncio.wait_for(self._channel.channel_ready(), timeout=timeout)

    async def close(self, *, grace: Optional[float] = None) -> None:
        if self._channel is not None:
            await self._channel.close(grace=grace)
        self._channel = None
        self._stub = None

    def _require_stub(self) -> pb_grpc.OpenEngineStub:
        if self._stub is None:
            raise RuntimeError("OpenEngineClient not connected; call connect()")
        return self._stub

    # -- Metadata / lifecycle RPCs ------------------------------------------

    async def get_engine_info(self) -> pb.EngineInfo:
        return await self._require_stub().GetEngineInfo(pb.GetEngineInfoRequest())

    async def get_model_info(self) -> pb.ModelInfo:
        return await self._require_stub().GetModelInfo(pb.GetModelInfoRequest())

    async def get_load(self, *, include_per_rank: bool = False) -> pb.LoadInfo:
        return await self._require_stub().GetLoad(
            pb.GetLoadRequest(include_per_rank=include_per_rank)
        )

    async def health(
        self, *, include_inference_probe: bool = False
    ) -> pb.HealthResponse:
        return await self._require_stub().Health(
            pb.HealthRequest(include_inference_probe=include_inference_probe)
        )

    async def get_kv_event_sources(
        self, *, data_parallel_ranks: Optional[list[int]] = None
    ) -> pb.GetKvEventSourcesResponse:
        return await self._require_stub().GetKvEventSources(
            pb.GetKvEventSourcesRequest(
                data_parallel_ranks=list(data_parallel_ranks or [])
            )
        )

    # -- Streaming / control RPCs -------------------------------------------

    def generate(self, request: pb.GenerateRequest):
        """Return the live unary-stream call. Iterate with ``async for``;
        cancel mid-stream with ``call.cancel()``."""
        return self._require_stub().Generate(request)

    async def abort(
        self, *, request_id: Optional[str] = None, abort_all: bool = False
    ) -> pb.AbortResponse:
        req = pb.AbortRequest(abort_all=abort_all)
        if request_id is not None:
            req.request_id = request_id
        return await self._require_stub().Abort(req)

    def drain(
        self,
        *,
        stop_accepting_new_requests: bool = True,
        deadline_ms: int = 0,
        abort_after_deadline: bool = False,
    ):
        """Return the live drain stream call."""
        return self._require_stub().Drain(
            pb.DrainRequest(
                stop_accepting_new_requests=stop_accepting_new_requests,
                deadline_ms=deadline_ms,
                abort_after_deadline=abort_after_deadline,
            )
        )
