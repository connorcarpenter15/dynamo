# SPDX-License-Identifier: Apache-2.0
"""A minimal, dynamo-free OpenEngine servicer for CPU end-to-end client tests.

Implements just enough of the contract to exercise ``client.py`` over a real
``grpc.aio`` server: scripted Generate, plus metadata/lifecycle RPCs. Built on
the sidecar's generated stubs (no vllm, no dynamo runtime).
"""

from __future__ import annotations

import json
from typing import Optional

from _bootstrap import load

_client_mod, pb, pb_grpc = load()


class FakeOpenEngineServicer(pb_grpc.OpenEngineServicer):
    def __init__(
        self,
        *,
        role: int = pb.ENGINE_ROLE_AGGREGATED,
        token_script: Optional[list[list[int]]] = None,
        finish_reason: int = pb.FINISH_REASON_STOP,
        prefill_kv_params: Optional[dict] = None,
        transfer_backend: str = "nixl",
        healthy: bool = True,
    ) -> None:
        self.role = role
        self.token_script = token_script or [[10], [11, 12]]
        self.finish_reason = finish_reason
        self.prefill_kv_params = prefill_kv_params
        self.transfer_backend = transfer_backend
        self.healthy = healthy
        self.aborted: list[str] = []
        self.last_request: Optional[pb.GenerateRequest] = None

    async def Generate(self, request, context):
        self.last_request = request
        prompt_tokens = len(request.token_ids.ids)
        completion = 0

        if self.role == pb.ENGINE_ROLE_PREFILL and self.prefill_kv_params is not None:
            sess = pb.KvSessionRef(
                session_id=request.request_id,
                transfer_backend=self.transfer_backend,
                attributes={"kv_transfer_params": json.dumps(self.prefill_kv_params)},
            )
            yield pb.GenerateResponse(
                request_id=request.request_id,
                prefill_ready=pb.PrefillReady(kv_session=sess),
            )

        for ids in self.token_script:
            completion += len(ids)
            yield pb.GenerateResponse(
                request_id=request.request_id,
                token=pb.TokenOutput(token_ids=ids),
            )

        yield pb.GenerateResponse(
            request_id=request.request_id,
            finished=pb.GenerationFinished(reason=self.finish_reason),
            usage=pb.Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion,
                total_tokens=prompt_tokens + completion,
            ),
        )

    async def GetEngineInfo(self, request, context):
        return pb.EngineInfo(
            engine_name="vllm",
            engine_version="test",
            api_version="openengine.v1",
            role=self.role,
            parallelism=pb.ParallelismInfo(
                tensor_parallel_size=1,
                data_parallel_size=1,
                data_parallel_start_rank=0,
            ),
            kv_connector=pb.KvConnectorInfo(
                enabled=True, transfer_backend=self.transfer_backend
            ),
        )

    async def GetModelInfo(self, request, context):
        return pb.ModelInfo(
            model_id="m",
            served_model_name="m",
            max_context_length=2048,
            kv_block_size=16,
            total_kv_blocks=100,
            max_running_requests=8,
            max_batched_tokens=2048,
        )

    async def GetLoad(self, request, context):
        return pb.LoadInfo(running_requests=0, total_kv_blocks=100)

    async def Health(self, request, context):
        state = pb.HEALTH_STATE_READY if self.healthy else pb.HEALTH_STATE_NOT_READY
        return pb.HealthResponse(state=state)

    async def Abort(self, request, context):
        if request.abort_all:
            return pb.AbortResponse(status=pb.ABORT_STATUS_ABORTED)
        self.aborted.append(request.request_id)
        return pb.AbortResponse(status=pb.ABORT_STATUS_ABORTED)

    async def Drain(self, request, context):
        yield pb.DrainResponse(state=pb.DRAIN_STATE_STARTED, in_flight_requests=0)
        yield pb.DrainResponse(state=pb.DRAIN_STATE_COMPLETE, in_flight_requests=0)

    async def GetKvEventSources(self, request, context):
        return pb.GetKvEventSourcesResponse(
            sources=[
                pb.KvEventSource(
                    transport="zmq",
                    endpoint="tcp://127.0.0.1:5557",
                    data_parallel_rank=0,
                )
            ]
        )
