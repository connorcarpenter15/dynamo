# SPDX-License-Identifier: Apache-2.0
"""CPU end-to-end test: sidecar OpenEngineClient <-> real grpc.aio server <->
fake servicer. Validates the wire path and client.py without dynamo or vllm."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import grpc  # noqa: E402
from _bootstrap import load  # noqa: E402
from fake_servicer import FakeOpenEngineServicer  # noqa: E402

client_mod, pb, pb_grpc = load()
OpenEngineClient = client_mod.OpenEngineClient


async def _serve(servicer):
    server = grpc.aio.server()
    pb_grpc.add_OpenEngineServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    return server, port


async def _with_client(servicer, fn):
    server, port = await _serve(servicer)
    client = OpenEngineClient(f"127.0.0.1:{port}")
    try:
        await client.wait_until_channel_ready(timeout=5.0)
        return await fn(client)
    finally:
        await client.close(grace=None)
        await server.stop(grace=None)


def _run(servicer, fn):
    return asyncio.run(_with_client(servicer, fn))


def test_channel_ready_and_health():
    async def fn(client):
        resp = await client.health()
        return resp.state

    assert _run(FakeOpenEngineServicer(), fn) == pb.HEALTH_STATE_READY


def test_engine_and_model_info():
    async def fn(client):
        info = await client.get_engine_info()
        minfo = await client.get_model_info()
        return info, minfo

    info, minfo = _run(FakeOpenEngineServicer(), fn)
    assert info.engine_name == "vllm"
    assert info.role == pb.ENGINE_ROLE_AGGREGATED
    assert info.kv_connector.transfer_backend == "nixl"
    assert minfo.served_model_name == "m"
    assert minfo.total_kv_blocks == 100
    assert minfo.kv_block_size == 16


def test_generate_happy_path():
    async def fn(client):
        req = pb.GenerateRequest(
            request_id="req-1",
            model="m",
            token_ids=pb.TokenIds(ids=[1, 2, 3]),
            sampling=pb.SamplingParams(max_tokens=8),
        )
        events = []
        async for resp in client.generate(req):
            events.append(resp)
        return events

    servicer = FakeOpenEngineServicer(token_script=[[10], [11, 12], [13]])
    events = _run(servicer, fn)

    tokens = [
        list(e.token.token_ids)
        for e in events
        if e.WhichOneof("event") == "token"
    ]
    assert tokens == [[10], [11, 12], [13]]

    finished = [e for e in events if e.WhichOneof("event") == "finished"]
    assert len(finished) == 1
    assert finished[0].finished.reason == pb.FINISH_REASON_STOP
    assert finished[0].usage.prompt_tokens == 3
    assert finished[0].usage.completion_tokens == 4
    assert finished[0].usage.total_tokens == 7


def test_generate_prefill_emits_kv_session():
    kv_params = {"remote_block_ids": [1, 2], "remote_engine_id": "eng-7"}

    async def fn(client):
        req = pb.GenerateRequest(
            request_id="req-1",
            model="m",
            token_ids=pb.TokenIds(ids=[1, 2, 3]),
        )
        events = []
        async for resp in client.generate(req):
            events.append(resp)
        return events

    servicer = FakeOpenEngineServicer(
        role=pb.ENGINE_ROLE_PREFILL,
        token_script=[[99]],
        prefill_kv_params=kv_params,
    )
    events = _run(servicer, fn)
    ready = [e for e in events if e.WhichOneof("event") == "prefill_ready"]
    assert len(ready) == 1
    sess = ready[0].prefill_ready.kv_session
    import json

    assert json.loads(sess.attributes["kv_transfer_params"]) == kv_params


def test_abort_and_drain_and_kv_sources():
    async def fn(client):
        abort_resp = await client.abort(request_id="req-1")
        drain_states = [s.state async for s in client.drain()]
        kv = await client.get_kv_event_sources(data_parallel_ranks=[0])
        return abort_resp, drain_states, kv

    servicer = FakeOpenEngineServicer()
    abort_resp, drain_states, kv = _run(servicer, fn)
    assert abort_resp.status == pb.ABORT_STATUS_ABORTED
    assert servicer.aborted == ["req-1"]
    assert drain_states[0] == pb.DRAIN_STATE_STARTED
    assert drain_states[-1] == pb.DRAIN_STATE_COMPLETE
    assert len(kv.sources) == 1
    assert kv.sources[0].transport == "zmq"


def test_decode_kv_session_roundtrip_over_wire():
    # Mirror how the sidecar packs a decode request: kv_transfer_params from
    # the prefill peer encoded into KvSessionRef.attributes as JSON.
    import json

    kv_params = {"remote_block_ids": [4, 5, 6], "remote_engine_id": "eng-1"}

    async def fn(client):
        req = pb.GenerateRequest(
            request_id="req-d",
            model="m",
            token_ids=pb.TokenIds(ids=[1, 2, 3]),
            kv_session=pb.KvSessionRef(
                session_id="req-d",
                transfer_backend="nixl",
                attributes={"kv_transfer_params": json.dumps(kv_params)},
            ),
        )
        async for _ in client.generate(req):
            pass

    servicer = FakeOpenEngineServicer(role=pb.ENGINE_ROLE_DECODE)
    _run(servicer, fn)
    got = servicer.last_request.kv_session
    assert got.session_id == "req-d"
    assert got.transfer_backend == "nixl"
    assert json.loads(got.attributes["kv_transfer_params"]) == kv_params


def test_generate_cancel_midstream():
    async def fn(client):
        req = pb.GenerateRequest(
            request_id="req-1", model="m", token_ids=pb.TokenIds(ids=[1])
        )
        call = client.generate(req)
        first = await call.read()
        call.cancel()
        return first

    # Slow script so we can cancel before completion.
    servicer = FakeOpenEngineServicer(token_script=[[10]] * 50)
    first = _run(servicer, fn)
    assert first.WhichOneof("event") == "token"
