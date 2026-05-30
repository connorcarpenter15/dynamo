#!/usr/bin/env python3
"""Direct OpenEngine gRPC smoke client: GetEngineInfo + a short Generate."""

from __future__ import annotations

import argparse
import asyncio

import grpc
from vllm.entrypoints.openengine._openengine import openengine_pb2 as pb
from vllm.entrypoints.openengine._openengine import openengine_pb2_grpc as pb_grpc


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default="127.0.0.1:50051")
    p.add_argument("--prompt", default="The capital of France is")
    args = p.parse_args()

    async with grpc.aio.insecure_channel(args.endpoint) as ch:
        stub = pb_grpc.OpenEngineStub(ch)

        info = await stub.GetEngineInfo(pb.GetEngineInfoRequest())
        print("=== GetEngineInfo ===")
        print(info)

        model = await stub.GetModelInfo(pb.GetModelInfoRequest())
        print("=== GetModelInfo ===")
        print(model)

        health = await stub.Health(pb.HealthRequest())
        print("=== Health ===")
        print(health)

        print("=== Generate ===")
        req = pb.GenerateRequest(
            request_id="smoke-1",
            prompt=args.prompt,
        )
        req.sampling.max_tokens = 16
        req.sampling.temperature = 0.0
        text = ""
        async for resp in stub.Generate(req):
            which = resp.WhichOneof("event")
            if which == "token":
                text += resp.token.text
                print("TOKEN:", repr(resp.token.text))
            elif which == "prefill_ready":
                print("PREFILL_READY:", resp.prefill_ready)
            elif which == "finished":
                print("FINISHED:", resp.finished)
            elif which == "error":
                print("ERROR:", resp.error)
        print("=== full text ===")
        print(text)


if __name__ == "__main__":
    asyncio.run(main())
