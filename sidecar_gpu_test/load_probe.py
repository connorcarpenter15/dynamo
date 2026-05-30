#!/usr/bin/env python3
"""Print the engine's current GetLoad.running_requests (one shot).

Used by the cancellation e2e to assert that aborting a stream releases the
engine's running slot (running_requests returns to 0).
"""

from __future__ import annotations

import argparse
import asyncio

import grpc
from vllm.entrypoints.openengine._openengine import openengine_pb2 as pb
from vllm.entrypoints.openengine._openengine import openengine_pb2_grpc as pb_grpc


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default="127.0.0.1:50051")
    args = p.parse_args()
    async with grpc.aio.insecure_channel(args.endpoint) as ch:
        stub = pb_grpc.OpenEngineStub(ch)
        load = await stub.GetLoad(pb.GetLoadRequest())
        print(f"running_requests={load.running_requests}")
        print(f"queued_requests={load.queued_requests}")


if __name__ == "__main__":
    asyncio.run(main())
