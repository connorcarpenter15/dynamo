#!/usr/bin/env python3
"""Standalone OpenEngine server launcher for GPU e2e testing.

Builds a vLLM ``AsyncLLM`` from the container's installed vLLM and serves the
OpenEngine v1 gRPC contract via ``OpenEngineServer`` (overlaid fork module).
This avoids patching ``vllm serve`` (cli_args/api_server) into the container's
installed vllm; the openengine module is self-contained and uses only stable
vLLM APIs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("engine_launcher")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--openengine-host", default="127.0.0.1")
    p.add_argument("--openengine-port", type=int, default=50051)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--kv-transfer-config", default="")
    p.add_argument("--kv-events-config", default="")
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.entrypoints.openengine.server import OpenEngineServer

    engine_kwargs = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    if args.kv_transfer_config:
        from vllm.config import KVTransferConfig

        engine_kwargs["kv_transfer_config"] = KVTransferConfig(
            **json.loads(args.kv_transfer_config)
        )
    if args.kv_events_config:
        from vllm.config import KVEventsConfig

        engine_kwargs["kv_events_config"] = KVEventsConfig(
            **json.loads(args.kv_events_config)
        )

    engine_args = AsyncEngineArgs(**engine_kwargs)
    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER
    )

    logger.info("Building AsyncLLM for %s ...", args.model)
    engine = AsyncLLM.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=UsageContext.OPENAI_API_SERVER,
    )

    server = OpenEngineServer(
        engine,
        vllm_config,
        host=args.openengine_host,
        port=args.openengine_port,
    )
    await server.start()
    logger.info(
        "OpenEngine server ready on %s:%s (role=%s)",
        args.openengine_host,
        server.port,
        server.config.role,
    )
    try:
        await server.wait_for_termination()
    finally:
        await server.shutdown(grace=5.0)


if __name__ == "__main__":
    asyncio.run(main())
