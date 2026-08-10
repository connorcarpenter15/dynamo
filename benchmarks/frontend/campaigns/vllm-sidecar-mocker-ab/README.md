<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CPU-only vLLM sidecar AgentX A/B campaign

This campaign compares the direct Python vLLM Mocker LiveEngine boundary with `dynamo-vllm-sidecar` plus the CPU-only `dynamo-vllm-mocker-server`. It measures native request-plane/gRPC overhead under the pinned InferenceX AgentX traffic shape; it does not run vLLM EngineCore, load model weights, use a GPU, or represent official AgentX cache performance.

## Locked workload

- AIPerf 0.12.0 at the exact commit in `manifest.json`.
- Scenario `inferencex-agentx-mvp` with `semianalysis_cc_traces_weka_062126_256k`, chat streaming, server token counts, a 262,144-token context limit, seed `20260809`, throughput-oriented phase starts, and a 60-second session-concurrency ramp in both warmup and profiling. Trace selection and inter-turn timing remain AgentX-faithful; only the multi-day recorded phase-boundary ramps are replaced by this fixed ramp so high concurrency does not become a synchronized TCP connection storm.
- Live trajectory-tree concurrency `{1024,4096,8192}`.
- Four 900-second legs per point in crossover order `direct, sidecar, sidecar, direct`.
- One 8,192-concurrency qualification leg before measurement. It rejects load-generator CPU, trajectory realization, FD, socket, or admission-queue contamination. Preserved `c131072`, `c65536`, `c32768`, and `c16384` pre-campaign attempts saturated the 64 load-generator cores, so the locked matrix stops at the largest planned concurrency this fixed host allocation can validly generate.
- AIPerf round-robins over eight equivalent `127.0.0.x` frontend addresses so local TCP tuple capacity does not cap concurrency at one ephemeral-port range. This changes neither frontend routing nor the measured backend topology.
- The Dynamo TCP request timeout is 300 seconds in both arms, matching AIPerf's request timeout so a queued burst request does not trip the runtime's five-second default and temporarily remove the only backend worker.
- Sidecar pool sweep `{8,16,32,64,128}` ascending and descending at concurrency 8,192 and the measured sidecar capacity peak. Valid eight-connection main legs are reused.

The Mocker scheduler is shared between arms: DP1 aggregated vLLM mode, speedup zero, block size 64, prefix caching disabled, 524,288 sequences, 67,108,864 batch tokens, and 4,194,304 simulated KV blocks. Prefix caching is intentionally disabled so this remains a request-boundary comparison, not a cache benchmark.

## Host requirements

- A clean committed Dynamo worktree descended from the manifest’s pinned `ai-dynamo/main` ancestor.
- A dedicated DLCluster node with at least 128 physical cores, 512 GiB RAM, and a soft FD limit of at least 262,144.
- Release `dynamo-vllm-sidecar` and `dynamo-vllm-mocker-server` binaries plus Dynamo Python bindings from the same worktree.
- etcd, NATS, `jq`, `ss`, `taskset`, `nc`, and the pinned AIPerf source installed in the same environment.
- Output outside the source worktree. DLCluster `/tmp` is temporary only; copy results to approved storage before releasing the allocation.

## Inspect and run

```bash
python3 benchmarks/frontend/campaigns/vllm-sidecar-mocker-ab/run_campaign.py plan

CAMPAIGN_OUTPUT=/approved/scratch/vllm-sidecar-agentx-ab
python3 benchmarks/frontend/campaigns/vllm-sidecar-mocker-ab/run_campaign.py run --output-root "$CAMPAIGN_OUTPUT" --phase smoke
python3 benchmarks/frontend/campaigns/vllm-sidecar-mocker-ab/run_campaign.py run --output-root "$CAMPAIGN_OUTPUT" --phase qualification
python3 benchmarks/frontend/campaigns/vllm-sidecar-mocker-ab/run_campaign.py run --output-root "$CAMPAIGN_OUTPUT" --phase main
python3 benchmarks/frontend/campaigns/vllm-sidecar-mocker-ab/run_campaign.py run --output-root "$CAMPAIGN_OUTPUT" --phase connections
python3 benchmarks/frontend/campaigns/vllm-sidecar-mocker-ab/run_campaign.py analyze --output-root "$CAMPAIGN_OUTPUT"
```

`--phase all` performs the sequence and analysis. A failed leg stops the campaign and preserves its immutable attempt. `--retry-failed` repeats the same locked leg in a new attempt directory.

The driver records the manifest and SHA-256, exact source and binary hashes, environment inventory, core and resolved plans, the capacity-peak decision, raw AIPerf records and logs, process/CPU/socket telemetry, every individual run, paired comparisons, connection diagnostics, charts, flags, and a scoped report.

## Validation

```bash
python3 benchmarks/frontend/campaigns/vllm-sidecar-mocker-ab/test_campaign.py
cargo test -p dynamo-vllm-mocker --test sidecar
cargo test -p dynamo-vllm-sidecar --test executable
```
