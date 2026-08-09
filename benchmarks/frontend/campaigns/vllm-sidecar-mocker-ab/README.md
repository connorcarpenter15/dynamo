<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CPU-only vLLM sidecar AgentX A/B campaign

This campaign compares the direct Python vLLM Mocker LiveEngine boundary with `dynamo-vllm-sidecar` plus the CPU-only `dynamo-vllm-mocker-server`. It measures native request-plane/gRPC overhead under the pinned InferenceX AgentX traffic shape; it does not run vLLM EngineCore, load model weights, use a GPU, or represent official AgentX cache performance.

## Locked workload

- AIPerf 0.12.0 at the exact commit in `manifest.json`.
- Scenario `inferencex-agentx-mvp` with `semianalysis_cc_traces_weka_062126_256k`, chat streaming, server token counts, a 262,144-token context limit, and seed `20260809`.
- Live trajectory-tree concurrency `{1024,4096,8192,16384,32768,65536,131072}`.
- Four 900-second legs per point in crossover order `direct, sidecar, sidecar, direct`.
- One 131,072-concurrency qualification leg before measurement. It rejects load-generator CPU, trajectory realization, FD, socket, or admission-queue contamination.
- Sidecar pool sweep `{8,16,32,64,128}` ascending and descending at concurrency 32,768 and the measured sidecar capacity peak. Valid eight-connection main legs are reused.

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
