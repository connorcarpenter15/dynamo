<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CPU-only vLLM sidecar A/B campaign

This campaign compares the direct Dynamo Mocker request path with `dynamo-vllm-sidecar` and the CPU-only `dynamo-vllm-mocker-server`. It measures the incremental native-vLLM gRPC sidecar cost; it does not run vLLM EngineCore, load weights, or use a GPU.

## Prerequisites

- Use a clean committed Dynamo worktree descended from `f9863c6e420a4fc7af9d9458f0216957e3f757bf`.
- Build release-mode Rust binaries and install the Dynamo Python package and bindings from the same worktree.
- Install AIPerf 0.10.0, etcd, NATS, `jq`, `ss`, `taskset`, and `nc`.
- Run on a dedicated host with at least 48 physical cores. The driver selects one logical CPU per physical core and aborts when preflight utilization exceeds 5%.
- Put the output root outside the Git worktree.

Example build:

```bash
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:?set CARGO_TARGET_DIR outside the worktree}"
cargo build --release -p dynamo-vllm-sidecar --bin dynamo-vllm-sidecar
cargo build --release -p dynamo-vllm-mocker --bin dynamo-vllm-mocker-server
cd lib/bindings/python
maturin develop --uv --release
cd ../../..
uv pip install --no-deps -e .
```

## Inspect and run

Render the deterministic schedule without launching processes:

```bash
python3 benchmarks/frontend/campaigns/vllm-sidecar-mocker-ab/run_campaign.py plan --high-concurrency-shards 4
```

Run the smoke and load-generator preflight first. The preflight writes an immutable one-or-four-shard decision; later phases refuse to run without it.

```bash
CAMPAIGN_OUTPUT=/home/connorc/vllm-sidecar-mocker-ab-$(date -u +%Y%m%dT%H%M%SZ)
python3 benchmarks/frontend/campaigns/vllm-sidecar-mocker-ab/run_campaign.py run --output-root "$CAMPAIGN_OUTPUT" --phase smoke
python3 benchmarks/frontend/campaigns/vllm-sidecar-mocker-ab/run_campaign.py run --output-root "$CAMPAIGN_OUTPUT" --phase preflight
python3 benchmarks/frontend/campaigns/vllm-sidecar-mocker-ab/run_campaign.py run --output-root "$CAMPAIGN_OUTPUT" --phase main
python3 benchmarks/frontend/campaigns/vllm-sidecar-mocker-ab/run_campaign.py run --output-root "$CAMPAIGN_OUTPUT" --phase connections
python3 benchmarks/frontend/campaigns/vllm-sidecar-mocker-ab/run_campaign.py run --output-root "$CAMPAIGN_OUTPUT" --phase profiles
python3 benchmarks/frontend/campaigns/vllm-sidecar-mocker-ab/run_campaign.py analyze --output-root "$CAMPAIGN_OUTPUT"
```

`--phase all` performs the same sequence and runs analysis. A failed leg stops the campaign and preserves its artifacts. `--retry-failed` repeats the identical locked leg in a new attempt directory; it cannot change the workload, topology, affinity, or crossover order.

The native sidecar currently publishes its resolved `--model-path` as the served model identity and has no separate served-name override. The driver therefore passes the fixture's absolute resolved path as both `--model` and `--model-name` in both arms; the local fixture remains the tokenizer source as well.

For completion requests, the campaign overrides AIPerf's approximate synthetic text with exactly `isl` copies of the manifest-locked token ID. This keeps server-visible input lengths exact without changing the shared model or tokenizer identity.

Timed phases use a 15-second request timeout and a bounded 15-second grace after the warmup and measurement boundaries. Saturated requests therefore become recorded failures and drain before the next phase. Because the explicit prompt replaces every generated prompt, AIPerf materializes one reusable synthetic dataset entry per leg.

When saturation produces request timeouts or HTTP 500/503 responses, AIPerf exits nonzero but still emits complete records. The campaign accepts only complete exports containing successes and those three saturation failure classes; configuration errors such as HTTP 400 and other AIPerf failures remain invalid infrastructure legs.

The output root contains the manifest and SHA-256, source/environment/binary hashes, the resolved schedule, per-leg raw artifacts, individual runs, matched-pair CSV/JSON, point-level paired medians, connection-pool diagnostics, SVG comparison charts, flagged points, and `results/report.md`. Four-shard latency percentiles are pooled from AIPerf's records exports; throughput uses aggregate token/request totals divided by the maximum shard wall time.

## Validation

```bash
python3 benchmarks/frontend/campaigns/vllm-sidecar-mocker-ab/test_campaign.py
cargo test -p dynamo-vllm-mocker --test sidecar
cargo test -p dynamo-vllm-sidecar --test executable
```
