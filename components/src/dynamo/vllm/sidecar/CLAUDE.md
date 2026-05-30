# vLLM Sidecar (OpenEngine client)

`VllmSidecarLLMEngine` is a `dynamo.common.backend.LLMEngine` subclass that
drives a **native `vllm serve` process** over the OpenEngine v1 gRPC contract,
instead of importing `vllm` in-process. It is a sibling of
`dynamo.vllm.llm_engine:VllmLLMEngine` — see `../CLAUDE.md` for the boundary.

**Hard rule: this package must never `import vllm`.** Every value the engine
needs comes from an OpenEngine RPC. If you reach for a vLLM type here, it
belongs in an RPC response instead.

## Files (planned)

```
sidecar/
  __init__.py
  __main__.py        python -m dynamo.vllm.sidecar
  unified_main.py    run(VllmSidecarLLMEngine)
  args.py            sidecar CLI args (WorkerConfig + OpenEngine endpoint flags)
  client.py          thin wrapper over generated gRPC stub (reconnect, deadlines, cancel)
  llm_engine.py      VllmSidecarLLMEngine(LLMEngine)
  _openengine/       GENERATED stubs (openengine_pb2*.py) — do NOT hand-edit
  tests/             unit tests vs in-process fake servicer
```

`_openengine/` is generated from the canonical proto:
`../../../../../../../openengine/gen.sh <this>/_openengine` (run from the
`openengine/` peer dir; see `openengine/README.md`). Regenerate after any proto
change. Treat it as a build artifact.

## LLMEngine method → OpenEngine RPC mapping

| `LLMEngine` method | OpenEngine RPC(s) | Notes |
|---|---|---|
| `from_args` | — | parse CLI, build `WorkerConfig`, construct (not-connected) client |
| `start(worker_id)` | `GetEngineInfo`, `GetModelInfo`, `Health` (poll until READY) | validate role matches `disaggregation_mode`; build `EngineConfig` from responses; `bootstrap_host/port` stay `None` (vLLM KV transport is internal) |
| `generate(request, ctx)` | `Generate` (stream) | translate TypedDict ↔ proto; see streaming contract below |
| `abort(ctx)` | `Abort(request_id=ctx.id())` | idempotent |
| `drain()` | `Drain` (stream) | log + swallow failures (matches ABC contract) |
| `cleanup()` | — | close channel, cancel pending streams; null-safe + idempotent |
| `kv_event_sources()` | `GetKvEventSources` | map to `dynamo.common.backend.publisher.KvEventSource`; opting in = advertising sources |
| `health_check_payload()` | — | return a 1-token canary; runtime round-trips it through `generate()` |

`register_prometheus` / `component_metrics_dp_ranks` /
`attach_snapshot_publisher` are **deferred** in v1 (lifecycle gauges emit from
the Rust side for free). A later version polls `GetLoad` and pushes
`ComponentSnapshot`s — cadence design is a follow-up.

## generate() streaming contract

Translate the `GenerateRequest` TypedDict → OpenEngine `GenerateRequest`
(token_ids, sampling, stop). Forward W3C trace headers via
`dynamo.common.backend.telemetry.engine_trace_kwargs(context)`. Then
async-iterate `GenerateResponse`:

- `token` → `yield GenerateChunk(token_ids=..., index=0)`.
- `prefill_ready` (prefill role) → stash the `KvSessionRef`; emit it on the
  terminal chunk's `disaggregated_params`.
- `finished` → terminal `GenerateChunk` with `finish_reason`,
  `completion_usage`, and (prefill role) the `disaggregated_params` blob.
- `error` → raise the matching `DynamoException` subclass so the Rust bridge
  surfaces a typed `BackendError`. Map `ErrorCode` → exception type.

## Disaggregation handshake (KvSessionRef ↔ kv_transfer_params)

`WorkerConfig.disaggregation_mode` is the single source of truth (see backend
CLAUDE.md). Use the shared helpers in `dynamo.common.backend.disagg`; do not
reinvent.

- **Prefill** (`DisaggregationMode.Prefill`):
  1. `enforce_prefill_max_tokens` (cap output to 1 token).
  2. `Generate` → expect `PrefillReady(KvSessionRef)` before/with first token.
  3. Wrap the `KvSessionRef` (session_id, transfer_backend, endpoints, dp_rank,
     attributes) into terminal `GenerateChunk.disaggregated_params` so Dynamo's
     `PrefillRouter` forwards it to the decode peer.
- **Decode** (`DisaggregationMode.Decode`):
  1. `require_prefill_result(request)` → read
     `request["prefill_result"]["disaggregated_params"]` (fail loud if missing).
  2. Repack into OpenEngine `KvSessionRef`, set on `GenerateRequest.kv_session`.
  3. Stream tokens normally.

The vLLM-side servicer turns `KvSessionRef.attributes` back into vLLM's
`kv_transfer_params` blob — that mapping lives on the engine side
(`vllm/vllm/entrypoints/openengine/`), keeping the NixlConnector path identical
to vLLM's existing HTTP disagg path.

Migration is **not** an OpenEngine concern: Dynamo's frontend handles it via
token-replay. The sidecar just must not break it (test: kill engine mid-stream,
frontend reroutes — no dup/loss).

## Failure modes to handle in client.py

- **Engine restart / channel down**: `start()` polls `Health` with backoff up
  to a deadline before giving up. Mid-stream channel loss surfaces as a
  `Generate` stream error → typed `DynamoException` (lets the frontend migrate).
- **gRPC stream cancel**: `abort()` calls OpenEngine `Abort`; ensure the
  client-side stream is cancelled too so the engine releases KV/scheduler slots.
  Idempotent — Dynamo may abort an already-finished request.
- **Drain races**: `drain()` runs after discovery-unregister + grace period
  while NATS/etcd are still alive. Open the `Drain` stream, read
  `DrainResponse` until `in_flight_requests==0`/`open_kv_sessions==0` or
  deadline. Swallow failures (ABC contract: shutdown proceeds regardless).
- **Model mismatch**: `start()` validates `--model`/`--served-model-name`
  against `GetEngineInfo`/`GetModelInfo`; refuse to register if mismatched.

## Out of scope for v1

Multimodal/vision/embedding/diffusion, LoRA dynamic loading, native
`SubscribeKvEvents` (use `GetKvEventSources` compatibility path),
`SubscribeRuntimeEvents`. Fail-fast on multimodal disagg edge cases (expanded
prompt token IDs / image embeddings in the handoff) — addressed in follow-up.

## Testing

- Unit: `generate`/`abort`/`drain`/`cleanup` against an in-process fake
  OpenEngine servicer (CPU-only, runs locally).
- Conformance: feed through the backend conformance kit (see
  `common/backend/CLAUDE.md`).
- Integration / disagg / KV-routing: computelab or lyris only (see root
  `CLAUDE.md`).
