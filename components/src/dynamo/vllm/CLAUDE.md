# vLLM Component

Dynamo's vLLM backend. There are **two** ways Dynamo drives vLLM, and they
coexist:

1. **In-process** (`llm_engine.py` → `VllmLLMEngine`): the default. Imports
   `vllm`, instantiates `AsyncLLM` directly inside the Dynamo worker process.
   Shares lifecycle, signals, and the GPU process with the engine.
2. **Sidecar** (`sidecar/` → `VllmSidecarLLMEngine`): NEW. The Dynamo worker
   talks to a separate native `vllm serve` process over the OpenEngine v1 gRPC
   contract. No `import vllm` in the worker. See `sidecar/CLAUDE.md`.

Both are `dynamo.common.backend.LLMEngine` subclasses wired through
`run(...)`. Read `dynamo/components/src/dynamo/common/backend/CLAUDE.md` first
— it owns the lifecycle, the request/response TypedDict contract, and the
**zero-duplication-across-engines** constraint.

## The boundary

| | In-process (`VllmLLMEngine`) | Sidecar (`VllmSidecarLLMEngine`) |
|---|---|---|
| Module | `llm_engine.py` | `sidecar/` |
| Entry | `unified_main.py` (`run(VllmLLMEngine)`) | `sidecar/unified_main.py` (`run(VllmSidecarLLMEngine)`) |
| Engine | `vllm.AsyncLLM` in-process | native `vllm serve` over gRPC |
| Imports vllm? | yes | **no** — OpenEngine client only |
| KV transport | NixlConnector (internal) | NixlConnector on the engine side; sidecar advertises sources |
| GPU | same process | separate process/container |

The sidecar exists to decouple Dynamo's container/runtime from vLLM's and to
let users keep the native `vllm serve` UX. It is **not** a replacement for the
in-process path — do not delete or refactor `VllmLLMEngine` when working on the
sidecar.

## In-process engine (`VllmLLMEngine`) — method map

`llm_engine.py`:

- `from_args(argv)` — parse vLLM `AsyncEngineArgs` + Dynamo args, build
  `WorkerConfig`, construct (not start) the engine.
- `start(worker_id)` — `AsyncLLM.from_vllm_config(...)`, return `EngineConfig`
  (model, context length, block size, total KV blocks). vLLM's KV transport is
  internal (NixlConnector) so `bootstrap_host/port` stay `None`.
- `generate(request, context)` — map `GenerateRequest` TypedDict →
  `TokensPrompt` + `SamplingParams`, iterate `engine_client.generate(...)`,
  yield `GenerateChunk`s. Disagg dispatch keys off
  `WorkerConfig.disaggregation_mode` (prefill caps 1 token + packs
  `kv_transfer_params` into terminal `disaggregated_params`; decode reads
  `prefill_result`).
- `kv_event_sources()` — when KV routing enabled, returns the ZMQ publisher
  sources.
- `component_metrics_dp_ranks()` / `attach_snapshot_publisher()` —
  per-iteration stat-logger (`_UnifiedStatLogger`) pushes `ComponentSnapshot`.
- `register_prometheus()` — bridges vLLM's `vllm:` registry.
- `abort()`, `health_check_payload()`, `cleanup()` (null-safe).

The sidecar engine mirrors this method-for-method but sources every value from
OpenEngine RPCs instead of in-process vLLM objects.

## Other files

`args.py`, `backend_args.py`, `handlers.py`, `main.py`, `worker_factory.py`,
`publisher.py`, `health_check.py`, etc. are the **legacy / parallel** entry
path (pre-`unified_main`). Per the backend module's design constraint, that
path stays untouched — the `unified_main.py` files are the separate,
current path.

## Tips for AI assistants

- **Read the backend ABC + CLAUDE.md before editing either engine.**
  `common/backend/engine.py` defines `LLMEngine`, `GenerateRequest`,
  `GenerateChunk`, `EngineConfig`.
- **Keep logging standardized** across vllm/sglang/trtllm (see backend
  CLAUDE.md "Logging").
- **Sidecar must not import `vllm`.** If you find yourself reaching for a vLLM
  type in `sidecar/`, the value belongs in an OpenEngine RPC response instead.
- **Real-engine tests on computelab/lyris** (see root `CLAUDE.md`). Local =
  unit tests + CPU fake-servicer only.
