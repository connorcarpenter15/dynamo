# DEP draft: SGLang native gRPC sidecar

> Status: local prototype draft. File this in `ai-dynamo/enhancements` and obtain
> approval before moving the associated Dynamo pull request out of draft.

## Area

`backend-sglang`

## Summary

Make SGLang's native `sglang.runtime.v1` gRPC service a supported out-of-process
Dynamo backend. A Rust sidecar discovers the runtime, rejects incompatible
descriptors before worker registration, and selects `LLMEngine` for token
generation or `RawEngine` for embeddings and image/video generation. The same
contract exposes management operations while high-volume metrics and KV-cache
events remain on their native HTTP and ZMQ transports.

## Motivation

The integrated Python backend couples Dynamo and SGLang process lifecycles and
cannot independently evolve or restart the serving adapter. The previous native
gRPC prototype covered only basic, single-choice text generation and would have
silently dropped or string-parsed multimodal inputs, choice attribution,
logprobs, usage, LoRA lifecycle, advanced controls, embeddings, diffusion, and
observability. A versioned typed boundary is required before the sidecar can be
considered a feature-preserving replacement.

## Proposal

- Treat the protobuf descriptor as the compatibility boundary. SGLang reports a
  protocol revision and SHA-256 of its compiled descriptor; the sidecar embeds
  the same descriptor and fails before registration on any mismatch.
- Register LLM runtimes through `LLMEngine` and embedding/image/video runtimes
  through `RawEngine`. Reject unknown runtime kinds and non-aggregated raw
  runtimes.
- Preserve all Dynamo generation semantics represented by SGLang: independently
  interleaved choices, typed terminals and partial-choice errors, guided
  decoding, reasoning controls, stop visibility, logprobs, usage, multimodal
  routing hashes, prompt embeddings, and disaggregated handoff.
- Carry decoded media and prompt tensors inline for small payloads or through
  typed NIXL external-buffer descriptors. Retain source registrations until the
  request terminates and propagate cancellation through gRPC cleanup.
- Reuse Dynamo's LoRA downloader/cache. Serialize operations per adapter,
  publish topology-matching discovery cards, and roll back either SGLang or
  discovery state when the other side fails.
- Expose memory, profiling, disk/tensor/distributed/IPC weight updates, and
  version updates as typed controls. Unregister the worker around disruptive
  operations and restore registration deterministically.
- Discover SGLang's per-rank ZMQ KV-event endpoints and native Prometheus URLs.
  The sidecar bridges Prometheus exposition to Dynamo's metrics endpoint; it
  does not proxy samples or KV events through request/response gRPC.
- Temporarily vendor the exact SGLang proto until a released SGLang wheel ships
  the source contract. Record the source commit and proto hash beside the copy.

## Alternate solutions

- Keep the integrated Python backend: retains parity but does not provide the
  desired process and release boundary.
- Use OpenAI-shaped JSON for all traffic: avoids schema growth but loses typed
  validation, choice/terminal invariants, and zero-copy tensor descriptors.
- Tunnel metrics and KV events over the serving RPC: creates head-of-line
  blocking and duplicates existing purpose-built transports.
- Maintain old and new generation schemas simultaneously: rejected for this
  prototype because it prolongs the stringly typed `meta_info` contract and
  obscures compatibility failures.

## Requirements

- The sidecar MUST reject protocol or descriptor mismatches before registering.
- Every requested choice MUST produce exactly one typed terminal result.
- Cancellation MUST terminate SGLang work and release external-buffer guards.
- Disruptive controls MUST not leave a stale discovery registration.
- LoRA publication and SGLang load state MUST converge or report rollback
  failure explicitly.
- Metrics and KV events MUST remain out of the generation data plane.
- Unknown runtime kinds and malformed tensors/descriptors MUST fail
  deterministically.
- The prototype PR MUST remain draft until this DEP is filed and approved.

## References

- SGLang contract: `proto/sglang/runtime/v1/sglang.proto`
- Sidecar implementation: `lib/sglang-remote`
- Dynamo engine traits: `lib/backend-common/src/engine.rs`
- Temporary vendoring record: `lib/sglang-remote/proto/README.md`
