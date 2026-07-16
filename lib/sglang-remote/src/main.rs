// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Entry point for the `dynamo-sglang-remote` binary.
//!
//! Mirrors the mocker backend: bootstrap-discover the engine in `from_args`
//! (building the [`WorkerConfig`](dynamo_backend_common::WorkerConfig) `run`
//! needs synchronously), then hand the engine to the shared runtime harness.

use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let (engine, config) = dynamo_sglang_remote::SglangRemoteEngine::from_args(None)?;
    let runtime_kind = engine.runtime_kind();
    let engine = Arc::new(engine);
    match runtime_kind {
        dynamo_sglang_remote::proto::RuntimeKind::Llm => dynamo_backend_common::run(engine, config),
        dynamo_sglang_remote::proto::RuntimeKind::Embedding
        | dynamo_sglang_remote::proto::RuntimeKind::Image
        | dynamo_sglang_remote::proto::RuntimeKind::Video => {
            dynamo_backend_common::run_raw(engine, config)
        }
        dynamo_sglang_remote::proto::RuntimeKind::Unspecified => {
            anyhow::bail!("SGLang runtime kind is unspecified")
        }
    }
}
