// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compiles the vendored OpenEngine v1 proto into client + server stubs.
//!
//! The proto is a synced copy of the canonical contract at
//! `openengine/proto/openengine.proto` (see `openengine/gen.sh`). Mirrors the
//! tonic-build setup used by `lib/llm/build.rs`.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure().compile_protos(&["proto/openengine.proto"], &["proto"])?;
    println!("cargo:rerun-if-changed=proto/openengine.proto");
    Ok(())
}
