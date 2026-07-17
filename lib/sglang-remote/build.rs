// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compiles client stubs from the temporarily vendored SGLang gRPC contract.

use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let proto_dir = manifest_dir.join("proto");
    // Preserve the canonical import path used by SGLang's own build. The file
    // name is part of the generated descriptor set, so flattening the vendored
    // copy to `proto/sglang.proto` would produce a different descriptor hash
    // even when the schema bytes are identical.
    let proto_path = proto_dir.join("sglang/runtime/v1/sglang.proto");
    let protoc_include = protoc_bin_vendored::include_path()?;
    let mut prost_config = prost_build::Config::new();
    prost_config.protoc_executable(protoc_bin_vendored::protoc_bin_path()?);

    // Keep the compatibility flag accepted by both the vendored compiler and
    // older downstream toolchains. It is a no-op after proto3 optional became
    // stable, but still documents the contract's optional-field requirement.
    tonic_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional")
        .file_descriptor_set_path(PathBuf::from(env::var("OUT_DIR")?).join("sglang_descriptor.bin"))
        .compile_protos_with_config(
            prost_config,
            &[proto_path.as_path()],
            &[proto_dir.as_path(), protoc_include.as_path()],
        )?;

    println!("cargo:rerun-if-changed={}", proto_path.display());
    println!(
        "cargo:rerun-if-changed={}",
        proto_dir.join("sglang/runtime/v1/SCHEMA.sha256").display()
    );
    Ok(())
}
