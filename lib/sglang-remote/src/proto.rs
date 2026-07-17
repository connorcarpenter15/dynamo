// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generated protobuf / gRPC types for the `sglang.runtime.v1` package.
//!
//! Codegen is driven by `build.rs` from a checksum-verified SGLang source pin.

#![allow(clippy::all)]
#![allow(missing_docs)]

tonic::include_proto!("sglang.runtime.v1");

pub const PROTOCOL_REVISION: &str = "sglang.runtime.v1.full-parity.1";

pub fn descriptor_sha256() -> String {
    use sha2::{Digest, Sha256};

    let digest = Sha256::digest(include_bytes!(concat!(
        env!("OUT_DIR"),
        "/sglang_descriptor.bin"
    )));
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    #[test]
    fn vendored_schema_matches_the_sglang_baseline() {
        let baseline = include_str!("../proto/sglang/runtime/v1/SCHEMA.sha256");
        let value = |name: &str| {
            baseline
                .lines()
                .find_map(|line| line.strip_prefix(&format!("{name}=")))
                .unwrap_or_else(|| panic!("missing {name} in SCHEMA.sha256"))
        };
        let source = include_bytes!("../proto/sglang/runtime/v1/sglang.proto");
        assert_eq!(
            format!("{:x}", Sha256::digest(source)),
            value("source_sha256")
        );
        assert_eq!(super::descriptor_sha256(), value("descriptor_sha256"));
        assert_eq!(super::PROTOCOL_REVISION, value("protocol_revision"));
    }
}
