// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generated sidecar protobuf / gRPC types for the `sglang.runtime.v1` package.
//!
//! Codegen is driven by `build.rs` from a checksum-verified SGLang source pin.

#![allow(clippy::all)]
#![allow(missing_docs)]

tonic::include_proto!("sglang.runtime.v1");

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    #[test]
    fn vendored_schema_matches_the_focused_sglang_contract() {
        let source = include_bytes!("../proto/sglang.proto");
        assert_eq!(
            format!("{:x}", Sha256::digest(source)),
            "3fd236326112e46b1ec7b3f3ee62933ba4757e96499d11620c404f6c6d134abe"
        );
    }
}
