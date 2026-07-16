<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Temporary SGLang gRPC contract

This copy is temporary while Dynamo waits for SGLang to include the source
`sglang/runtime/v1/sglang.proto` contract in a release wheel. Once the contract is
available there, Dynamo should remove this directory, pin and install the
matching `sglang` wheel as a build dependency, and compile the packaged proto
instead.

The contract is an exact copy from Connor Carpenter's SGLang fork commit
[`024402b532779f90722c18517cafeb53511c7a93`](https://github.com/connorcarpenter15/sglang/blob/024402b532779f90722c18517cafeb53511c7a93/proto/sglang/runtime/v1/sglang.proto).
Both files have SHA-256
`23e16a9ee5e1eb0bcaabb0bb7fa6039b6d6ccb41ff0de2f571cc24fedd879806`.
The adjacent `SCHEMA.sha256` is copied from the same commit and pins the
intentional v1 source, generated descriptor, and protocol-revision baseline.

The generated descriptor hash is embedded in the sidecar and compared with
`GetRuntimeInfo.descriptor_sha256` before the Dynamo worker registers. A mismatch
is a startup error rather than a best-effort compatibility mode.
