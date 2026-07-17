<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Temporary SGLang sidecar gRPC contract

This byte-for-byte copy is temporary while Dynamo waits for SGLang to include
the source proto in a release wheel. Once the contract is available there,
Dynamo should remove this directory, pin and install the matching `sglang`
wheel as a build dependency, and compile the packaged proto instead.

The focused typed-generation contract was copied from Connor's SGLang fork at
schema commit
[`924be7754dc7c285f7afbbfd11939ebea9fc0b0b`](https://github.com/connorcarpenter15/sglang/commit/924be7754dc7c285f7afbbfd11939ebea9fc0b0b).
The source and vendored file SHA-256 is
`2e6f234ef1a96158d1d8fe0ed4d8d203ee6014e21ec9437f4475526f6c93ae82`.
