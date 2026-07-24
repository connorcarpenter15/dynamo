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
[`7b10460ddcb5fe6c8f0f12e5b046c0499a568441`](https://github.com/connorcarpenter15/sglang/commit/7b10460ddcb5fe6c8f0f12e5b046c0499a568441).
The source and vendored file SHA-256 is
`73dbab9686eccd5442044eb89b172fca230d12487e4dc85d5668e5bef21c9002`.
