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
[`9d40482adade66da9fa06ecb08c3de6f34f8dd2d`](https://github.com/connorcarpenter15/sglang/commit/9d40482adade66da9fa06ecb08c3de6f34f8dd2d).
The source and vendored file SHA-256 is
`23ad2ca173f6d4ea250953c95fb863be2470b4b7c368973ee4741feb91bbe84a`.
