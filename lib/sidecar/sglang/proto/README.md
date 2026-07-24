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
[`2cfbd2de055efd52b6997d739a3639e3ef35a3b9`](https://github.com/connorcarpenter15/sglang/commit/2cfbd2de055efd52b6997d739a3639e3ef35a3b9).
The source and vendored file SHA-256 is
`3fd236326112e46b1ec7b3f3ee62933ba4757e96499d11620c404f6c6d134abe`.
