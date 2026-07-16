// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin client for SGLang's native `sglang.runtime.v1.SglangService`.

use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use dynamo_backend_common::{BackendError, DynamoError, ErrorType};
use tokio::time::{Instant, timeout_at};
use tonic::transport::{Channel, Endpoint};

use crate::args::TransportConfig;
use crate::proto as pb;
use crate::proto::sglang_service_client::SglangServiceClient;

const MAX_MESSAGE_SIZE: usize = 64 * 1024 * 1024;

pub type Client = SglangServiceClient<Channel>;

/// Metadata exposed by SGLang's model/server discovery RPCs.
#[derive(Clone, Debug)]
pub struct Discovery {
    pub model_path: String,
    pub tokenizer_path: String,
    pub served_model_name: Option<String>,
    pub max_model_len: Option<u32>,
    pub runtime_kind: pb::RuntimeKind,
    pub worker_role: pb::WorkerRole,
    pub capacity: pb::RuntimeCapacity,
    pub dp_topology: pb::DataParallelTopology,
    pub bootstrap: Option<pb::DisaggregatedParams>,
    pub observability: Vec<pb::ObservabilityEndpoint>,
    pub reasoning_parser: Option<String>,
    pub tool_call_parser: Option<String>,
    pub weight_version: Option<String>,
}

pub async fn connect(
    uri: &str,
    cfg: &TransportConfig,
    deadline: Instant,
) -> Result<Client, DynamoError> {
    let endpoint = Endpoint::from_shared(uri.to_string())
        .map_err(|err| invalid_arg(format!("invalid SGLang gRPC endpoint `{uri}`: {err}")))?;
    let mut last_err;
    loop {
        match try_connect_once(&endpoint, cfg, deadline).await {
            Ok(client) => return Ok(client),
            Err(err) => {
                last_err = err;
                if Instant::now() >= deadline {
                    return Err(cannot_connect(format!(
                        "could not reach SGLang gRPC at {uri} within {:?}: {last_err}",
                        cfg.deadline
                    )));
                }
                tokio::time::sleep_until((Instant::now() + cfg.poll_interval).min(deadline)).await;
            }
        }
    }
}

async fn try_connect_once(
    endpoint: &Endpoint,
    cfg: &TransportConfig,
    deadline: Instant,
) -> Result<Client, String> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Err("startup deadline elapsed".to_string());
    }
    let endpoint = endpoint
        .clone()
        .connect_timeout(cfg.connect_timeout.min(remaining));
    let channel = timeout_at(deadline, endpoint.connect())
        .await
        .map_err(|_| "startup deadline elapsed while connecting".to_string())?
        .map_err(|e| e.to_string())?;
    Ok(client_from_channel(channel))
}

fn client_from_channel(channel: Channel) -> Client {
    SglangServiceClient::new(channel)
        .max_decoding_message_size(MAX_MESSAGE_SIZE)
        .max_encoding_message_size(MAX_MESSAGE_SIZE)
}

/// Fixed-size pool of independent HTTP/2 connections. Generation calls are
/// round-robined so high concurrency does not funnel through one codec task.
pub struct Pool {
    clients: Vec<Client>,
    next: AtomicUsize,
}

impl Pool {
    pub async fn connect(
        uri: &str,
        cfg: &TransportConfig,
        size: usize,
        deadline: Instant,
    ) -> Result<Self, DynamoError> {
        let size = size.max(1);
        let mut clients = Vec::with_capacity(size);
        for _ in 0..size {
            clients.push(connect(uri, cfg, deadline).await?);
        }
        Ok(Self {
            clients,
            next: AtomicUsize::new(0),
        })
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.clients.len()
    }

    pub fn stream_client(&self) -> Client {
        let index = self.next.fetch_add(1, Ordering::Relaxed) % self.clients.len();
        self.clients[index].clone()
    }

    pub fn control_client(&self) -> Client {
        self.clients[0].clone()
    }
}

pub async fn discover(client: &mut Client, deadline: Instant) -> Result<Discovery, DynamoError> {
    let runtime = rpc_with_deadline(
        "GetRuntimeInfo",
        deadline,
        client.get_runtime_info(pb::GetRuntimeInfoRequest {}),
    )
    .await?
    .into_inner();
    parse_discovery(runtime)
}

pub async fn health_check(client: &mut Client, deadline: Instant) -> Result<bool, DynamoError> {
    rpc_with_deadline(
        "HealthCheck",
        deadline,
        client.health_check(pb::HealthCheckRequest {}),
    )
    .await
    .map(|response| response.into_inner().healthy)
}

pub async fn abort(
    client: &mut Client,
    request: pb::AbortRequest,
    timeout: Duration,
) -> Result<(), DynamoError> {
    rpc_with_deadline("Abort", Instant::now() + timeout, client.abort(request))
        .await
        .map(|_| ())
}

async fn rpc_with_deadline<T, F>(rpc: &str, deadline: Instant, future: F) -> Result<T, DynamoError>
where
    F: Future<Output = Result<T, tonic::Status>>,
{
    match timeout_at(deadline, future).await {
        Ok(Ok(response)) => Ok(response),
        Ok(Err(status)) => Err(status_to_dynamo(rpc, status)),
        Err(_) => Err(connection_timeout(format!(
            "{rpc} exceeded the configured deadline"
        ))),
    }
}

fn parse_discovery(runtime: pb::GetRuntimeInfoResponse) -> Result<Discovery, DynamoError> {
    let model_path = runtime.model_path;
    if model_path.trim().is_empty() {
        return Err(protocol_error(
            "SGLang GetRuntimeInfo returned an empty model_path",
        ));
    }
    if runtime.protocol_revision != pb::PROTOCOL_REVISION {
        return Err(protocol_error(format!(
            "SGLang protocol revision mismatch: sidecar expects `{}`, server reports `{}`",
            pb::PROTOCOL_REVISION,
            runtime.protocol_revision
        )));
    }
    let expected_descriptor = pb::descriptor_sha256();
    if runtime.descriptor_sha256 != expected_descriptor {
        return Err(protocol_error(format!(
            "SGLang descriptor mismatch: sidecar expects {expected_descriptor}, server reports {}; rebuild the sidecar and SGLang from the same proto",
            runtime.descriptor_sha256
        )));
    }
    let runtime_kind = pb::RuntimeKind::try_from(runtime.runtime_kind).map_err(|_| {
        protocol_error(format!(
            "SGLang reported unknown runtime kind {}",
            runtime.runtime_kind
        ))
    })?;
    if runtime_kind == pb::RuntimeKind::Unspecified {
        return Err(protocol_error("SGLang runtime kind is unspecified"));
    }
    let worker_role = pb::WorkerRole::try_from(runtime.worker_role).map_err(|_| {
        protocol_error(format!(
            "SGLang reported unknown worker role {}",
            runtime.worker_role
        ))
    })?;
    if worker_role == pb::WorkerRole::Unspecified {
        return Err(protocol_error("SGLang worker role is unspecified"));
    }
    let capacity = runtime.capacity.unwrap_or_default();
    let max_model_len = u32::try_from(capacity.max_context_length).ok();
    let tokenizer_path = if runtime.tokenizer_path.trim().is_empty() {
        model_path.clone()
    } else {
        runtime.tokenizer_path
    };

    Ok(Discovery {
        model_path,
        tokenizer_path,
        served_model_name: runtime.served_model_name,
        max_model_len,
        runtime_kind,
        worker_role,
        capacity,
        dp_topology: runtime.dp_topology.unwrap_or_default(),
        bootstrap: runtime.bootstrap,
        observability: runtime.observability,
        reasoning_parser: runtime.reasoning_parser,
        tool_call_parser: runtime.tool_call_parser,
        weight_version: runtime.weight_version,
    })
}

fn backend(kind: BackendError, message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(kind))
        .message(message)
        .build()
}

pub fn invalid_arg(message: impl Into<String>) -> DynamoError {
    backend(BackendError::InvalidArgument, message)
}

pub fn engine_shutdown(message: impl Into<String>) -> DynamoError {
    backend(BackendError::EngineShutdown, message)
}

pub fn cannot_connect(message: impl Into<String>) -> DynamoError {
    backend(BackendError::CannotConnect, message)
}

fn connection_timeout(message: impl Into<String>) -> DynamoError {
    backend(BackendError::ConnectionTimeout, message)
}

pub fn protocol_error(message: impl Into<String>) -> DynamoError {
    backend(BackendError::Unknown, message)
}

pub fn status_to_dynamo(rpc: &str, status: tonic::Status) -> DynamoError {
    let kind = match status.code() {
        tonic::Code::InvalidArgument | tonic::Code::NotFound | tonic::Code::OutOfRange => {
            BackendError::InvalidArgument
        }
        tonic::Code::Unavailable => BackendError::CannotConnect,
        tonic::Code::Cancelled => BackendError::Cancelled,
        tonic::Code::DeadlineExceeded => BackendError::ConnectionTimeout,
        _ => BackendError::Unknown,
    };
    backend(
        kind,
        format!("{rpc}: {} ({:?})", status.message(), status.code()),
    )
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use dynamo_backend_common::{BackendError, ErrorType};
    use tokio::net::TcpListener;
    use tokio::time::{Instant, timeout};
    use tonic::transport::Endpoint;

    use super::{client_from_channel, connect, discover, parse_discovery};
    use crate::args::TransportConfig;
    use crate::proto as pb;

    #[test]
    fn discovery_preserves_distinct_tokenizer_path_and_checks_descriptor() {
        let discovery = parse_discovery(pb::GetRuntimeInfoResponse {
            model_path: "model-repo".to_string(),
            tokenizer_path: "tokenizer-repo".to_string(),
            runtime_kind: pb::RuntimeKind::Llm.into(),
            worker_role: pb::WorkerRole::Aggregated.into(),
            protocol_revision: pb::PROTOCOL_REVISION.to_string(),
            descriptor_sha256: pb::descriptor_sha256(),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(discovery.model_path, "model-repo");
        assert_eq!(discovery.tokenizer_path, "tokenizer-repo");
    }

    #[test]
    fn discovery_rejects_descriptor_mismatch() {
        let error = parse_discovery(pb::GetRuntimeInfoResponse {
            model_path: "model-repo".to_string(),
            runtime_kind: pb::RuntimeKind::Llm.into(),
            worker_role: pb::WorkerRole::Aggregated.into(),
            protocol_revision: pb::PROTOCOL_REVISION.to_string(),
            descriptor_sha256: "wrong".to_string(),
            ..Default::default()
        })
        .unwrap_err();
        assert!(error.to_string().contains("descriptor mismatch"));
    }

    #[test]
    fn discovery_rejects_unknown_runtime_before_registration() {
        let error = parse_discovery(pb::GetRuntimeInfoResponse {
            model_path: "model-repo".to_string(),
            runtime_kind: 999,
            worker_role: pb::WorkerRole::Aggregated.into(),
            protocol_revision: pb::PROTOCOL_REVISION.to_string(),
            descriptor_sha256: pb::descriptor_sha256(),
            ..Default::default()
        })
        .unwrap_err();
        assert!(error.to_string().contains("unknown runtime kind"));
    }

    #[tokio::test]
    async fn discovery_deadline_bounds_a_half_open_peer() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let peer = tokio::spawn(async move {
            let (_socket, _) = listener.accept().await.unwrap();
            tokio::time::sleep(Duration::from_secs(5)).await;
        });
        let channel = Endpoint::from_shared(format!("http://{address}"))
            .unwrap()
            .connect_lazy();
        let mut client = client_from_channel(channel);
        let started = Instant::now();
        let result = discover(&mut client, started + Duration::from_millis(100)).await;
        peer.abort();

        assert!(result.is_err());
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[tokio::test]
    async fn malformed_endpoint_fails_before_retrying() {
        let transport = TransportConfig {
            poll_interval: Duration::from_secs(5),
            deadline: Duration::from_secs(30),
            ..TransportConfig::default()
        };
        let result = timeout(
            Duration::from_secs(1),
            connect("http://", &transport, Instant::now() + transport.deadline),
        )
        .await
        .expect("invalid endpoint should not enter the retry loop");
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("invalid endpoint unexpectedly connected"),
        };

        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
    }
}
