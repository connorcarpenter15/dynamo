// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Thin OpenEngine v1 gRPC client: connect-with-backoff, two-RPC discovery,
//! and error mapping into [`DynamoError`].
//!
//! The generated tonic [`Client`] is cheap to clone (the underlying
//! [`Channel`] multiplexes over one HTTP/2 connection), so the engine stores a
//! single client and clones it per call.

use dynamo_backend_common::{BackendError, DynamoError, ErrorType};
use tokio::time::Instant;
use tonic::transport::{Channel, Endpoint};

use crate::args::TransportConfig;
use crate::proto as pb;
use crate::proto::open_engine_client::OpenEngineClient;

/// Connected OpenEngine client over a tonic [`Channel`].
pub type Client = OpenEngineClient<Channel>;

/// Engine discovery snapshot: identity / role / parallelism plus model caps.
#[derive(Clone, Debug)]
pub struct Discovery {
    pub engine: pb::EngineInfo,
    pub model: pb::ModelInfo,
}

/// Dial the engine, retrying with backoff until reachable or the deadline
/// elapses.
///
/// Each attempt is bounded by [`TransportConfig::connect_timeout`]; failed
/// attempts are retried every [`TransportConfig::poll_interval`] until
/// [`TransportConfig::deadline`].
pub async fn connect(uri: &str, cfg: &TransportConfig) -> Result<Client, DynamoError> {
    let deadline = Instant::now() + cfg.deadline;
    let mut last_err;
    loop {
        match try_connect_once(uri, cfg).await {
            Ok(client) => return Ok(client),
            Err(e) => {
                last_err = e;
                if Instant::now() >= deadline {
                    return Err(cannot_connect(format!(
                        "could not reach OpenEngine at {uri} within {:?}: {last_err}",
                        cfg.deadline
                    )));
                }
                tokio::time::sleep(cfg.poll_interval).await;
            }
        }
    }
}

async fn try_connect_once(uri: &str, cfg: &TransportConfig) -> Result<Client, String> {
    let endpoint = Endpoint::from_shared(uri.to_string())
        .map_err(|e| format!("invalid endpoint `{uri}`: {e}"))?
        .connect_timeout(cfg.connect_timeout);
    let channel = endpoint.connect().await.map_err(|e| e.to_string())?;
    Ok(OpenEngineClient::new(channel))
}

/// Fetch engine + model metadata in one shot.
pub async fn discover(client: &mut Client) -> Result<Discovery, DynamoError> {
    let engine = client
        .get_engine_info(pb::GetEngineInfoRequest {})
        .await
        .map_err(|s| status_to_dynamo("GetEngineInfo", s))?
        .into_inner();
    let model = client
        .get_model_info(pb::GetModelInfoRequest {})
        .await
        .map_err(|s| status_to_dynamo("GetModelInfo", s))?
        .into_inner();
    Ok(Discovery { engine, model })
}

// ============================================================================
// Error mapping
// ============================================================================

fn backend(kind: BackendError, msg: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(ErrorType::Backend(kind))
        .message(msg)
        .build()
}

/// Client-supplied bad input.
pub fn invalid_arg(msg: impl Into<String>) -> DynamoError {
    backend(BackendError::InvalidArgument, msg)
}

/// The engine is gone / never came up / used before start.
pub fn engine_shutdown(msg: impl Into<String>) -> DynamoError {
    backend(BackendError::EngineShutdown, msg)
}

/// Could not establish the transport to the engine.
pub fn cannot_connect(msg: impl Into<String>) -> DynamoError {
    backend(BackendError::CannotConnect, msg)
}

/// Map a tonic transport-level [`Status`](tonic::Status) to a typed error.
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
    backend(kind, format!("{rpc}: {} ({:?})", status.message(), status.code()))
}

/// Map a structured [`pb::EngineError`] stream event to a typed error.
pub fn engine_error_to_dynamo(err: &pb::EngineError) -> DynamoError {
    let code = pb::ErrorCode::try_from(err.code).unwrap_or(pb::ErrorCode::Unspecified);
    let kind = match code {
        pb::ErrorCode::InvalidArgument
        | pb::ErrorCode::UnsupportedFeature
        | pb::ErrorCode::RoleMismatch
        | pb::ErrorCode::ModelNotFound
        | pb::ErrorCode::KvSessionNotFound => BackendError::InvalidArgument,
        pb::ErrorCode::Cancelled => BackendError::Cancelled,
        pb::ErrorCode::Draining => BackendError::EngineShutdown,
        pb::ErrorCode::KvTransferFailed => BackendError::Disconnected,
        _ => BackendError::Unknown,
    };
    backend(kind, format!("engine error [{code:?}]: {}", err.message))
}
