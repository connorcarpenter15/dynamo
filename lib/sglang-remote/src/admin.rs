// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed engine-control conversion for SGLang's native gRPC service.

use std::collections::HashMap;
use std::future::Future;
use std::time::Duration;

use base64::Engine;
use dynamo_backend_common::DynamoError;
use serde::Deserialize;
use serde_json::{Value, json};
use tonic::Response;

use crate::client::{self, Client};
use crate::proto as pb;
use crate::protocol::json_object_to_struct;

pub(crate) const CONTROLS: [&str; 9] = [
    "start_profile",
    "stop_profile",
    "release_memory_occupation",
    "resume_memory_occupation",
    "update_weights_from_disk",
    "update_weights_from_tensor",
    "update_weights_from_distributed",
    "update_weights_from_ipc",
    "update_weight_version",
];

pub(crate) fn is_memory_release(control: &str) -> bool {
    control == "release_memory_occupation"
}

pub(crate) fn is_memory_resume(control: &str) -> bool {
    control == "resume_memory_occupation"
}

pub(crate) fn is_disruptive(control: &str) -> bool {
    matches!(
        control,
        "release_memory_occupation"
            | "resume_memory_occupation"
            | "update_weights_from_disk"
            | "update_weights_from_tensor"
            | "update_weights_from_distributed"
            | "update_weights_from_ipc"
            | "update_weight_version"
    )
}

async fn rpc<T, F>(name: &str, timeout: Duration, future: F) -> Result<T, DynamoError>
where
    F: Future<Output = Result<Response<T>, tonic::Status>>,
{
    match tokio::time::timeout(timeout, future).await {
        Ok(Ok(response)) => Ok(response.into_inner()),
        Ok(Err(status)) => Err(client::status_to_dynamo(name, status)),
        Err(_) => Err(client::cannot_connect(format!(
            "{name} exceeded the configured {:?} timeout",
            timeout
        ))),
    }
}

fn response(success: bool, message: impl Into<String>) -> Value {
    json!({
        "status": if success { "ok" } else { "error" },
        "success": success,
        "message": message.into(),
    })
}

fn behavior(body: &Value) -> pb::UpdateBehavior {
    pb::UpdateBehavior {
        flush_cache: body
            .get("flush_cache")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        abort_all_requests: body
            .get("abort_all_requests")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        weight_version: body
            .get("weight_version")
            .and_then(Value::as_str)
            .map(str::to_owned),
        torch_empty_cache: body
            .get("torch_empty_cache")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    }
}

#[derive(Default, Deserialize)]
struct ProfileBody {
    output_dir: Option<String>,
    start_step: Option<i32>,
    num_steps: Option<i32>,
    #[serde(default)]
    activities: Vec<String>,
    #[serde(default)]
    profile_by_stage: bool,
    with_stack: Option<bool>,
    record_shapes: Option<bool>,
    profile_id: Option<String>,
    #[serde(default)]
    merge_profiles: bool,
    profile_prefix: Option<String>,
    #[serde(default)]
    profile_stages: Vec<String>,
}

#[derive(Deserialize)]
struct DiskBody {
    model_path: String,
    load_format: Option<String>,
    #[serde(default, alias = "is_async")]
    async_update: bool,
    #[serde(default)]
    keep_pause: bool,
    #[serde(default)]
    recapture_cuda_graph: bool,
    #[serde(default)]
    token_step: i64,
    manifest: Option<Value>,
}

#[derive(Deserialize)]
struct TensorBody {
    serialized_named_tensors: Vec<Value>,
    load_format: Option<String>,
    disable_draft_model: Option<bool>,
}

#[derive(Deserialize)]
struct NamedTensorBody {
    name: String,
    dtype: String,
    shape: Vec<i64>,
}

#[derive(Deserialize)]
struct DistributedBody {
    #[serde(default)]
    tensors: Vec<NamedTensorBody>,
    #[serde(default)]
    names: Vec<String>,
    #[serde(default)]
    dtypes: Vec<String>,
    #[serde(default)]
    shapes: Vec<Vec<i64>>,
    #[serde(default = "default_group_name")]
    group_name: String,
    load_format: Option<String>,
}

fn default_group_name() -> String {
    "weight_update_group".to_string()
}

#[derive(Deserialize)]
struct IpcBody {
    zmq_handles: HashMap<String, String>,
}

fn parse_body<T: for<'de> Deserialize<'de>>(
    operation: &str,
    body: Value,
) -> Result<T, DynamoError> {
    serde_json::from_value(body)
        .map_err(|error| client::invalid_arg(format!("invalid {operation} body: {error}")))
}

fn decode_bytes(value: &Value) -> Result<Vec<u8>, DynamoError> {
    if let Some(encoded) = value.as_str() {
        return base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|error| client::invalid_arg(format!("invalid base64 tensor: {error}")));
    }
    let values = value
        .as_array()
        .ok_or_else(|| client::invalid_arg("serialized tensor must be base64 or a byte array"))?;
    values
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| u8::try_from(value).ok())
                .ok_or_else(|| client::invalid_arg("serialized tensor byte is outside 0..=255"))
        })
        .collect()
}

fn normalize_named_tensors(body: DistributedBody) -> Result<Vec<pb::NamedTensorSpec>, DynamoError> {
    if !body.tensors.is_empty() {
        return Ok(body
            .tensors
            .into_iter()
            .map(|tensor| pb::NamedTensorSpec {
                name: tensor.name,
                dtype: tensor.dtype,
                shape: tensor.shape,
            })
            .collect());
    }
    if body.names.len() != body.dtypes.len() || body.names.len() != body.shapes.len() {
        return Err(client::invalid_arg(
            "distributed weight names, dtypes, and shapes must have identical lengths",
        ));
    }
    Ok(body
        .names
        .into_iter()
        .zip(body.dtypes)
        .zip(body.shapes)
        .map(|((name, dtype), shape)| pb::NamedTensorSpec { name, dtype, shape })
        .collect())
}

pub(crate) async fn execute(
    client: &mut Client,
    timeout: Duration,
    control: &str,
    body: Value,
) -> Result<Value, DynamoError> {
    let body = if body.is_null() { json!({}) } else { body };
    match control {
        "release_memory_occupation" => {
            let tags = body
                .get("tags")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default();
            let result = rpc(
                "ReleaseMemory",
                timeout,
                client.release_memory(pb::ReleaseMemoryRequest { tags }),
            )
            .await?;
            Ok(response(result.success, result.message))
        }
        "resume_memory_occupation" => {
            let tags = body
                .get("tags")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default();
            let result = rpc(
                "ResumeMemory",
                timeout,
                client.resume_memory(pb::ResumeMemoryRequest { tags }),
            )
            .await?;
            Ok(response(result.success, result.message))
        }
        "start_profile" => {
            let body: ProfileBody = parse_body(control, body)?;
            let result = rpc(
                "StartProfile",
                timeout,
                client.start_profile(pb::StartProfileRequest {
                    output_dir: body.output_dir,
                    start_step: body.start_step,
                    num_steps: body.num_steps,
                    activities: body.activities,
                    profile_by_stage: body.profile_by_stage,
                    with_stack: body.with_stack,
                    record_shapes: body.record_shapes,
                    profile_id: body.profile_id,
                    merge_profiles: body.merge_profiles,
                    profile_prefix: body.profile_prefix,
                    profile_stages: body.profile_stages,
                }),
            )
            .await?;
            Ok(response(result.success, result.message))
        }
        "stop_profile" => {
            let profile_id = body
                .get("profile_id")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let result = rpc(
                "StopProfile",
                timeout,
                client.stop_profile(pb::StopProfileRequest { profile_id }),
            )
            .await?;
            Ok(response(result.success, result.message))
        }
        "update_weights_from_disk" => {
            let parsed: DiskBody = parse_body(control, body.clone())?;
            let result = rpc(
                "UpdateWeightsFromDisk",
                timeout,
                client.update_weights_from_disk(pb::UpdateWeightsFromDiskRequest {
                    model_path: parsed.model_path,
                    load_format: parsed.load_format,
                    behavior: Some(behavior(&body)),
                    async_update: parsed.async_update,
                    keep_pause: parsed.keep_pause,
                    recapture_cuda_graph: parsed.recapture_cuda_graph,
                    token_step: parsed.token_step,
                    manifest: parsed
                        .manifest
                        .map(|value| json_object_to_struct(&value))
                        .transpose()?,
                }),
            )
            .await?;
            Ok(json!({
                "status": if result.success { "ok" } else { "error" },
                "success": result.success,
                "message": result.message,
                "num_paused_requests": result.num_paused_requests,
            }))
        }
        "update_weights_from_tensor" => {
            let parsed: TensorBody = parse_body(control, body.clone())?;
            let serialized_named_tensors = parsed
                .serialized_named_tensors
                .iter()
                .map(decode_bytes)
                .collect::<Result<Vec<_>, _>>()?;
            let result = rpc(
                "UpdateWeightsFromTensor",
                timeout,
                client.update_weights_from_tensor(pb::UpdateWeightsFromTensorRequest {
                    serialized_named_tensors,
                    load_format: parsed.load_format,
                    behavior: Some(behavior(&body)),
                    disable_draft_model: parsed.disable_draft_model,
                }),
            )
            .await?;
            Ok(response(result.success, result.message))
        }
        "update_weights_from_distributed" => {
            let parsed: DistributedBody = parse_body(control, body.clone())?;
            let group_name = parsed.group_name.clone();
            let load_format = parsed.load_format.clone();
            let tensors = normalize_named_tensors(parsed)?;
            let result = rpc(
                "UpdateWeightsFromDistributed",
                timeout,
                client.update_weights_from_distributed(pb::UpdateWeightsFromDistributedRequest {
                    tensors,
                    group_name,
                    load_format,
                    behavior: Some(behavior(&body)),
                }),
            )
            .await?;
            Ok(response(result.success, result.message))
        }
        "update_weights_from_ipc" => {
            let parsed: IpcBody = parse_body(control, body.clone())?;
            let result = rpc(
                "UpdateWeightsFromIPC",
                timeout,
                client.update_weights_from_ipc(pb::UpdateWeightsFromIpcRequest {
                    zmq_handles: parsed.zmq_handles,
                    behavior: Some(behavior(&body)),
                }),
            )
            .await?;
            Ok(response(result.success, result.message))
        }
        "update_weight_version" => {
            let version = body
                .get("new_version")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| client::invalid_arg("new_version is required"))?;
            let result = rpc(
                "UpdateWeightVersion",
                timeout,
                client.update_weight_version(pb::UpdateWeightVersionRequest {
                    new_version: version.to_string(),
                    abort_all_requests: body
                        .get("abort_all_requests")
                        .and_then(Value::as_bool)
                        .unwrap_or(true),
                }),
            )
            .await?;
            Ok(response(result.success, result.message))
        }
        _ => Ok(json!({
            "status": "error",
            "message": format!("unsupported engine control: {control}"),
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distributed_parallel_arrays_require_equal_lengths() {
        let body: DistributedBody = serde_json::from_value(json!({
            "names": ["a"], "dtypes": ["float16", "float32"], "shapes": [[2, 2]]
        }))
        .unwrap();
        assert!(normalize_named_tensors(body).is_err());
    }

    #[test]
    fn tensor_bytes_accept_base64_and_arrays() {
        assert_eq!(decode_bytes(&json!("AQID")).unwrap(), vec![1, 2, 3]);
        assert_eq!(decode_bytes(&json!([1, 2, 3])).unwrap(), vec![1, 2, 3]);
        assert!(decode_bytes(&json!([256])).is_err());
    }

    #[test]
    fn disruptive_control_classification_is_explicit() {
        assert!(!is_disruptive("start_profile"));
        assert!(is_disruptive("update_weights_from_disk"));
        assert!(is_memory_release("release_memory_occupation"));
        assert!(is_memory_resume("resume_memory_occupation"));
    }
}
