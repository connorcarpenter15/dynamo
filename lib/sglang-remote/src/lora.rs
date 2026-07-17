// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamic LoRA lifecycle for the native SGLang sidecar.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use dynamo_backend_common::{DisaggregationMode, DynamoError};
use dynamo_llm::local_model::runtime_config::{DisaggregatedEndpoint, ModelRuntimeConfig};
use dynamo_llm::local_model::{LocalModel, LocalModelBuilder};
use dynamo_llm::lora::{LoRACache, LoRADownloader, LocalLoRASource, S3LoRASource};
use dynamo_llm::model_card::LoraInfo;
use dynamo_llm::model_type::{ModelInput, ModelType};
use dynamo_llm::worker_type::WorkerType;
use dynamo_runtime::component::Endpoint;
use serde_json::{Value, json};
use tokio::sync::{Mutex as AsyncMutex, OnceCell};

use crate::client::{self, Discovery, Pool};
use crate::proto as pb;

pub(crate) const UPDATES: [&str; 3] = ["load_lora", "unload_lora", "list_loras"];

#[derive(Clone, Debug)]
struct LoadedAdapter {
    path: String,
    id: Option<String>,
    pinned: bool,
}

pub(crate) struct LoraManager {
    downloader: Arc<LoRADownloader>,
    endpoint: OnceCell<Endpoint>,
    locks: Mutex<HashMap<String, Arc<AsyncMutex<()>>>>,
    loaded: AsyncMutex<HashMap<String, LoadedAdapter>>,
    discovery: OnceCell<Discovery>,
    mode: DisaggregationMode,
    endpoint_types: String,
}

impl LoraManager {
    pub(crate) fn new(
        mode: DisaggregationMode,
        endpoint_types: String,
    ) -> Result<Self, DynamoError> {
        let cache = LoRACache::from_env()
            .map_err(|error| client::invalid_arg(format!("LoRA cache configuration: {error}")))?;
        let mut sources: Vec<Arc<dyn dynamo_llm::lora::LoRASource>> =
            vec![Arc::new(LocalLoRASource::new())];
        if let Ok(source) = S3LoRASource::from_env() {
            // AWS_ENDPOINT/AWS_ALLOW_HTTP make the same source work with MinIO.
            sources.push(Arc::new(source));
        }
        Ok(Self {
            downloader: Arc::new(LoRADownloader::new(sources, cache)),
            endpoint: OnceCell::new(),
            locks: Mutex::new(HashMap::new()),
            loaded: AsyncMutex::new(HashMap::new()),
            discovery: OnceCell::new(),
            mode,
            endpoint_types,
        })
    }

    pub(crate) fn enabled(&self) -> bool {
        self.discovery
            .get()
            .is_some_and(|info| info.capacity.max_lora_adapters > 0)
    }

    pub(crate) fn set_endpoint(&self, endpoint: Endpoint) -> Result<(), DynamoError> {
        self.endpoint.set(endpoint).map_err(|_| {
            client::engine_shutdown("SGLang endpoint was handed to LoRA manager twice")
        })
    }

    pub(crate) fn set_discovery(&self, discovery: Discovery) -> Result<(), DynamoError> {
        self.discovery
            .set(discovery)
            .map_err(|_| client::engine_shutdown("SGLang discovery was initialized twice"))
    }

    fn operation_lock(&self, name: &str) -> Arc<AsyncMutex<()>> {
        let mut locks = self
            .locks
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        locks
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(AsyncMutex::new(())))
            .clone()
    }

    fn endpoint(&self) -> Result<&Endpoint, DynamoError> {
        self.endpoint
            .get()
            .ok_or_else(|| client::engine_shutdown("LoRA update arrived before endpoint handoff"))
    }

    fn discovery(&self) -> Result<&Discovery, DynamoError> {
        self.discovery
            .get()
            .ok_or_else(|| client::engine_shutdown("LoRA update arrived before runtime discovery"))
    }

    async fn list_server(
        &self,
        pool: &Pool,
        timeout: Duration,
    ) -> Result<Vec<pb::LoRaAdapter>, DynamoError> {
        let mut client = pool.control_client();
        match tokio::time::timeout(timeout, client.list_lo_r_as(pb::ListLoRAsRequest {})).await {
            Ok(Ok(response)) => Ok(response.into_inner().adapters),
            Ok(Err(status)) => Err(client::status_to_dynamo("ListLoRAs", status)),
            Err(_) => Err(client::cannot_connect(
                "ListLoRAs exceeded the configured timeout",
            )),
        }
    }

    async fn load_server(
        &self,
        pool: &Pool,
        timeout: Duration,
        name: &str,
        path: &str,
        pinned: bool,
        id: Option<String>,
    ) -> Result<pb::LoRaUpdateResponse, DynamoError> {
        let mut client = pool.control_client();
        match tokio::time::timeout(
            timeout,
            client.load_lo_ra(pb::LoadLoRaRequest {
                name: name.to_string(),
                path: path.to_string(),
                pinned,
                id,
            }),
        )
        .await
        {
            Ok(Ok(response)) => Ok(response.into_inner()),
            Ok(Err(status)) => Err(client::status_to_dynamo("LoadLoRA", status)),
            Err(_) => Err(client::cannot_connect(
                "LoadLoRA exceeded the configured timeout",
            )),
        }
    }

    async fn unload_server(
        &self,
        pool: &Pool,
        timeout: Duration,
        name: &str,
        id: Option<String>,
    ) -> Result<pb::LoRaUpdateResponse, DynamoError> {
        let mut client = pool.control_client();
        match tokio::time::timeout(
            timeout,
            client.unload_lo_ra(pb::UnloadLoRaRequest {
                name: name.to_string(),
                id,
            }),
        )
        .await
        {
            Ok(Ok(response)) => Ok(response.into_inner()),
            Ok(Err(status)) => Err(client::status_to_dynamo("UnloadLoRA", status)),
            Err(_) => Err(client::cannot_connect(
                "UnloadLoRA exceeded the configured timeout",
            )),
        }
    }

    async fn publish(&self, name: &str) -> Result<(), DynamoError> {
        let info = self.discovery()?;
        let endpoint = self.endpoint()?;
        let source = info.model_path.clone();
        let metadata_path = if std::fs::exists(&source)
            .map_err(|error| client::invalid_arg(format!("base model path: {error}")))?
        {
            PathBuf::from(&source)
        } else {
            LocalModel::fetch(&source, true).await.map_err(|error| {
                client::cannot_connect(format!(
                    "fetch base model metadata for LoRA discovery card: {error}"
                ))
            })?
        };
        let mut runtime_data = HashMap::new();
        runtime_data.insert(
            "grpc_service".to_string(),
            Value::String("sglang.runtime.v1.SglangService".to_string()),
        );
        runtime_data.insert(
            "protocol_revision".to_string(),
            Value::String(pb::PROTOCOL_REVISION.to_string()),
        );
        runtime_data.insert(
            "descriptor_sha256".to_string(),
            Value::String(pb::descriptor_sha256()),
        );
        let bootstrap = info.bootstrap.as_ref().map(|value| DisaggregatedEndpoint {
            bootstrap_host: Some(value.bootstrap_host.clone()),
            bootstrap_port: u16::try_from(value.bootstrap_port).ok(),
        });
        let runtime_config = ModelRuntimeConfig {
            context_length: u32::try_from(info.capacity.max_context_length).ok(),
            total_kv_blocks: (info.capacity.total_kv_blocks > 0)
                .then_some(info.capacity.total_kv_blocks),
            max_num_seqs: (info.capacity.max_running_requests > 0)
                .then_some(info.capacity.max_running_requests),
            max_num_batched_tokens: (info.capacity.max_total_tokens > 0)
                .then_some(info.capacity.max_total_tokens),
            data_parallel_size: info.dp_topology.local_size.max(1),
            data_parallel_start_rank: info.dp_topology.local_start_rank,
            tool_call_parser: info.tool_call_parser.clone(),
            reasoning_parser: info.reasoning_parser.clone(),
            disaggregated_endpoint: self.mode.is_prefill().then_some(bootstrap).flatten(),
            max_gpu_lora_count: (info.capacity.max_lora_adapters > 0)
                .then_some(info.capacity.max_lora_adapters),
            runtime_data,
            ..Default::default()
        };
        let mut builder = LocalModelBuilder::default();
        builder
            .model_path(metadata_path)
            .source_path(PathBuf::from(&source))
            .model_name(Some(name.to_string()))
            .kv_cache_block_size(u32::try_from(info.capacity.kv_block_size).ok())
            .runtime_config(runtime_config);
        let mut model = builder
            .build()
            .await
            .map_err(|error| client::protocol_error(format!("build LoRA model card: {error}")))?;
        let (model_type, worker_type, needs) = topology(self.mode, &self.endpoint_types)?;
        model
            .attach(
                endpoint,
                model_type,
                ModelInput::Tokens,
                Some(LoraInfo {
                    name: name.to_string(),
                    max_gpu_lora_count: (info.capacity.max_lora_adapters > 0)
                        .then_some(info.capacity.max_lora_adapters),
                }),
                Some(worker_type),
                needs,
            )
            .await
            .map_err(|error| client::protocol_error(format!("publish LoRA model card: {error}")))
    }

    async fn unpublish(&self, name: &str) -> Result<(), DynamoError> {
        LocalModel::detach_from_endpoint(self.endpoint()?, Some(name))
            .await
            .map_err(|error| client::protocol_error(format!("unpublish LoRA model card: {error}")))
    }

    /// Detach every model card published by this sidecar before its discovery
    /// connection is torn down. SGLang owns adapter memory, so shutdown does
    /// not unload adapters from the engine; it only prevents stale Dynamo
    /// registrations from surviving this worker instance.
    pub(crate) async fn cleanup(&self) -> Result<(), DynamoError> {
        let names = self.loaded.lock().await.keys().cloned().collect::<Vec<_>>();
        let mut failures = Vec::new();
        for name in names {
            let _guard = self.operation_lock(&name).lock_owned().await;
            match self.unpublish(&name).await {
                Ok(()) => {
                    self.loaded.lock().await.remove(&name);
                }
                Err(error) => failures.push(format!("{name}: {error}")),
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(client::protocol_error(format!(
                "failed to remove LoRA discovery cards during shutdown: {}",
                failures.join("; ")
            )))
        }
    }

    pub(crate) async fn execute(
        &self,
        pool: &Pool,
        timeout: Duration,
        update: &str,
        body: Value,
    ) -> Result<Value, DynamoError> {
        match update {
            "list_loras" => {
                let adapters = self.list_server(pool, timeout).await?;
                let published = self.loaded.lock().await;
                Ok(json!({
                    "status": "ok",
                    "adapters": adapters.into_iter().map(|adapter| {
                        let sidecar = published.get(&adapter.name);
                        json!({
                            "lora_name": adapter.name,
                            "path": adapter.path,
                            "id": adapter.id,
                            "pinned": adapter.pinned,
                            "discovery_published": sidecar.is_some(),
                            "sidecar_path": sidecar.map(|value| value.path.as_str()),
                            "sidecar_id": sidecar.and_then(|value| value.id.as_deref()),
                            "sidecar_pinned": sidecar.map(|value| value.pinned),
                        })
                    }).collect::<Vec<_>>()
                }))
            }
            "load_lora" => {
                let name = lora_name(&body)?;
                let _guard = self.operation_lock(&name).lock_owned().await;
                let info = self.discovery()?;
                if name == info.model_path
                    || info.served_model_name.as_deref() == Some(name.as_str())
                {
                    return Ok(error(format!(
                        "LoRA name `{name}` collides with the base model"
                    )));
                }
                let existing = self
                    .list_server(pool, timeout)
                    .await?
                    .into_iter()
                    .find(|adapter| adapter.name == name);
                if let Some(adapter) = existing {
                    if !self.loaded.lock().await.contains_key(&name) {
                        self.publish(&name).await?;
                        self.loaded.lock().await.insert(
                            name.clone(),
                            LoadedAdapter {
                                path: adapter.path,
                                id: adapter.id,
                                pinned: adapter.pinned,
                            },
                        );
                    }
                    return Ok(success(&name, "already loaded"));
                }
                let uri = body
                    .get("source")
                    .and_then(|source| source.get("uri"))
                    .or_else(|| body.get("path"))
                    .and_then(Value::as_str)
                    .ok_or_else(|| client::invalid_arg("load_lora requires source.uri"))?;
                let uri = normalize_local_uri(uri)?;
                let local_path = self
                    .downloader
                    .download_if_needed(&uri)
                    .await
                    .map_err(|error| client::cannot_connect(format!("download LoRA: {error}")))?;
                let local_path = local_path.to_string_lossy().into_owned();
                let pinned = body.get("pinned").and_then(Value::as_bool).unwrap_or(false);
                let id = body.get("id").and_then(Value::as_str).map(str::to_owned);
                let loaded = self
                    .load_server(pool, timeout, &name, &local_path, pinned, id.clone())
                    .await?;
                if !loaded.success {
                    return Ok(error(
                        loaded
                            .error_message
                            .unwrap_or_else(|| format!("SGLang failed to load `{name}`")),
                    ));
                }
                if let Err(publish_error) = self.publish(&name).await {
                    let rollback = self
                        .unload_server(pool, timeout, &name, id.clone())
                        .await
                        .map(|response| response.success)
                        .unwrap_or(false);
                    return Err(client::protocol_error(format!(
                        "LoRA `{name}` loaded but discovery publication failed: {publish_error}; SGLang rollback {}",
                        if rollback { "succeeded" } else { "failed" }
                    )));
                }
                self.loaded.lock().await.insert(
                    name.clone(),
                    LoadedAdapter {
                        path: local_path,
                        id,
                        pinned,
                    },
                );
                Ok(success(&name, "loaded"))
            }
            "unload_lora" => {
                let name = lora_name(&body)?;
                let _guard = self.operation_lock(&name).lock_owned().await;
                let adapter = self
                    .list_server(pool, timeout)
                    .await?
                    .into_iter()
                    .find(|adapter| adapter.name == name);
                let Some(adapter) = adapter else {
                    if self.loaded.lock().await.contains_key(&name) {
                        self.unpublish(&name).await?;
                        self.loaded.lock().await.remove(&name);
                    }
                    return Ok(success(&name, "already unloaded"));
                };
                self.unpublish(&name).await?;
                let unloaded = self
                    .unload_server(pool, timeout, &name, adapter.id.clone())
                    .await?;
                if !unloaded.success {
                    let republished = self.publish(&name).await.is_ok();
                    return Err(client::protocol_error(format!(
                        "SGLang failed to unload LoRA `{name}`: {}; discovery rollback {}",
                        unloaded.error_message.unwrap_or_default(),
                        if republished { "succeeded" } else { "failed" }
                    )));
                }
                self.loaded.lock().await.remove(&name);
                Ok(success(&name, "unloaded"))
            }
            _ => Ok(error(format!("unsupported engine update: {update}"))),
        }
    }
}

fn normalize_local_uri(uri: &str) -> Result<String, DynamoError> {
    if uri.starts_with("file://") || uri.starts_with("s3://") {
        return Ok(uri.to_string());
    }
    let path = PathBuf::from(uri);
    if path.is_absolute() {
        return Ok(format!("file://{}", path.display()));
    }
    Err(client::invalid_arg(
        "LoRA URI must use file:// or s3://, or be an absolute local path",
    ))
}

fn lora_name(body: &Value) -> Result<String, DynamoError> {
    body.get("lora_name")
        .or_else(|| body.get("name"))
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| client::invalid_arg("lora_name is required"))
}

fn success(name: &str, action: &str) -> Value {
    json!({
        "status": "ok",
        "success": true,
        "message": format!("LoRA adapter `{name}` {action}"),
        "lora_name": name,
    })
}

fn error(message: impl Into<String>) -> Value {
    json!({"status": "error", "success": false, "message": message.into()})
}

fn topology(
    mode: DisaggregationMode,
    endpoint_types: &str,
) -> Result<(ModelType, WorkerType, Vec<Vec<WorkerType>>), DynamoError> {
    match mode {
        DisaggregationMode::Prefill => Ok((
            ModelType::Prefill,
            WorkerType::Prefill,
            vec![vec![WorkerType::Decode]],
        )),
        DisaggregationMode::Decode => Ok((
            parse_model_type(endpoint_types)?,
            WorkerType::Decode,
            vec![vec![WorkerType::Prefill]],
        )),
        DisaggregationMode::Aggregated => Ok((
            parse_model_type(endpoint_types)?,
            WorkerType::Aggregated,
            Vec::new(),
        )),
        DisaggregationMode::Encode => Err(client::invalid_arg(
            "LoRA updates are unavailable on encode workers",
        )),
    }
}

fn parse_model_type(endpoint_types: &str) -> Result<ModelType, DynamoError> {
    let mut model_type = ModelType::empty();
    for value in endpoint_types
        .split(',')
        .map(|value| value.trim().to_ascii_lowercase())
    {
        model_type |= match value.as_str() {
            "chat" => ModelType::Chat,
            "completions" => ModelType::Completions,
            "embedding" | "embeddings" => ModelType::Embedding,
            "images" | "image" => ModelType::Images,
            "videos" | "video" => ModelType::Videos,
            "audios" | "audio" => ModelType::Audios,
            "tensor" => ModelType::TensorBased,
            "" => continue,
            unknown => {
                return Err(client::invalid_arg(format!(
                    "unknown endpoint type `{unknown}`"
                )));
            }
        };
    }
    if model_type.is_empty() {
        return Err(client::invalid_arg("endpoint_types cannot be empty"));
    }
    Ok(model_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_paths_are_normalized_but_relative_paths_are_rejected() {
        assert_eq!(
            normalize_local_uri("/models/a").unwrap(),
            "file:///models/a"
        );
        assert!(normalize_local_uri("models/a").is_err());
        assert_eq!(
            normalize_local_uri("s3://bucket/a").unwrap(),
            "s3://bucket/a"
        );
    }

    #[test]
    fn lora_topology_matches_base_worker_role() {
        let (model_type, worker, needs) =
            topology(DisaggregationMode::Prefill, "chat,completions").unwrap();
        assert_eq!(model_type, ModelType::Prefill);
        assert_eq!(worker, WorkerType::Prefill);
        assert_eq!(needs, vec![vec![WorkerType::Decode]]);
    }

    #[test]
    fn load_name_accepts_canonical_and_compatibility_fields() {
        assert_eq!(lora_name(&json!({"lora_name": "a"})).unwrap(), "a");
        assert_eq!(lora_name(&json!({"name": "b"})).unwrap(), "b");
        assert!(lora_name(&json!({})).is_err());
    }

    #[test]
    fn management_results_use_the_standard_status_envelope() {
        assert_eq!(
            success("adapter", "loaded"),
            json!({
                "status": "ok",
                "success": true,
                "message": "LoRA adapter `adapter` loaded",
                "lora_name": "adapter",
            })
        );
        assert_eq!(
            error("failed"),
            json!({"status": "error", "success": false, "message": "failed"})
        );
    }
}
