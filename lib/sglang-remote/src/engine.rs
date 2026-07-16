// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo backend for SGLang's native `sglang.runtime.v1` gRPC server.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, DisaggregationMode, DynamoError, EngineConfig, GenerateContext,
    KvEventSource, LLMEngine, LLMEngineOutput, LLMEngineOutputExt, LlmRegistration,
    MetricsBindings, MetricsCtx, ModelInput, PreprocessedRequest, RawEngine, WorkerConfig, usage,
};
use futures::stream::BoxStream;
use tokio::sync::{Mutex, OnceCell};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::admin;
use crate::args::{Args, TransportConfig, normalize_endpoint};
use crate::client::{self, Client, Discovery, Pool};
use crate::lora::LoraManager;
use crate::observability;
use crate::proto as pb;
use crate::protocol::{
    build_embed_request, build_generate_request, json_object_to_struct, map_embed_response,
    map_generate_response, prost_struct_to_json,
};

pub struct SglangRemoteEngine {
    endpoint: String,
    transport: TransportConfig,
    disaggregation_mode: DisaggregationMode,
    bootstrap_host: Option<String>,
    bootstrap_port: Option<u16>,
    runtime_kind: pb::RuntimeKind,
    pool: OnceCell<Pool>,
    discovery: OnceCell<Discovery>,
    endpoint_handle: OnceCell<dynamo_runtime::component::Endpoint>,
    endpoint_registered: AtomicBool,
    disruptive_lock: Mutex<()>,
    lora: LoraManager,
    cancel: CancellationToken,
}

impl SglangRemoteEngine {
    pub(crate) fn new(
        endpoint: impl Into<String>,
        transport: TransportConfig,
        disaggregation_mode: DisaggregationMode,
        bootstrap_host: Option<String>,
        bootstrap_port: Option<u16>,
        runtime_kind: pb::RuntimeKind,
        endpoint_types: String,
    ) -> Result<Self, DynamoError> {
        Ok(Self {
            endpoint: endpoint.into(),
            transport,
            disaggregation_mode,
            bootstrap_host,
            bootstrap_port,
            runtime_kind,
            pool: OnceCell::new(),
            discovery: OnceCell::new(),
            endpoint_handle: OnceCell::new(),
            endpoint_registered: AtomicBool::new(false),
            disruptive_lock: Mutex::new(()),
            lora: LoraManager::new(disaggregation_mode, endpoint_types)?,
            cancel: CancellationToken::new(),
        })
    }

    pub fn from_args(argv: Option<Vec<String>>) -> Result<(Self, WorkerConfig), DynamoError> {
        let args = match argv {
            Some(args) => <Args as clap::Parser>::try_parse_from(args),
            None => <Args as clap::Parser>::try_parse(),
        }
        .map_err(|err| client::invalid_arg(err.to_string()))?;

        let endpoint = normalize_endpoint(&args.sglang_endpoint).map_err(client::invalid_arg)?;
        let transport = args.transport();
        let discovery = bootstrap_discover(&endpoint, &transport)?;
        let disaggregation_mode = discovery_mode(&discovery)?;
        if discovery.runtime_kind != pb::RuntimeKind::Llm
            && disaggregation_mode != DisaggregationMode::Aggregated
        {
            return Err(client::protocol_error(
                "embedding and media runtimes must report the aggregated worker role",
            ));
        }
        let bootstrap_host = if disaggregation_mode.is_prefill() {
            resolve_bootstrap_host(args.bootstrap_host.as_deref(), &endpoint, &discovery)?
        } else {
            None
        };
        let bootstrap_port = if disaggregation_mode.is_prefill() {
            discovery_bootstrap_port(&discovery)?
        } else {
            None
        };

        tracing::info!(
            %endpoint,
            mode = ?disaggregation_mode,
            model = %discovery.model_path,
            "sglang remote backend bootstrapped native gRPC discovery"
        );

        let (model_input, endpoint_types, model_name) = match discovery.runtime_kind {
            pb::RuntimeKind::Llm => (
                ModelInput::Tokens,
                args.endpoint_types,
                discovery.tokenizer_path.clone(),
            ),
            pb::RuntimeKind::Embedding => {
                (ModelInput::Text, "embeddings".to_string(), String::new())
            }
            pb::RuntimeKind::Image => (ModelInput::Text, "images".to_string(), String::new()),
            pb::RuntimeKind::Video => (ModelInput::Text, "videos".to_string(), String::new()),
            pb::RuntimeKind::Unspecified => {
                return Err(client::protocol_error("SGLang runtime kind is unspecified"));
            }
        };
        let lora_endpoint_types = endpoint_types.clone();
        let config = WorkerConfig {
            namespace: args.namespace,
            component: component_for_mode(disaggregation_mode).to_string(),
            endpoint: args.endpoint,
            endpoint_types,
            custom_jinja_template: args.custom_jinja_template,
            disaggregation_mode,
            model_name,
            served_model_name: discovery.served_model_name.clone(),
            model_input,
            reasoning_parser: discovery.reasoning_parser.clone(),
            tool_call_parser: discovery.tool_call_parser.clone(),
            ..Default::default()
        };

        Ok((
            Self::new(
                endpoint,
                transport,
                disaggregation_mode,
                bootstrap_host,
                bootstrap_port,
                discovery.runtime_kind,
                lora_endpoint_types,
            )?,
            config,
        ))
    }

    pub fn runtime_kind(&self) -> pb::RuntimeKind {
        self.runtime_kind
    }

    async fn await_ready(&self, client: &mut Client, deadline: Instant) -> Result<(), DynamoError> {
        loop {
            let retry_message = match client::health_check(client, deadline).await {
                Ok(healthy) => {
                    if healthy {
                        return Ok(());
                    }
                    "SGLang reported unhealthy".to_string()
                }
                Err(error) => format!("HealthCheck RPC failed: {error}"),
            };
            if Instant::now() >= deadline {
                return Err(client::engine_shutdown(format!(
                    "SGLang did not become healthy within {:?}: {retry_message}",
                    self.transport.deadline
                )));
            }
            tokio::time::sleep_until((Instant::now() + self.transport.poll_interval).min(deadline))
                .await;
        }
    }

    fn pool(&self) -> Result<&Pool, DynamoError> {
        self.pool
            .get()
            .ok_or_else(|| client::engine_shutdown("SGLang control arrived before start"))
    }

    fn endpoint_handle(&self) -> Result<&dynamo_runtime::component::Endpoint, DynamoError> {
        self.endpoint_handle
            .get()
            .ok_or_else(|| client::engine_shutdown("SGLang endpoint handoff has not completed"))
    }

    async fn unregister_endpoint(&self) -> Result<bool, DynamoError> {
        if !self.endpoint_registered.swap(false, Ordering::AcqRel) {
            return Ok(false);
        }
        if let Err(error) = self.endpoint_handle()?.unregister_endpoint_instance().await {
            self.endpoint_registered.store(true, Ordering::Release);
            return Err(client::protocol_error(format!(
                "unregister worker before disruptive SGLang control: {error}"
            )));
        }
        Ok(true)
    }

    async fn register_endpoint(&self) -> Result<(), DynamoError> {
        if self.endpoint_registered.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        if let Err(error) = self.endpoint_handle()?.register_endpoint_instance().await {
            self.endpoint_registered.store(false, Ordering::Release);
            return Err(client::protocol_error(format!(
                "re-register worker after disruptive SGLang control: {error}"
            )));
        }
        Ok(())
    }
}

#[async_trait]
impl LLMEngine for SglangRemoteEngine {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        if self.pool.initialized() {
            return Err(client::engine_shutdown(
                "sglang remote backend already started",
            ));
        }

        let deadline = Instant::now() + self.transport.deadline;
        let pool = Pool::connect(
            &self.endpoint,
            &self.transport,
            self.transport.connections,
            deadline,
        )
        .await?;
        let mut control = pool.control_client();
        self.await_ready(&mut control, deadline).await?;
        let discovery = client::discover(&mut control, deadline).await?;
        let observed_mode = discovery_mode(&discovery)?;
        if observed_mode != self.disaggregation_mode {
            return Err(client::invalid_arg(format!(
                "SGLang role changed since bootstrap: registered as {:?}, now reports {:?}",
                self.disaggregation_mode, observed_mode
            )));
        }

        let config = build_engine_config(
            &discovery,
            self.disaggregation_mode,
            self.bootstrap_host.clone(),
            self.bootstrap_port,
        )?;
        self.lora.set_discovery(discovery.clone())?;
        self.discovery
            .set(discovery)
            .map_err(|_| client::engine_shutdown("SGLang discovery was initialized twice"))?;
        let connection_count = pool.len();
        self.pool
            .set(pool)
            .map_err(|_| client::engine_shutdown("sglang remote backend already started"))?;
        tracing::info!(
            model = %config.model,
            mode = ?self.disaggregation_mode,
            connections = connection_count,
            "sglang remote backend started"
        );
        Ok(config)
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        if self.runtime_kind != pb::RuntimeKind::Llm {
            return Err(client::invalid_arg(format!(
                "Generate is unavailable for {:?} runtimes; use the RawEngine endpoint",
                self.runtime_kind
            )));
        }
        let mut grpc_client = self
            .pool
            .get()
            .map(Pool::stream_client)
            .ok_or_else(|| client::engine_shutdown("generate called before start"))?;

        let prompt_tokens = u32::try_from(request.token_ids.len()).unwrap_or(u32::MAX);
        let return_tokens_as_ids = request
            .output_options
            .return_tokens_as_token_ids
            .unwrap_or(false);
        let grpc_request = build_generate_request(
            &request,
            ctx.id(),
            self.disaggregation_mode,
            self.bootstrap_host.as_deref(),
            self.bootstrap_port,
        )?;
        let expected_choices = if self.disaggregation_mode.is_prefill() {
            1_u32
        } else {
            u32::from(request.sampling_options.n.unwrap_or(1))
        };
        // Cloned decoded-media descriptors retain their source_storage Arc, keeping
        // NIXL registrations alive until SGLang has materialized the request.
        let external_buffer_guard = request.multi_modal_data.clone();
        let cancel = self.cancel.clone();
        let is_prefill = self.disaggregation_mode.is_prefill();

        Ok(Box::pin(async_stream::stream! {
            let _external_buffer_guard = external_buffer_guard;
            if ctx.is_stopped() || cancel.is_cancelled() {
                for index in 0..expected_choices {
                    let mut output = LLMEngineOutput::cancelled().with_usage(usage(prompt_tokens, 0));
                    output.index = Some(index);
                    yield Ok(output);
                }
                return;
            }

            tracing::debug!(request_id = %ctx.id(), "sending request to SGLang gRPC");
            let opened = tokio::select! {
                biased;
                _ = ctx.stopped() => None,
                _ = cancel.cancelled() => None,
                response = grpc_client.generate(grpc_request) => Some(response),
            };
            let Some(opened) = opened else {
                for index in 0..expected_choices {
                    let mut output = LLMEngineOutput::cancelled().with_usage(usage(prompt_tokens, 0));
                    output.index = Some(index);
                    yield Ok(output);
                }
                return;
            };
            let mut stream = match opened {
                Ok(response) => response.into_inner(),
                Err(status) => {
                    yield Err(client::status_to_dynamo("Generate", status));
                    return;
                }
            };

            let mut generated_by_choice = HashMap::<u32, u32>::new();
            let mut terminal_choices = std::collections::HashSet::<u32>::new();
            loop {
                tokio::select! {
                    biased;
                    _ = ctx.stopped() => {
                        for index in 0..expected_choices {
                            if terminal_choices.insert(index) {
                                let generated = generated_by_choice.get(&index).copied().unwrap_or(0);
                                let mut output = LLMEngineOutput::cancelled()
                                    .with_usage(usage(prompt_tokens, generated));
                                output.index = Some(index);
                                yield Ok(output);
                            }
                        }
                        break;
                    }
                    _ = cancel.cancelled() => {
                        for index in 0..expected_choices {
                            if terminal_choices.insert(index) {
                                let generated = generated_by_choice.get(&index).copied().unwrap_or(0);
                                let mut output = LLMEngineOutput::cancelled()
                                    .with_usage(usage(prompt_tokens, generated));
                                output.index = Some(index);
                                yield Ok(output);
                            }
                        }
                        break;
                    }
                    message = stream.message() => {
                        let response = match message {
                            Ok(Some(response)) => response,
                            Ok(None) => {
                                if terminal_choices.len() != expected_choices as usize {
                                    yield Err(client::engine_shutdown(format!(
                                        "SGLang closed Generate after {}/{} terminal choices",
                                        terminal_choices.len(), expected_choices
                                    )));
                                }
                                break;
                            }
                            Err(status) => {
                                yield Err(client::status_to_dynamo("Generate", status));
                                break;
                            }
                        };

                        let choice = match u32::try_from(response.choice_index) {
                            Ok(choice) if choice < expected_choices => choice,
                            _ => {
                                yield Err(client::protocol_error(format!(
                                    "SGLang returned choice index {} outside 0..{}",
                                    response.choice_index, expected_choices
                                )));
                                break;
                            }
                        };
                        if terminal_choices.contains(&choice) {
                            yield Err(client::protocol_error(format!(
                                "SGLang returned data after terminal for choice {choice}"
                            )));
                            break;
                        }
                        let delta_len = u32::try_from(response.delta_output_ids.len())
                            .unwrap_or(u32::MAX);
                        let generated = generated_by_choice
                            .entry(choice)
                            .or_default();
                        *generated = generated.saturating_add(delta_len);
                        let is_terminal = response.terminal.is_some();
                        let mapped = match map_generate_response(
                            response,
                            prompt_tokens,
                            *generated,
                            return_tokens_as_ids,
                        ) {
                            Ok(mapped) => mapped,
                            Err(error) => {
                                yield Err(error);
                                break;
                            }
                        };
                        if is_prefill && !is_terminal {
                            continue;
                        }
                        if is_terminal {
                            terminal_choices.insert(choice);
                        }
                        yield Ok(mapped);
                        if terminal_choices.len() == expected_choices as usize {
                            break;
                        }
                    }
                }
            }
        }))
    }

    async fn abort(&self, ctx: Arc<dyn AsyncEngineContext>) {
        let Some(mut grpc_client) = self.pool.get().map(Pool::control_client) else {
            return;
        };
        let request = pb::AbortRequest {
            rid: ctx.id().to_string(),
            abort_all: false,
        };
        if let Err(error) =
            client::abort(&mut grpc_client, request, self.transport.connect_timeout).await
        {
            tracing::debug!(
                request_id = ctx.id(),
                %error,
                "SGLang Abort RPC failed"
            );
        }
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        let lora_cleanup = self.lora.cleanup().await;
        self.cancel.cancel();
        tracing::info!("sglang remote backend shutdown complete");
        lora_cleanup
    }

    async fn kv_event_sources(&self) -> Result<Vec<KvEventSource>, DynamoError> {
        let discovery = self
            .discovery
            .get()
            .ok_or_else(|| client::engine_shutdown("KV event discovery requested before start"))?;
        observability::kv_event_sources(discovery)
    }

    async fn setup_metrics(&self, ctx: MetricsCtx<'_>) -> Result<MetricsBindings, DynamoError> {
        let discovery = self
            .discovery
            .get()
            .ok_or_else(|| client::engine_shutdown("metrics setup requested before start"))?;
        observability::setup_metrics(discovery, ctx, self.cancel.clone())
    }

    async fn supported_controls(&self) -> Result<Vec<String>, DynamoError> {
        Ok(admin::CONTROLS
            .iter()
            .map(|value| value.to_string())
            .collect())
    }

    async fn engine_control(
        &self,
        control: String,
        body: serde_json::Value,
    ) -> Result<serde_json::Value, DynamoError> {
        if !admin::CONTROLS.contains(&control.as_str()) {
            return Ok(serde_json::json!({
                "status": "error",
                "message": format!("unsupported engine control: {control}"),
            }));
        }
        if !admin::is_disruptive(&control) {
            let mut grpc = self.pool()?.control_client();
            return admin::execute(&mut grpc, self.transport.connect_timeout, &control, body).await;
        }

        let _guard = self.disruptive_lock.lock().await;
        if admin::is_memory_resume(&control) {
            if self.endpoint_registered.load(Ordering::Acquire) {
                return Ok(serde_json::json!({
                    "status": "ok", "success": true, "message": "Memory already resumed"
                }));
            }
            let mut grpc = self.pool()?.control_client();
            let response =
                admin::execute(&mut grpc, self.transport.connect_timeout, &control, body).await?;
            if response.get("success").and_then(serde_json::Value::as_bool) == Some(true) {
                self.register_endpoint().await?;
            }
            return Ok(response);
        }

        let was_registered = self.unregister_endpoint().await?;
        if admin::is_memory_release(&control) && !was_registered {
            return Ok(serde_json::json!({
                "status": "ok", "success": true, "message": "Memory already released"
            }));
        }
        let mut grpc = self.pool()?.control_client();
        let result =
            admin::execute(&mut grpc, self.transport.connect_timeout, &control, body).await;
        if admin::is_memory_release(&control) {
            if result.is_err()
                || result
                    .as_ref()
                    .ok()
                    .and_then(|value| value.get("success"))
                    .and_then(serde_json::Value::as_bool)
                    != Some(true)
            {
                self.register_endpoint().await?;
            }
            return result;
        }
        let registration = if was_registered {
            self.register_endpoint().await
        } else {
            Ok(())
        };
        match (result, registration) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(operation), Ok(())) => Err(operation),
            (Ok(_), Err(registration)) => Err(registration),
            (Err(operation), Err(registration)) => Err(client::protocol_error(format!(
                "SGLang control failed: {operation}; endpoint recovery failed: {registration}"
            ))),
        }
    }

    async fn supported_updates(&self) -> Result<Vec<String>, DynamoError> {
        if self.lora.enabled() {
            Ok(crate::lora::UPDATES
                .iter()
                .map(|value| value.to_string())
                .collect())
        } else {
            Ok(Vec::new())
        }
    }

    async fn engine_update(
        &self,
        update: String,
        body: serde_json::Value,
    ) -> Result<serde_json::Value, DynamoError> {
        if !self.lora.enabled() {
            return Ok(serde_json::json!({
                "status": "error",
                "message": "SGLang did not advertise dynamic LoRA capacity",
            }));
        }
        self.lora
            .execute(self.pool()?, self.transport.connect_timeout, &update, body)
            .await
    }

    async fn on_endpoint_ready(
        &self,
        endpoint: dynamo_runtime::component::Endpoint,
    ) -> Result<(), DynamoError> {
        self.lora.set_endpoint(endpoint.clone())?;
        self.endpoint_handle
            .set(endpoint)
            .map_err(|_| client::engine_shutdown("SGLang endpoint was handed off twice"))?;
        // Worker calls this immediately before attaching the base model and
        // registering the endpoint. Control routes are installed afterward.
        self.endpoint_registered.store(true, Ordering::Release);
        Ok(())
    }
}

#[async_trait]
impl RawEngine for SglangRemoteEngine {
    async fn start(&self, worker_id: u64) -> Result<EngineConfig, DynamoError> {
        if self.runtime_kind == pb::RuntimeKind::Llm {
            return Err(client::invalid_arg(
                "LLM runtimes must be registered through LLMEngine",
            ));
        }
        <Self as LLMEngine>::start(self, worker_id).await
    }

    async fn generate(
        &self,
        request: serde_json::Value,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<serde_json::Value, DynamoError>>, DynamoError> {
        let mut grpc_client = self
            .pool
            .get()
            .map(Pool::stream_client)
            .ok_or_else(|| client::engine_shutdown("generate called before start"))?;
        let runtime_kind = self.runtime_kind;
        let request_id = ctx.id().to_string();
        let model = request
            .get("model")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string();
        let embed_request = (runtime_kind == pb::RuntimeKind::Embedding)
            .then(|| build_embed_request(&request, &request_id))
            .transpose()?;
        let media_request = matches!(
            runtime_kind,
            pb::RuntimeKind::Image | pb::RuntimeKind::Video
        )
        .then(|| {
            let mut trace_headers = HashMap::new();
            dynamo_runtime::logging::inject_trace_headers_into_map(&mut trace_headers);
            Ok::<_, DynamoError>(pb::MediaGenerateRequest {
                request: Some(json_object_to_struct(&request)?),
                trace_headers,
            })
        })
        .transpose()?;
        let cancel = self.cancel.clone();

        Ok(Box::pin(async_stream::stream! {
            if ctx.is_stopped() || cancel.is_cancelled() {
                return;
            }
            let response = match runtime_kind {
                pb::RuntimeKind::Embedding => {
                    let rpc = grpc_client.embed(embed_request.expect("built for embedding"));
                    tokio::select! {
                        biased;
                        _ = ctx.stopped() => return,
                        _ = cancel.cancelled() => return,
                        response = rpc => match response {
                            Ok(response) => map_embed_response(response.into_inner(), &model),
                            Err(status) => Err(client::status_to_dynamo("Embed", status)),
                        },
                    }
                }
                pb::RuntimeKind::Image | pb::RuntimeKind::Video => {
                    let rpc = grpc_client.media_generate(media_request.expect("built for media"));
                    tokio::select! {
                        biased;
                        _ = ctx.stopped() => return,
                        _ = cancel.cancelled() => return,
                        response = rpc => match response {
                            Ok(response) => {
                                let response = response.into_inner();
                                if !(200..300).contains(&response.status_code) {
                                    Err(client::protocol_error(format!(
                                        "MediaGenerate returned HTTP-compatible status {}: {}",
                                        response.status_code,
                                        response.response.map(prost_struct_to_json).unwrap_or_default()
                                    )))
                                } else {
                                    Ok(response.response.map(prost_struct_to_json).unwrap_or_default())
                                }
                            }
                            Err(status) => Err(client::status_to_dynamo("MediaGenerate", status)),
                        },
                    }
                }
                pb::RuntimeKind::Llm | pb::RuntimeKind::Unspecified => Err(client::invalid_arg(
                    "RawEngine dispatch requires embedding, image, or video runtime",
                )),
            };
            yield response;
        }))
    }

    async fn abort(&self, ctx: Arc<dyn AsyncEngineContext>) {
        <Self as LLMEngine>::abort(self, ctx).await;
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        <Self as LLMEngine>::cleanup(self).await
    }
}

fn bootstrap_discover(
    endpoint: &str,
    transport: &TransportConfig,
) -> Result<Discovery, DynamoError> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| client::engine_shutdown(format!("bootstrap runtime: {err}")))?;
    runtime.block_on(async {
        let deadline = Instant::now() + transport.deadline;
        let mut grpc_client = client::connect(endpoint, transport, deadline).await?;
        client::discover(&mut grpc_client, deadline).await
    })
}

fn discovery_mode(discovery: &Discovery) -> Result<DisaggregationMode, DynamoError> {
    match discovery.worker_role {
        pb::WorkerRole::Aggregated => Ok(DisaggregationMode::Aggregated),
        pb::WorkerRole::Prefill => Ok(DisaggregationMode::Prefill),
        pb::WorkerRole::Decode => Ok(DisaggregationMode::Decode),
        pb::WorkerRole::Unspecified => {
            Err(client::protocol_error("SGLang worker role is unspecified"))
        }
    }
}

fn component_for_mode(mode: DisaggregationMode) -> &'static str {
    if mode.is_prefill() {
        "prefill"
    } else {
        "backend"
    }
}

fn discovery_bootstrap_port(discovery: &Discovery) -> Result<Option<u16>, DynamoError> {
    let Some(bootstrap) = discovery.bootstrap.as_ref() else {
        return Err(client::protocol_error(
            "prefill SGLang server did not report bootstrap information",
        ));
    };
    let port = u16::try_from(bootstrap.bootstrap_port).map_err(|_| {
        client::protocol_error(format!(
            "SGLang bootstrap_port is out of range: {}",
            bootstrap.bootstrap_port
        ))
    })?;
    if port == 0 {
        return Err(client::protocol_error(
            "prefill SGLang server reported bootstrap_port 0",
        ));
    }
    Ok(Some(port))
}

fn resolve_bootstrap_host(
    explicit: Option<&str>,
    endpoint: &str,
    discovery: &Discovery,
) -> Result<Option<String>, DynamoError> {
    let local_host = dynamo_runtime::utils::local_ip_for_advertise();
    resolve_bootstrap_host_with_local(explicit, endpoint, discovery, &local_host)
}

fn resolve_bootstrap_host_with_local(
    explicit: Option<&str>,
    endpoint: &str,
    discovery: &Discovery,
    local_host: &str,
) -> Result<Option<String>, DynamoError> {
    if let Some(host) = explicit.filter(|host| !host.trim().is_empty()) {
        return Ok(Some(host.trim().to_string()));
    }
    let from_discovery = discovery
        .bootstrap
        .as_ref()
        .map(|bootstrap| bootstrap.bootstrap_host.as_str())
        .filter(|host| is_routable_host(host));
    if let Some(host) = from_discovery {
        return Ok(Some(host.to_string()));
    }
    if is_routable_host(local_host) {
        return Ok(Some(local_host.to_string()));
    }
    let from_endpoint = url::Url::parse(endpoint)
        .ok()
        .and_then(|url| url.host_str().map(str::to_string))
        .filter(|host| is_routable_host(host));
    from_endpoint.map(Some).ok_or_else(|| {
        client::invalid_arg(
            "could not derive a routable prefill bootstrap host; set --bootstrap-host",
        )
    })
}

fn is_routable_host(host: &str) -> bool {
    let host = host.trim().trim_matches(&['[', ']'][..]);
    if host.is_empty()
        || host.eq_ignore_ascii_case("localhost")
        || host.to_ascii_lowercase().ends_with(".localhost")
    {
        return false;
    }
    host.parse::<IpAddr>()
        .map(|address| !address.is_loopback() && !address.is_unspecified())
        .unwrap_or(true)
}

fn build_engine_config(
    discovery: &Discovery,
    mode: DisaggregationMode,
    bootstrap_host: Option<String>,
    bootstrap_port: Option<u16>,
) -> Result<EngineConfig, DynamoError> {
    let page_size = (discovery.capacity.kv_block_size > 0)
        .then(|| u32::try_from(discovery.capacity.kv_block_size).ok())
        .flatten();
    let total_kv_blocks =
        (discovery.capacity.total_kv_blocks > 0).then_some(discovery.capacity.total_kv_blocks);
    let max_num_seqs = (discovery.capacity.max_running_requests > 0)
        .then_some(discovery.capacity.max_running_requests);
    let max_num_batched_tokens =
        (discovery.capacity.max_total_tokens > 0).then_some(discovery.capacity.max_total_tokens);
    let data_parallel_start_rank = Some(discovery.dp_topology.local_start_rank);
    let data_parallel_size = Some(discovery.dp_topology.local_size.max(1));

    if mode.is_prefill() && (bootstrap_host.is_none() || bootstrap_port.is_none()) {
        return Err(client::protocol_error(
            "prefill SGLang discovery did not provide a usable bootstrap address",
        ));
    }

    let mut runtime_data = HashMap::new();
    runtime_data.insert(
        "grpc_service".to_string(),
        serde_json::Value::String("sglang.runtime.v1.SglangService".to_string()),
    );
    runtime_data.insert(
        "protocol_revision".to_string(),
        serde_json::Value::String(pb::PROTOCOL_REVISION.to_string()),
    );
    runtime_data.insert(
        "descriptor_sha256".to_string(),
        serde_json::Value::String(pb::descriptor_sha256()),
    );

    Ok(EngineConfig {
        model: discovery.model_path.clone(),
        served_model_name: discovery.served_model_name.clone(),
        runtime_data,
        llm: (discovery.runtime_kind == pb::RuntimeKind::Llm).then_some(LlmRegistration {
            context_length: discovery.max_model_len,
            kv_cache_block_size: page_size,
            total_kv_blocks,
            max_num_seqs,
            max_num_batched_tokens,
            data_parallel_size,
            data_parallel_start_rank,
            bootstrap_host: mode.is_prefill().then_some(bootstrap_host).flatten(),
            bootstrap_port: mode.is_prefill().then_some(bootstrap_port).flatten(),
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::{Discovery, resolve_bootstrap_host_with_local};
    use crate::proto as pb;

    fn discovery(bootstrap_host: Option<&str>) -> Discovery {
        Discovery {
            model_path: "model".to_string(),
            tokenizer_path: "tokenizer".to_string(),
            served_model_name: None,
            max_model_len: None,
            runtime_kind: pb::RuntimeKind::Llm,
            worker_role: pb::WorkerRole::Prefill,
            capacity: Default::default(),
            dp_topology: Default::default(),
            bootstrap: bootstrap_host.map(|host| pb::DisaggregatedParams {
                bootstrap_host: host.to_string(),
                bootstrap_port: 5001,
                bootstrap_room: 0,
                ..Default::default()
            }),
            observability: Vec::new(),
            reasoning_parser: None,
            tool_call_parser: None,
            weight_version: None,
        }
    }

    #[test]
    fn explicit_bootstrap_host_takes_precedence() {
        let host = resolve_bootstrap_host_with_local(
            Some("prefill.example"),
            "http://127.0.0.1:30001",
            &discovery(Some("10.0.0.1")),
            "10.0.0.2",
        )
        .unwrap();
        assert_eq!(host.as_deref(), Some("prefill.example"));
    }

    #[test]
    fn discovered_bootstrap_host_precedes_local_address() {
        let host = resolve_bootstrap_host_with_local(
            None,
            "http://127.0.0.1:30001",
            &discovery(Some("10.0.0.1")),
            "10.0.0.2",
        )
        .unwrap();
        assert_eq!(host.as_deref(), Some("10.0.0.1"));
    }

    #[test]
    fn loopback_endpoint_uses_routable_local_address() {
        let host = resolve_bootstrap_host_with_local(
            None,
            "http://127.0.0.1:30001",
            &discovery(None),
            "10.0.0.2",
        )
        .unwrap();
        assert_eq!(host.as_deref(), Some("10.0.0.2"));
    }

    #[test]
    fn loopback_only_discovery_requires_override() {
        let error = resolve_bootstrap_host_with_local(
            None,
            "http://localhost:30001",
            &discovery(Some("0.0.0.0")),
            "127.0.0.1",
        )
        .unwrap_err();
        assert!(error.to_string().contains("--bootstrap-host"));
    }
}
