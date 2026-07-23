// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo backend for SGLang's native `sglang.runtime.v1` gRPC server.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;

use async_trait::async_trait;
use dynamo_backend_common::{
    AsyncEngineContext, DisaggregationMode, DynamoError, EngineConfig, GenerateContext, LLMEngine,
    LLMEngineOutput, LLMEngineOutputExt, LlmRegistration, ModelInput, PreprocessedRequest,
    WorkerConfig, usage,
};
use futures::stream::BoxStream;
use serde_json::Value;
use tokio::sync::OnceCell;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::args::{Args, TransportConfig, normalize_endpoint};
use crate::client::{self, Client, Discovery, Pool};
use crate::proto as pb;
use crate::protocol::{
    build_generate_request, disaggregated_params_to_json, map_typed_generate_response,
};

pub struct SglangSidecarEngine {
    endpoint: String,
    transport: TransportConfig,
    disaggregation_mode: DisaggregationMode,
    bootstrap_host: Option<String>,
    bootstrap_port: Option<u16>,
    pool: OnceCell<Pool>,
    cancel: CancellationToken,
}

impl SglangSidecarEngine {
    pub(crate) fn new(
        endpoint: impl Into<String>,
        transport: TransportConfig,
        disaggregation_mode: DisaggregationMode,
        bootstrap_host: Option<String>,
        bootstrap_port: Option<u16>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            transport,
            disaggregation_mode,
            bootstrap_host,
            bootstrap_port,
            pool: OnceCell::new(),
            cancel: CancellationToken::new(),
        }
    }

    pub fn from_args(argv: Option<Vec<String>>) -> Result<(Self, WorkerConfig), DynamoError> {
        let args = match argv {
            Some(args) => <Args as clap::Parser>::try_parse_from(args)
                .map_err(|err| client::invalid_arg(err.to_string()))?,
            None => <Args as clap::Parser>::parse(),
        };

        let endpoint = normalize_endpoint(&args.sglang_endpoint).map_err(client::invalid_arg)?;
        let transport = args.transport();
        let discovery = bootstrap_discover(&endpoint, &transport)?;
        let disaggregation_mode = discovery_mode(&discovery)?;
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
            "sglang sidecar bootstrapped native gRPC discovery"
        );

        let config = WorkerConfig {
            namespace: args.namespace,
            component: component_for_mode(disaggregation_mode).to_string(),
            endpoint: args.endpoint,
            endpoint_types: args.endpoint_types,
            custom_jinja_template: args.custom_jinja_template,
            disaggregation_mode,
            model_name: discovery.tokenizer_path.clone(),
            served_model_name: discovery.served_model_name.clone(),
            model_input: ModelInput::Tokens,
            reasoning_parser: discovery_string(&discovery.server_info, "reasoning_parser"),
            tool_call_parser: discovery_string(&discovery.server_info, "tool_call_parser"),
            ..Default::default()
        };

        Ok((
            Self::new(
                endpoint,
                transport,
                disaggregation_mode,
                bootstrap_host,
                bootstrap_port,
            ),
            config,
        ))
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
}

#[async_trait]
impl LLMEngine for SglangSidecarEngine {
    async fn start(&self, _worker_id: u64) -> Result<EngineConfig, DynamoError> {
        if self.pool.initialized() {
            return Err(client::engine_shutdown("sglang sidecar already started"));
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
        let connection_count = pool.len();
        self.pool
            .set(pool)
            .map_err(|_| client::engine_shutdown("sglang sidecar already started"))?;
        tracing::info!(
            model = %config.model,
            mode = ?self.disaggregation_mode,
            connections = connection_count,
            "sglang sidecar started"
        );
        Ok(config)
    }

    async fn generate(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
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
        let prefill_handoff = if self.disaggregation_mode.is_prefill() {
            grpc_request
                .disaggregated_params
                .as_ref()
                .map(disaggregated_params_to_json)
        } else {
            None
        };
        let expected_choices = if self.disaggregation_mode.is_prefill() {
            1_u32
        } else {
            u32::from(request.sampling_options.n.unwrap_or(1))
        };
        let cancel = self.cancel.clone();
        let is_prefill = self.disaggregation_mode.is_prefill();

        Ok(Box::pin(async_stream::stream! {
            if ctx.is_stopped() || cancel.is_cancelled() {
                for index in 0..expected_choices {
                    let mut output = LLMEngineOutput::cancelled()
                        .with_usage(usage(prompt_tokens, 0));
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
                response = grpc_client.typed_generate(grpc_request) => Some(response),
            };
            let Some(opened) = opened else {
                for index in 0..expected_choices {
                    let mut output = LLMEngineOutput::cancelled()
                        .with_usage(usage(prompt_tokens, 0));
                    output.index = Some(index);
                    yield Ok(output);
                }
                return;
            };
            let mut stream = match opened {
                Ok(response) => response.into_inner(),
                Err(status) => {
                    yield Err(client::typed_generate_status_to_dynamo(status));
                    return;
                }
            };

            let mut generated_by_choice = HashMap::<u32, u32>::new();
            let mut prompt_logprobs_by_choice = HashMap::<u32, Value>::new();
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
                                defer_prompt_logprobs_until_terminal(
                                    &mut output,
                                    index,
                                    true,
                                    &mut prompt_logprobs_by_choice,
                                );
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
                                defer_prompt_logprobs_until_terminal(
                                    &mut output,
                                    index,
                                    true,
                                    &mut prompt_logprobs_by_choice,
                                );
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
                                        "SGLang closed TypedGenerate after {}/{} terminal choices",
                                        terminal_choices.len(), expected_choices
                                    )));
                                }
                                break;
                            }
                            Err(status) => {
                                yield Err(client::typed_generate_status_to_dynamo(status));
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
                        let generated = generated_by_choice.entry(choice).or_default();
                        *generated = generated.saturating_add(delta_len);
                        let is_terminal = response.terminal.is_some();
                        let mut mapped = match map_typed_generate_response(
                            response,
                            ctx.id(),
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
                        defer_prompt_logprobs_until_terminal(
                            &mut mapped,
                            choice,
                            is_terminal,
                            &mut prompt_logprobs_by_choice,
                        );
                        if is_prefill && !is_terminal {
                            continue;
                        }
                        if is_prefill && is_terminal {
                            mapped.disaggregated_params = prefill_handoff.clone();
                        }
                        if is_terminal {
                            terminal_choices.insert(choice);
                        }
                        if !mapped.token_ids.is_empty() || is_terminal {
                            yield Ok(mapped);
                        }
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
        self.cancel.cancel();
        tracing::info!("sglang sidecar shutdown complete");
        Ok(())
    }
}

fn defer_prompt_logprobs_until_terminal(
    output: &mut LLMEngineOutput,
    choice: u32,
    is_terminal: bool,
    deferred: &mut HashMap<u32, Value>,
) {
    let prompt_logprobs = output
        .engine_data
        .as_mut()
        .and_then(Value::as_object_mut)
        .and_then(|data| data.remove("prompt_logprobs"));
    if let Some(prompt_logprobs) = prompt_logprobs {
        deferred.entry(choice).or_insert(prompt_logprobs);
    }
    if output
        .engine_data
        .as_ref()
        .and_then(Value::as_object)
        .is_some_and(serde_json::Map::is_empty)
    {
        output.engine_data = None;
    }
    if is_terminal && let Some(prompt_logprobs) = deferred.remove(&choice) {
        let data = output
            .engine_data
            .get_or_insert_with(|| Value::Object(serde_json::Map::new()))
            .as_object_mut()
            .expect("engine_data initialized as an object");
        data.insert("prompt_logprobs".to_string(), prompt_logprobs);
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
    match discovery
        .server_info
        .get("disaggregation_mode")
        .and_then(Value::as_str)
        .unwrap_or("null")
    {
        "null" | "agg" | "aggregated" => Ok(DisaggregationMode::Aggregated),
        "prefill" => Ok(DisaggregationMode::Prefill),
        "decode" => Ok(DisaggregationMode::Decode),
        mode => Err(client::protocol_error(format!(
            "unsupported SGLang disaggregation_mode `{mode}`"
        ))),
    }
}

fn component_for_mode(mode: DisaggregationMode) -> &'static str {
    if mode.is_prefill() {
        "prefill"
    } else {
        "backend"
    }
}

fn discovery_string(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|entry| !entry.is_empty())
        .map(str::to_string)
}

fn discovery_bootstrap_port(discovery: &Discovery) -> Result<Option<u16>, DynamoError> {
    client::json_u64(&discovery.server_info, "disaggregation_bootstrap_port")
        .map(|port| {
            u16::try_from(port).map_err(|_| {
                client::protocol_error(format!(
                    "SGLang disaggregation_bootstrap_port is out of range: {port}"
                ))
            })
        })
        .transpose()
        .and_then(|port| {
            port.filter(|port| *port != 0).map_or_else(
                || {
                    Err(client::protocol_error(
                        "prefill SGLang server did not report disaggregation_bootstrap_port",
                    ))
                },
                |port| Ok(Some(port)),
            )
        })
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
    let from_server = discovery
        .server_info
        .get("host")
        .and_then(Value::as_str)
        .filter(|host| is_routable_host(host));
    if let Some(host) = from_server {
        return Ok(Some(host.trim().to_string()));
    }
    let from_dist = discovery
        .server_info
        .get("dist_init_addr")
        .and_then(Value::as_str)
        .and_then(host_from_address)
        .filter(|host| is_routable_host(host));
    if let Some(host) = from_dist {
        return Ok(Some(host));
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

fn host_from_address(address: &str) -> Option<String> {
    let candidate = if address.contains("://") {
        address.to_string()
    } else {
        format!("tcp://{address}")
    };
    url::Url::parse(&candidate)
        .ok()
        .and_then(|url| url.host_str().map(str::to_string))
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
    let page_size = client::json_u32(&discovery.server_info, "page_size");
    let max_total_tokens = client::json_u64(&discovery.server_info, "max_total_num_tokens");
    let total_kv_blocks = match (max_total_tokens, page_size) {
        (Some(tokens), Some(page_size)) if page_size > 0 => {
            Some(tokens.saturating_add(u64::from(page_size) - 1) / u64::from(page_size))
        }
        _ => None,
    };
    let dp_size = client::json_u32(&discovery.server_info, "dp_size")
        .unwrap_or(1)
        .max(1);
    let max_num_seqs =
        client::json_u64(&discovery.server_info, "max_running_requests").map(|value| {
            if dp_size > 1 {
                value / u64::from(dp_size)
            } else {
                value
            }
        });
    let max_num_batched_tokens =
        client::json_u64(&discovery.server_info, "max_prefill_tokens").or(max_total_tokens);

    let enable_dp_attention = discovery
        .server_info
        .get("enable_dp_attention")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let (data_parallel_start_rank, data_parallel_size) = if enable_dp_attention && dp_size > 1 {
        // Native gRPC is exposed by the rank-zero frontend for the complete
        // multi-node SGLang endpoint, so one sidecar registers every DP rank.
        (Some(0), Some(dp_size))
    } else {
        (Some(0), Some(1))
    };

    if mode.is_prefill() && (bootstrap_host.is_none() || bootstrap_port.is_none()) {
        return Err(client::protocol_error(
            "prefill SGLang discovery did not provide a usable bootstrap address",
        ));
    }

    let mut runtime_data = HashMap::new();
    runtime_data.insert(
        "grpc_service".to_string(),
        Value::String("sglang.runtime.v1.SglangService".to_string()),
    );

    Ok(EngineConfig {
        model: discovery.model_path.clone(),
        served_model_name: discovery.served_model_name.clone(),
        runtime_data,
        llm: Some(LlmRegistration {
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
    use serde_json::json;

    use super::{
        DisaggregationMode, Discovery, build_engine_config, defer_prompt_logprobs_until_terminal,
        resolve_bootstrap_host_with_local,
    };

    fn discovery(server_info: serde_json::Value) -> Discovery {
        Discovery {
            model_path: "model".to_string(),
            tokenizer_path: "tokenizer".to_string(),
            served_model_name: None,
            max_model_len: None,
            model_info: json!({}),
            server_info,
        }
    }

    #[test]
    fn prompt_logprobs_are_deferred_to_a_matching_cancellation_terminal() {
        let mut deferred = std::collections::HashMap::new();
        let mut initial = dynamo_backend_common::LLMEngineOutput {
            engine_data: Some(json!({"prompt_logprobs": [{"1": {"logprob": -0.1}}]})),
            ..Default::default()
        };
        defer_prompt_logprobs_until_terminal(&mut initial, 1, false, &mut deferred);
        assert!(initial.engine_data.is_none());

        let mut terminal = dynamo_backend_common::LLMEngineOutput::cancelled();
        defer_prompt_logprobs_until_terminal(&mut terminal, 1, true, &mut deferred);
        assert_eq!(
            terminal.engine_data.unwrap()["prompt_logprobs"],
            json!([{"1": {"logprob": -0.1}}])
        );
        assert!(deferred.is_empty());
    }

    #[test]
    fn explicit_bootstrap_host_takes_precedence() {
        let host = resolve_bootstrap_host_with_local(
            Some("prefill.example"),
            "http://127.0.0.1:30001",
            &discovery(json!({"dist_init_addr": "10.0.0.1:20000"})),
            "10.0.0.2",
        )
        .unwrap();
        assert_eq!(host.as_deref(), Some("prefill.example"));
    }

    #[test]
    fn server_host_precedes_dist_init_addr() {
        let host = resolve_bootstrap_host_with_local(
            None,
            "http://127.0.0.1:30001",
            &discovery(json!({
                "host": "10.0.0.1",
                "dist_init_addr": "10.0.0.2:20000"
            })),
            "10.0.0.3",
        )
        .unwrap();
        assert_eq!(host.as_deref(), Some("10.0.0.1"));
    }

    #[test]
    fn wildcard_server_host_uses_dist_init_addr() {
        let host = resolve_bootstrap_host_with_local(
            None,
            "http://127.0.0.1:30001",
            &discovery(json!({
                "host": "0.0.0.0",
                "dist_init_addr": "10.0.0.1:20000"
            })),
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
            &discovery(json!({})),
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
            &discovery(json!({"dist_init_addr": "0.0.0.0:20000"})),
            "127.0.0.1",
        )
        .unwrap_err();
        assert!(error.to_string().contains("--bootstrap-host"));
    }

    #[test]
    fn multi_node_grpc_endpoint_registers_all_dp_ranks() {
        let config = build_engine_config(
            &discovery(json!({
                "dp_size": 16,
                "enable_dp_attention": true,
                "nnodes": 4,
                "node_rank": 0,
            })),
            DisaggregationMode::Decode,
            None,
            None,
        )
        .unwrap();
        let registration = config.llm.unwrap();

        assert_eq!(registration.data_parallel_start_rank, Some(0));
        assert_eq!(registration.data_parallel_size, Some(16));
    }
}
