// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed request lowering and response conversion for SGLang's native gRPC protocol.

use std::collections::HashMap;

use base64::Engine as _;
use dynamo_backend_common::{
    DisaggregationMode, DynamoError, LLMEngineOutput, LLMEngineOutputExt, MultimodalData,
    PreprocessedRequest, StopReason, TopLogprob, usage,
};
use prost_types::{ListValue, Struct, Value as ProstValue, value::Kind};
use serde_json::{Map, Value};

use crate::client;
use crate::proto as pb;

pub(crate) fn build_generate_request(
    request: &PreprocessedRequest,
    request_id: &str,
    mode: DisaggregationMode,
    bootstrap_host: Option<&str>,
    bootstrap_port: Option<u16>,
) -> Result<pb::GenerateRequest, DynamoError> {
    validate_request(request)?;
    let input = if let Some(serialized) = request.prompt_embeds.as_ref() {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(serialized)
            .map_err(|err| client::invalid_arg(format!("invalid prompt_embeds base64: {err}")))?;
        Some(pb::generate_request::Input::InputEmbeds(pb::Tensor {
            dtype: pb::TensorDataType::Unspecified.into(),
            shape: Vec::new(),
            strides: Vec::new(),
            storage: Some(pb::tensor::Storage::Serialized(pb::SerializedTensor {
                format: "pytorch".to_string(),
                data: bytes,
            })),
        }))
    } else {
        Some(pb::generate_request::Input::InputIds(pb::TokenIds {
            values: request
                .token_ids
                .iter()
                .map(|token| {
                    i32::try_from(*token).map_err(|_| {
                        client::invalid_arg(format!("token id {token} does not fit in i32"))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        }))
    };

    let max_new_tokens = if mode.is_prefill() {
        Some(1)
    } else {
        request
            .stop_conditions
            .max_tokens
            .map(i32::try_from)
            .transpose()
            .map_err(|_| client::invalid_arg("max_tokens does not fit in i32"))?
    };
    let min_new_tokens = if mode.is_prefill() {
        None
    } else {
        request
            .stop_conditions
            .min_tokens
            .map(i32::try_from)
            .transpose()
            .map_err(|_| client::invalid_arg("min_tokens does not fit in i32"))?
    };
    let include_strings = request
        .sampling_options
        .include_stop_str_in_output
        .unwrap_or(false);
    let string_stops = request
        .stop_conditions
        .stop
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|value| pb::StringStop {
            value: value.clone(),
            include_in_output: include_strings,
        })
        .collect();
    let mut token_stops = Vec::new();
    let mut seen_tokens = HashMap::new();
    for (tokens, include_in_output) in [
        (request.stop_conditions.stop_token_ids.as_ref(), false),
        (
            request.stop_conditions.stop_token_ids_hidden.as_ref(),
            false,
        ),
        (
            request.stop_conditions.stop_token_ids_visible.as_ref(),
            true,
        ),
    ] {
        for token in tokens.into_iter().flatten() {
            let token_id = i32::try_from(*token).map_err(|_| {
                client::invalid_arg(format!("stop token id {token} does not fit in i32"))
            })?;
            if let Some(previous) = seen_tokens.insert(token_id, include_in_output) {
                if previous != include_in_output {
                    return Err(client::invalid_arg(format!(
                        "stop token id {token_id} is configured as both visible and hidden"
                    )));
                }
                continue;
            }
            token_stops.push(pb::TokenStop {
                token_id,
                include_in_output,
            });
        }
    }
    let guided_decoding = request
        .sampling_options
        .guided_decoding
        .as_ref()
        .map(guided_decoding_to_proto)
        .transpose()?;
    let require_reasoning = request
        .extra_args
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|extra| {
            extra
                .get("require_reasoning")
                .or_else(|| extra.get("reasoning"))
        })
        .and_then(Value::as_bool);
    let sampling_params = pb::SamplingParams {
        temperature: request.sampling_options.temperature,
        top_p: request.sampling_options.top_p,
        top_k: request.sampling_options.top_k,
        min_p: request.sampling_options.min_p,
        frequency_penalty: request.sampling_options.frequency_penalty,
        presence_penalty: request.sampling_options.presence_penalty,
        repetition_penalty: request.sampling_options.repetition_penalty,
        max_new_tokens,
        min_new_tokens,
        string_stops,
        token_stops,
        ignore_eos: request.stop_conditions.ignore_eos,
        n: Some(if mode.is_prefill() {
            1
        } else {
            i32::from(request.sampling_options.n.unwrap_or(1))
        }),
        seed: request.sampling_options.seed,
        guided_decoding,
        require_reasoning,
        max_thinking_tokens: request
            .stop_conditions
            .max_thinking_tokens
            .map(i32::try_from)
            .transpose()
            .map_err(|_| client::invalid_arg("max_thinking_tokens does not fit in i32"))?,
    };

    let output_options = &request.output_options;
    let return_logprobs = !mode.is_prefill()
        && (output_options.logprobs.is_some() || output_options.prompt_logprobs.is_some());
    let top_logprobs = if mode.is_prefill() {
        0
    } else {
        i32::try_from(
            output_options
                .logprobs
                .unwrap_or(0)
                .max(output_options.prompt_logprobs.unwrap_or(0)),
        )
        .map_err(|_| client::invalid_arg("requested logprobs does not fit in i32"))?
    };
    let logprob_options = pb::LogprobOptions {
        return_logprobs,
        top_logprobs,
        prompt_logprob_start: (!mode.is_prefill() && output_options.prompt_logprobs.is_some())
            .then_some(0),
        token_ids: Vec::new(),
        return_text: !output_options.return_tokens_as_token_ids.unwrap_or(false),
        return_routed_experts: request
            .annotations
            .iter()
            .any(|annotation| annotation == "routed_experts"),
        routed_experts_start: 0,
        return_prompt_token_ids: false,
    };
    let multimodal_inputs = multimodal_inputs(request)?;
    let processor_options = request
        .mm_processor_kwargs
        .as_ref()
        .map(json_object_to_struct)
        .transpose()?;
    let use_audio_in_video = request
        .mm_processor_kwargs
        .as_ref()
        .and_then(|value| value.get("use_audio_in_video"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let routed_dp_rank = request
        .routing
        .as_ref()
        .and_then(|routing| routing.dp_rank)
        .map(i32::try_from)
        .transpose()
        .map_err(|_| client::invalid_arg("routed dp_rank does not fit in i32"))?;
    let priority = request
        .routing
        .as_ref()
        .and_then(|routing| routing.priority)
        .unwrap_or_default();
    let lora_path = request
        .routing
        .as_ref()
        .and_then(|routing| routing.lora_name.clone());
    let mut trace_headers = HashMap::new();
    dynamo_runtime::logging::inject_trace_headers_into_map(&mut trace_headers);

    Ok(pb::GenerateRequest {
        input,
        sampling_params: Some(sampling_params),
        logprob_options: Some(logprob_options),
        multimodal_inputs,
        multimodal_processor_options: processor_options,
        use_audio_in_video,
        stream: true,
        rid: Some(request_id.to_string()),
        lora_path,
        lora_id: None,
        routing_key: request.mdc_sum.clone(),
        routed_dp_rank,
        priority,
        trace_headers,
        session_id: None,
        disaggregated_params: resolve_disaggregated_params(
            request,
            mode,
            bootstrap_host,
            bootstrap_port,
        )?,
    })
}

fn validate_request(request: &PreprocessedRequest) -> Result<(), DynamoError> {
    if request.token_ids.is_empty() && request.prompt_embeds.is_none() {
        return Err(client::invalid_arg(
            "token_ids must not be empty unless prompt_embeds is provided",
        ));
    }
    let n = request.sampling_options.n.unwrap_or(1);
    if request.sampling_options.best_of.unwrap_or(n) != n {
        return Err(client::invalid_arg(
            "best_of is unsupported unless it equals n; SGLang does not implement beam-search selection",
        ));
    }
    if request.sampling_options.n == Some(0) {
        return Err(client::invalid_arg("n must be greater than zero"));
    }
    if request.sampling_options.use_beam_search.unwrap_or(false) {
        return Err(client::invalid_arg(
            "beam search is not supported by SGLang",
        ));
    }
    if let Some(penalty) = request.sampling_options.length_penalty
        && (penalty - 1.0).abs() > f32::EPSILON
    {
        return Err(client::invalid_arg(
            "non-default length_penalty requires beam search, which SGLang does not support",
        ));
    }
    Ok(())
}

fn guided_decoding_to_proto(
    guided: &dynamo_backend_common::GuidedDecodingOptions,
) -> Result<pb::GuidedDecoding, DynamoError> {
    use pb::guided_decoding::Constraint;
    let constraints = [
        guided
            .json
            .as_ref()
            .map(|value| Constraint::JsonSchema(json_value_to_string(value))),
        guided.regex.clone().map(Constraint::Regex),
        guided.grammar.clone().map(Constraint::Ebnf),
        guided.choice.as_ref().map(|values| {
            Constraint::Choice(pb::ChoiceConstraint {
                values: values.clone(),
            })
        }),
        guided
            .structural_tag
            .as_ref()
            .map(|value| Constraint::StructuralTag(json_value_to_string(value))),
    ];
    let mut present = constraints.into_iter().flatten();
    let constraint = present.next();
    if present.next().is_some() {
        return Err(client::invalid_arg(
            "guided decoding accepts exactly one of json, regex, grammar, choice, or structural_tag",
        ));
    }
    Ok(pb::GuidedDecoding {
        constraint,
        backend: guided.backend.clone(),
        whitespace_pattern: guided.whitespace_pattern.clone(),
    })
}

fn multimodal_inputs(
    request: &PreprocessedRequest,
) -> Result<Vec<pb::MultimodalInput>, DynamoError> {
    let Some(media) = request.multi_modal_data.as_ref() else {
        return Ok(Vec::new());
    };
    let hashes = request
        .extra_args
        .as_ref()
        .and_then(|value| value.get("mm_hashes"))
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .map(|value| {
                    value.as_str().map(str::to_string).ok_or_else(|| {
                        client::invalid_arg("extra_args.mm_hashes entries must be strings")
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let count: usize = media.values().map(Vec::len).sum();
    if let Some(hashes) = hashes.as_ref()
        && hashes.len() != count
    {
        return Err(client::invalid_arg(format!(
            "multimodal routing hash count {} does not match media count {count}",
            hashes.len()
        )));
    }
    let mut hash_index = 0;
    let mut mapped = Vec::with_capacity(count);
    for key in ["image_url", "video_url", "audio_url"] {
        let Some(items) = media.get(key) else {
            continue;
        };
        let modality = match key {
            "image_url" => pb::Modality::Image,
            "video_url" => pb::Modality::Video,
            "audio_url" => pb::Modality::Audio,
            _ => unreachable!(),
        };
        for item in items {
            let source = match item {
                MultimodalData::Url(url) => {
                    pb::multimodal_input::Source::Url(url.as_str().to_string())
                }
                MultimodalData::RawUrl(url) => pb::multimodal_input::Source::Url(url.clone()),
                MultimodalData::Decoded(descriptor) => {
                    let value = serde_json::to_value(descriptor).map_err(|err| {
                        client::invalid_arg(format!("failed to serialize decoded media: {err}"))
                    })?;
                    let shape = value
                        .get("shape")
                        .and_then(Value::as_array)
                        .ok_or_else(|| client::invalid_arg("decoded media shape is missing"))?
                        .iter()
                        .map(|dim| {
                            dim.as_i64().ok_or_else(|| {
                                client::invalid_arg("decoded media shape must contain integers")
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let metadata = serde_json::to_vec(
                        value
                            .get("nixl_metadata")
                            .ok_or_else(|| client::invalid_arg("NIXL metadata is missing"))?,
                    )
                    .map_err(|err| client::invalid_arg(err.to_string()))?;
                    let descriptor = serde_json::to_vec(
                        value
                            .get("nixl_descriptor")
                            .ok_or_else(|| client::invalid_arg("NIXL descriptor is missing"))?,
                    )
                    .map_err(|err| client::invalid_arg(err.to_string()))?;
                    let length = value
                        .get("nixl_descriptor")
                        .and_then(|value| value.get("size"))
                        .and_then(Value::as_u64);
                    pb::multimodal_input::Source::DecodedTensor(pb::Tensor {
                        dtype: pb::TensorDataType::Uint8.into(),
                        shape,
                        strides: Vec::new(),
                        storage: Some(pb::tensor::Storage::External(pb::ExternalBuffer {
                            transport: Some(pb::external_buffer::Transport::Nixl(pb::NixlBuffer {
                                metadata,
                                descriptor,
                                agent_name: None,
                                length,
                            })),
                        })),
                    })
                }
            };
            mapped.push(pb::MultimodalInput {
                modality: modality.into(),
                source: Some(source),
                mime_type: None,
                routing_hash: hashes
                    .as_ref()
                    .and_then(|hashes| hashes.get(hash_index).cloned()),
            });
            hash_index += 1;
        }
    }
    if mapped.len() != count {
        let unsupported = media
            .keys()
            .filter(|key| !matches!(key.as_str(), "image_url" | "video_url" | "audio_url"))
            .cloned()
            .collect::<Vec<_>>();
        return Err(client::invalid_arg(format!(
            "unsupported multimodal keys: {}",
            unsupported.join(", ")
        )));
    }
    Ok(mapped)
}

pub(crate) fn json_object_to_struct(value: &Value) -> Result<Struct, DynamoError> {
    let object = value
        .as_object()
        .ok_or_else(|| client::invalid_arg("multimodal processor options must be an object"))?;
    Ok(Struct {
        fields: object
            .iter()
            .map(|(key, value)| (key.clone(), json_to_prost(value)))
            .collect(),
    })
}

fn json_to_prost(value: &Value) -> ProstValue {
    let kind = match value {
        Value::Null => Kind::NullValue(0),
        Value::Bool(value) => Kind::BoolValue(*value),
        Value::Number(value) => Kind::NumberValue(value.as_f64().unwrap_or_default()),
        Value::String(value) => Kind::StringValue(value.clone()),
        Value::Array(values) => Kind::ListValue(ListValue {
            values: values.iter().map(json_to_prost).collect(),
        }),
        Value::Object(values) => Kind::StructValue(Struct {
            fields: values
                .iter()
                .map(|(key, value)| (key.clone(), json_to_prost(value)))
                .collect(),
        }),
    };
    ProstValue { kind: Some(kind) }
}

pub(crate) fn prost_struct_to_json(value: Struct) -> Value {
    Value::Object(
        value
            .fields
            .into_iter()
            .map(|(key, value)| (key, prost_to_json(value)))
            .collect(),
    )
}

fn prost_to_json(value: ProstValue) -> Value {
    match value.kind {
        None | Some(Kind::NullValue(_)) => Value::Null,
        Some(Kind::BoolValue(value)) => Value::Bool(value),
        Some(Kind::NumberValue(value)) => prost_number_to_json(value),
        Some(Kind::StringValue(value)) => Value::String(value),
        Some(Kind::ListValue(value)) => {
            Value::Array(value.values.into_iter().map(prost_to_json).collect())
        }
        Some(Kind::StructValue(value)) => prost_struct_to_json(value),
    }
}

fn prost_number_to_json(value: f64) -> Value {
    // google.protobuf.Struct represents every JSON number as an f64. Recover
    // integer-valued numbers within IEEE-754's exact range so OpenAI-shaped
    // media responses still deserialize into fields such as `created: u32`.
    const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
    if value.is_finite() && value.fract() == 0.0 && value.abs() <= MAX_SAFE_INTEGER {
        return Value::Number((value as i64).into());
    }
    serde_json::Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn resolve_disaggregated_params(
    request: &PreprocessedRequest,
    mode: DisaggregationMode,
    bootstrap_host: Option<&str>,
    bootstrap_port: Option<u16>,
) -> Result<Option<pb::DisaggregatedParams>, DynamoError> {
    if mode == DisaggregationMode::Aggregated {
        return Ok(None);
    }
    if let Some(info) = request.bootstrap_info.as_ref() {
        return bootstrap_values_to_proto(
            &info.bootstrap_host,
            u64::from(info.bootstrap_port),
            info.bootstrap_room,
            request
                .routing
                .as_ref()
                .and_then(|routing| routing.prefill_dp_rank),
        )
        .map(Some);
    }
    if let Some(prefill) = request.prefill_result.as_ref() {
        return disaggregated_json_to_proto(&prefill.disaggregated_params).map(Some);
    }
    if mode.is_prefill() {
        let host = bootstrap_host.ok_or_else(|| {
            client::invalid_arg("prefill request has no bootstrap host from discovery")
        })?;
        let port = bootstrap_port.ok_or_else(|| {
            client::invalid_arg("prefill request has no bootstrap port from discovery")
        })?;
        let room = rand::random::<u64>() & (i64::MAX as u64);
        return bootstrap_values_to_proto(host, u64::from(port), room, None).map(Some);
    }
    Err(client::invalid_arg(
        "decode request has neither bootstrap_info nor prefill_result",
    ))
}

fn disaggregated_json_to_proto(value: &Value) -> Result<pb::DisaggregatedParams, DynamoError> {
    let host = value
        .get("bootstrap_host")
        .and_then(Value::as_str)
        .ok_or_else(|| client::invalid_arg("disaggregated_params.bootstrap_host is missing"))?;
    let port = value
        .get("bootstrap_port")
        .and_then(Value::as_u64)
        .ok_or_else(|| client::invalid_arg("disaggregated_params.bootstrap_port is missing"))?;
    let room = value
        .get("bootstrap_room")
        .and_then(Value::as_u64)
        .ok_or_else(|| client::invalid_arg("disaggregated_params.bootstrap_room is missing"))?;
    let mut params = bootstrap_values_to_proto(
        host,
        port,
        room,
        value
            .get("prefill_dp_rank")
            .and_then(Value::as_u64)
            .and_then(|rank| u32::try_from(rank).ok()),
    )?;
    params.bootstrap_pair_key = value
        .get("bootstrap_pair_key")
        .and_then(Value::as_str)
        .map(str::to_string);
    params.decode_tp_size = value
        .get("decode_tp_size")
        .and_then(Value::as_i64)
        .and_then(|size| i32::try_from(size).ok());
    Ok(params)
}

fn bootstrap_values_to_proto(
    host: &str,
    port: u64,
    room: u64,
    prefill_dp_rank: Option<u32>,
) -> Result<pb::DisaggregatedParams, DynamoError> {
    if host.trim().is_empty() {
        return Err(client::invalid_arg("bootstrap_host must not be empty"));
    }
    Ok(pb::DisaggregatedParams {
        bootstrap_host: host.to_string(),
        bootstrap_port: i32::try_from(port)
            .map_err(|_| client::invalid_arg(format!("bootstrap_port is out of range: {port}")))?,
        bootstrap_room: i64::try_from(room).map_err(|_| {
            client::invalid_arg(format!(
                "bootstrap_room must fit SGLang's signed int64 field: {room}"
            ))
        })?,
        prefill_dp_rank: prefill_dp_rank.map(|rank| i32::try_from(rank).unwrap_or(i32::MAX)),
        bootstrap_pair_key: None,
        decode_tp_size: None,
    })
}

pub(crate) fn disaggregated_params_to_json(params: &pb::DisaggregatedParams) -> Value {
    serde_json::json!({
        "bootstrap_host": params.bootstrap_host,
        "bootstrap_port": params.bootstrap_port,
        "bootstrap_room": params.bootstrap_room,
        "prefill_dp_rank": params.prefill_dp_rank,
        "bootstrap_pair_key": params.bootstrap_pair_key,
        "decode_tp_size": params.decode_tp_size,
    })
}

fn json_value_to_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        value => value.to_string(),
    }
}

pub(crate) fn map_generate_response(
    response: pb::GenerateResponse,
    fallback_prompt_tokens: u32,
    fallback_completion_tokens: u32,
    return_tokens_as_ids: bool,
) -> Result<LLMEngineOutput, DynamoError> {
    let token_ids = response
        .delta_output_ids
        .iter()
        .map(|id| {
            u32::try_from(*id).map_err(|_| {
                client::protocol_error(format!("SGLang returned a negative token id: {id}"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let usage_message = response.usage.as_ref();
    let prompt_tokens = usage_message
        .and_then(|usage| u32::try_from(usage.prompt_tokens).ok())
        .unwrap_or(fallback_prompt_tokens);
    let completion_tokens = usage_message
        .and_then(|usage| u32::try_from(usage.completion_tokens).ok())
        .unwrap_or(fallback_completion_tokens);
    let mut mapped = match response.terminal.as_ref() {
        None => LLMEngineOutput::default(),
        Some(pb::generate_response::Terminal::Finish(finish)) => {
            use pb::FinishReason;
            let reason = FinishReason::try_from(finish.reason).map_err(|_| {
                client::protocol_error(format!("unknown SGLang finish reason {}", finish.reason))
            })?;
            match reason {
                FinishReason::Stop => LLMEngineOutput::stop(),
                FinishReason::Length => LLMEngineOutput::length(),
                FinishReason::Abort | FinishReason::Cancelled => LLMEngineOutput::cancelled(),
                FinishReason::Unspecified => {
                    return Err(client::protocol_error(
                        "SGLang terminal finish reason is unspecified",
                    ));
                }
            }
            .with_usage(usage(prompt_tokens, completion_tokens))
        }
        Some(pb::generate_response::Terminal::Error(error)) => {
            LLMEngineOutput::error(error.message.clone())
                .with_usage(usage(prompt_tokens, completion_tokens))
        }
    };
    mapped.token_ids = token_ids;
    mapped.text = response.delta_text;
    mapped.index = Some(u32::try_from(response.choice_index).map_err(|_| {
        client::protocol_error(format!(
            "SGLang returned negative choice index {}",
            response.choice_index
        ))
    })?);
    if let Some(pb::generate_response::Terminal::Finish(finish)) = response.terminal.as_ref() {
        mapped.stop_reason = finish.stop_reason.as_ref().and_then(|reason| {
            reason.reason.as_ref().map(|reason| match reason {
                pb::stop_reason::Reason::MatchedString(value) => StopReason::String(value.clone()),
                pb::stop_reason::Reason::MatchedTokenId(value) => {
                    StopReason::Int(i64::from(*value))
                }
            })
        });
    }
    if let Some(logprobs) = response.logprobs.as_ref() {
        if logprobs.output.len() != mapped.token_ids.len() {
            return Err(client::protocol_error(format!(
                "SGLang returned {} output logprobs for {} delta token IDs",
                logprobs.output.len(),
                mapped.token_ids.len()
            )));
        }
        for (entry, token_id) in logprobs.output.iter().zip(&mapped.token_ids) {
            let entry_token_id = u32::try_from(entry.token_id)
                .map_err(|_| client::protocol_error("output-logprob token id does not fit u32"))?;
            if entry_token_id != *token_id {
                return Err(client::protocol_error(format!(
                    "SGLang output-logprob token ID {entry_token_id} does not align with delta token ID {token_id}"
                )));
            }
        }
        let (selected, top) = map_output_logprobs(&logprobs.output, return_tokens_as_ids)?;
        mapped.tokens = Some(
            logprobs
                .output
                .iter()
                .map(|entry| entry.text.clone())
                .collect(),
        );
        mapped.log_probs = (!selected.is_empty()).then_some(selected);
        mapped.top_logprobs = (!top.is_empty()).then_some(top);
    }
    let mut engine_data = response
        .engine_metadata
        .map(prost_struct_to_json)
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    if let Some(routed) = response.routed_experts {
        engine_data.insert(
            "routed_experts".to_string(),
            serde_json::json!({
                "expert_ids": routed.expert_ids,
                "shape": routed.shape,
                "start_position": routed.start_position,
            }),
        );
    }
    if let Some(logprobs) = response.logprobs.as_ref()
        && !logprobs.prompt.is_empty()
    {
        engine_data.insert(
            "prompt_logprobs".to_string(),
            prompt_logprobs_to_json(&logprobs.prompt),
        );
    }
    mapped.engine_data = (!engine_data.is_empty()).then_some(Value::Object(engine_data));
    mapped.disaggregated_params = response
        .prefill_handoff
        .as_ref()
        .map(disaggregated_params_to_json);
    if let (Some(usage), Some(completion_usage)) = (usage_message, mapped.completion_usage.as_mut())
        && let Ok(cached_tokens) = u32::try_from(usage.cached_prompt_tokens)
        && cached_tokens > 0
    {
        let mut details = dynamo_backend_common::PromptTokensDetails::default();
        details.cached_tokens = Some(cached_tokens);
        completion_usage.prompt_tokens_details = Some(details);
    }
    Ok(mapped)
}

fn map_output_logprobs(
    values: &[pb::TokenLogprob],
    return_tokens_as_ids: bool,
) -> Result<(Vec<f64>, Vec<Vec<TopLogprob>>), DynamoError> {
    let selected = values
        .iter()
        .map(|entry| f64::from(entry.logprob))
        .collect();
    let top = values
        .iter()
        .map(|entry| {
            entry
                .top_logprobs
                .iter()
                .enumerate()
                .map(|(index, alternative)| {
                    let token_id = u32::try_from(alternative.token_id).map_err(|_| {
                        client::protocol_error("top-logprob token id does not fit u32")
                    })?;
                    Ok(TopLogprob {
                        rank: alternative
                            .rank
                            .and_then(|rank| u32::try_from(rank).ok())
                            .unwrap_or_else(|| u32::try_from(index + 1).unwrap_or(u32::MAX)),
                        token_id,
                        token: if return_tokens_as_ids {
                            Some(format!("token_id:{token_id}"))
                        } else {
                            alternative.text.clone()
                        },
                        logprob: f64::from(alternative.logprob),
                        bytes: None,
                    })
                })
                .collect::<Result<Vec<_>, DynamoError>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((selected, top))
}

fn prompt_logprobs_to_json(values: &[pb::TokenLogprob]) -> Value {
    let mut positions = Vec::with_capacity(values.len() + 1);
    positions.push(Value::Null);
    for selected in values {
        let mut position = Map::new();
        position.insert(
            selected.token_id.to_string(),
            serde_json::json!({
                "logprob": selected.logprob,
                "decoded_token": selected.text,
            }),
        );
        for alternative in &selected.top_logprobs {
            position
                .entry(alternative.token_id.to_string())
                .or_insert_with(|| {
                    serde_json::json!({
                        "logprob": alternative.logprob,
                        "decoded_token": alternative.text,
                    })
                });
        }
        positions.push(Value::Object(position));
    }
    Value::Array(positions)
}

pub(crate) fn build_embed_request(
    request: &Value,
    request_id: &str,
) -> Result<pb::EmbedRequest, DynamoError> {
    let input = request
        .get("input")
        .ok_or_else(|| client::invalid_arg("embedding request is missing input"))?;
    let inputs = match input {
        Value::String(text) => vec![pb::EmbedInput {
            input: Some(pb::embed_input::Input::Text(text.clone())),
        }],
        Value::Array(values) if values.iter().all(Value::is_string) => values
            .iter()
            .map(|value| pb::EmbedInput {
                input: Some(pb::embed_input::Input::Text(
                    value.as_str().unwrap_or_default().to_string(),
                )),
            })
            .collect(),
        Value::Array(values) if values.iter().all(Value::is_number) => vec![pb::EmbedInput {
            input: Some(pb::embed_input::Input::InputIds(pb::TokenIds {
                values: json_token_ids(values)?,
            })),
        }],
        Value::Array(values) if values.iter().all(Value::is_array) => values
            .iter()
            .map(|value| {
                Ok(pb::EmbedInput {
                    input: Some(pb::embed_input::Input::InputIds(pb::TokenIds {
                        values: json_token_ids(value.as_array().expect("guarded by is_array"))?,
                    })),
                })
            })
            .collect::<Result<Vec<_>, DynamoError>>()?,
        _ => {
            return Err(client::invalid_arg(
                "embedding input must be text, token IDs, or a homogeneous batch",
            ));
        }
    };
    if inputs.is_empty() {
        return Err(client::invalid_arg(
            "embedding input batch must not be empty",
        ));
    }
    let encoding = match request
        .get("encoding_format")
        .and_then(Value::as_str)
        .unwrap_or("float")
    {
        "float" => pb::EmbeddingEncoding::Float,
        "base64" => pb::EmbeddingEncoding::Base64,
        other => {
            return Err(client::invalid_arg(format!(
                "unsupported embedding encoding_format `{other}`"
            )));
        }
    };
    let dimensions = request
        .get("dimensions")
        .map(|value| {
            value
                .as_i64()
                .and_then(|value| i32::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| client::invalid_arg("dimensions must be a positive integer"))
        })
        .transpose()?;
    let mut trace_headers = HashMap::new();
    dynamo_runtime::logging::inject_trace_headers_into_map(&mut trace_headers);
    Ok(pb::EmbedRequest {
        inputs,
        dimensions,
        encoding: encoding.into(),
        rid: Some(request_id.to_string()),
        routing_key: None,
        routed_dp_rank: None,
        priority: 0,
        trace_headers,
    })
}

fn json_token_ids(values: &[Value]) -> Result<Vec<i32>, DynamoError> {
    values
        .iter()
        .map(|value| {
            value
                .as_i64()
                .and_then(|value| i32::try_from(value).ok())
                .filter(|value| *value >= 0)
                .ok_or_else(|| client::invalid_arg("token IDs must be non-negative i32 values"))
        })
        .collect()
}

pub(crate) fn map_embed_response(
    response: pb::EmbedResponse,
    model: &str,
) -> Result<Value, DynamoError> {
    let data = response
        .embeddings
        .into_iter()
        .map(|embedding| {
            use pb::embedding::Data;
            let value = match embedding.data {
                Some(Data::FloatValues(values)) => serde_json::json!(values.values),
                Some(Data::PackedFloat32(bytes)) => {
                    Value::String(base64::engine::general_purpose::STANDARD.encode(bytes))
                }
                None => {
                    return Err(client::protocol_error(
                        "SGLang embedding response is missing data",
                    ));
                }
            };
            Ok(serde_json::json!({
                "object": "embedding",
                "index": embedding.index,
                "embedding": value,
            }))
        })
        .collect::<Result<Vec<_>, DynamoError>>()?;
    let usage = response.usage.unwrap_or_default();
    Ok(serde_json::json!({
        "object": "list",
        "data": data,
        "model": model,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "total_tokens": usage.total_tokens,
        },
    }))
}

#[cfg(test)]
mod tests {
    use dynamo_backend_common::{
        BootstrapInfo, DisaggregationMode, FinishReason, MultimodalData, OutputOptions,
        PrefillResult, PreprocessedRequest, SamplingOptions, StopConditions,
    };

    use super::{
        build_embed_request, build_generate_request, disaggregated_params_to_json,
        json_object_to_struct, map_embed_response, map_generate_response, prost_struct_to_json,
    };
    use crate::proto as pb;

    fn request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("Qwen/Qwen3-0.6B".to_string())
            .token_ids(vec![1, 2, 3])
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .stop_conditions(StopConditions {
                max_tokens: Some(8),
                ..Default::default()
            })
            .build()
            .unwrap()
    }

    #[test]
    fn request_maps_seed_priority_choices_guidance_and_visible_stops() {
        let mut request = request();
        request.sampling_options.seed = Some(42);
        request.sampling_options.n = Some(2);
        request.sampling_options.guided_decoding =
            Some(dynamo_backend_common::GuidedDecodingOptions::new(
                None,
                None,
                Some(vec!["a".into(), "b".into()]),
                None,
                Some("xgrammar".into()),
                None,
                None,
            ));
        request.stop_conditions.stop_token_ids_visible = Some(vec![9]);
        request.routing.get_or_insert_default().priority = Some(7);
        let mapped =
            build_generate_request(&request, "rid", DisaggregationMode::Aggregated, None, None)
                .unwrap();
        let sampling = mapped.sampling_params.unwrap();
        assert_eq!(sampling.seed, Some(42));
        assert_eq!(sampling.n, Some(2));
        assert!(sampling.token_stops[0].include_in_output);
        assert_eq!(mapped.priority, 7);
        assert!(matches!(
            sampling.guided_decoding.unwrap().constraint,
            Some(pb::guided_decoding::Constraint::Choice(_))
        ));
    }

    #[test]
    fn protobuf_struct_recovers_integer_valued_json_numbers() {
        let original = serde_json::json!({
            "created": 1_784_253_717,
            "num_inference_steps": 2,
            "guidance_scale": 7.5,
            "nested": [3, 1.25],
        });
        let round_trip =
            prost_struct_to_json(json_object_to_struct(&original).expect("JSON object"));

        assert_eq!(round_trip["created"].as_u64(), Some(1_784_253_717));
        assert_eq!(round_trip["num_inference_steps"].as_u64(), Some(2));
        assert_eq!(round_trip["guidance_scale"].as_f64(), Some(7.5));
        assert_eq!(round_trip["nested"][0].as_u64(), Some(3));
        assert_eq!(round_trip["nested"][1].as_f64(), Some(1.25));
    }

    #[test]
    fn request_maps_explicit_string_stop_visibility() {
        for (include, expected) in [(Some(false), false), (Some(true), true)] {
            let mut request = request();
            request.stop_conditions.stop = Some(vec!["END".to_string()]);
            request.sampling_options.include_stop_str_in_output = include;

            let mapped =
                build_generate_request(&request, "rid", DisaggregationMode::Aggregated, None, None)
                    .unwrap();
            let string_stops = mapped.sampling_params.unwrap().string_stops;
            assert_eq!(string_stops.len(), 1);
            assert_eq!(string_stops[0].value, "END");
            assert_eq!(string_stops[0].include_in_output, expected);
        }
    }

    #[test]
    fn request_maps_generation_limits_and_logprobs() {
        let mut request = request();
        request.stop_conditions.min_tokens = Some(8);
        request.stop_conditions.ignore_eos = Some(true);
        request.output_options.logprobs = Some(2);

        let mapped =
            build_generate_request(&request, "rid", DisaggregationMode::Aggregated, None, None)
                .unwrap();
        let sampling = mapped.sampling_params.unwrap();
        assert_eq!(sampling.max_new_tokens, Some(8));
        assert_eq!(sampling.min_new_tokens, Some(8));
        assert_eq!(sampling.ignore_eos, Some(true));

        let logprobs = mapped.logprob_options.unwrap();
        assert!(logprobs.return_logprobs);
        assert_eq!(logprobs.top_logprobs, 2);
    }

    #[test]
    fn request_rejects_explicit_best_of_that_differs_from_n() {
        let mut request = request();
        request.sampling_options.n = Some(2);
        request.sampling_options.best_of = Some(3);

        assert!(
            build_generate_request(&request, "rid", DisaggregationMode::Aggregated, None, None)
                .unwrap_err()
                .to_string()
                .contains("best_of is unsupported unless it equals n")
        );
    }

    #[test]
    fn request_rejects_conflicting_stop_visibility() {
        let mut request = request();
        request.stop_conditions.stop_token_ids_hidden = Some(vec![9]);
        request.stop_conditions.stop_token_ids_visible = Some(vec![9]);
        assert!(
            build_generate_request(&request, "rid", DisaggregationMode::Aggregated, None, None)
                .unwrap_err()
                .to_string()
                .contains("both visible and hidden")
        );
    }

    #[test]
    fn embedding_batch_preserves_dimensions_and_base64_encoding() {
        let request = serde_json::json!({
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "input": ["one", "two"],
            "dimensions": 2,
            "encoding_format": "base64",
        });
        let mapped = build_embed_request(&request, "embed-rid").unwrap();
        assert_eq!(mapped.inputs.len(), 2);
        assert_eq!(mapped.dimensions, Some(2));
        assert_eq!(mapped.encoding, pb::EmbeddingEncoding::Base64 as i32);

        let response = map_embed_response(
            pb::EmbedResponse {
                embeddings: vec![pb::Embedding {
                    index: 0,
                    data: Some(pb::embedding::Data::PackedFloat32(vec![0, 0, 128, 63])),
                }],
                usage: Some(pb::Usage {
                    prompt_tokens: 2,
                    total_tokens: 2,
                    ..Default::default()
                }),
            },
            "Qwen/Qwen3-Embedding-0.6B",
        )
        .unwrap();
        assert_eq!(response["data"][0]["embedding"], "AACAPw==");
        assert_eq!(response["usage"]["total_tokens"], 2);
    }

    #[test]
    fn multimodal_hash_cardinality_is_enforced() {
        let mut request = request();
        request.multi_modal_data = Some(std::collections::HashMap::from([(
            "image_url".to_string(),
            vec![MultimodalData::RawUrl("https://example/image.png".into())],
        )]));
        request.extra_args = Some(serde_json::json!({"mm_hashes": []}));
        assert!(
            build_generate_request(&request, "rid", DisaggregationMode::Aggregated, None, None)
                .is_err()
        );
    }

    #[test]
    fn decoded_multimodal_input_maps_to_nixl_external_tensor() {
        let descriptor = serde_json::from_value(serde_json::json!({
            "nixl_metadata": "b64:metadata",
            "nixl_descriptor": {
                "addr": 4096,
                "size": 48,
                "mem_type": "Dram",
                "device_id": 0,
            },
            "shape": [4, 4, 3],
            "dtype": "UINT8",
            "metadata": null,
        }))
        .unwrap();
        let mut request = request();
        request.multi_modal_data = Some(std::collections::HashMap::from([(
            "image_url".to_string(),
            vec![MultimodalData::Decoded(descriptor)],
        )]));
        request.extra_args = Some(serde_json::json!({"mm_hashes": ["canonical-hash"]}));

        let mapped =
            build_generate_request(&request, "rid", DisaggregationMode::Aggregated, None, None)
                .unwrap();
        assert_eq!(mapped.multimodal_inputs.len(), 1);
        let input = &mapped.multimodal_inputs[0];
        assert_eq!(input.routing_hash.as_deref(), Some("canonical-hash"));
        let Some(pb::multimodal_input::Source::DecodedTensor(tensor)) = input.source.as_ref()
        else {
            panic!("decoded media did not map to a tensor");
        };
        assert_eq!(tensor.dtype, pb::TensorDataType::Uint8 as i32);
        assert_eq!(tensor.shape, vec![4, 4, 3]);
        let Some(pb::tensor::Storage::External(buffer)) = tensor.storage.as_ref() else {
            panic!("decoded media did not map to external storage");
        };
        let Some(pb::external_buffer::Transport::Nixl(nixl)) = buffer.transport.as_ref() else {
            panic!("decoded media did not map to NIXL");
        };
        assert_eq!(nixl.length, Some(48));
        assert_eq!(nixl.metadata, br#""b64:metadata""#);
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&nixl.descriptor).unwrap(),
            serde_json::json!({
                "addr": 4096,
                "size": 48,
                "mem_type": "Dram",
                "device_id": 0,
            })
        );
    }

    #[test]
    fn prefill_handoff_round_trips_to_decode_request() {
        let prefill = build_generate_request(
            &request(),
            "rid-prefill",
            DisaggregationMode::Prefill,
            Some("prefill.internal"),
            Some(5001),
        )
        .unwrap();
        let handoff = prefill.disaggregated_params.unwrap();
        let mut decode_request = request();
        decode_request.prefill_result = Some(PrefillResult {
            disaggregated_params: disaggregated_params_to_json(&handoff),
            prompt_tokens_details: None,
        });
        let decode = build_generate_request(
            &decode_request,
            "rid-decode",
            DisaggregationMode::Decode,
            None,
            None,
        )
        .unwrap();
        assert_eq!(decode.disaggregated_params, Some(handoff));
    }

    #[test]
    fn response_preserves_choice_usage_logprobs_and_handoff() {
        let response = pb::GenerateResponse {
            choice_index: 1,
            delta_output_ids: vec![7],
            logprobs: Some(pb::Logprobs {
                output: vec![pb::TokenLogprob {
                    logprob: -0.2,
                    token_id: 7,
                    text: Some("x".into()),
                    top_logprobs: vec![],
                }],
                prompt: vec![],
            }),
            usage: Some(pb::Usage {
                prompt_tokens: 4,
                completion_tokens: 1,
                total_tokens: 5,
                cached_prompt_tokens: 2,
            }),
            prefill_handoff: Some(pb::DisaggregatedParams {
                bootstrap_host: "prefill".into(),
                bootstrap_port: 5001,
                bootstrap_room: 9,
                ..Default::default()
            }),
            terminal: Some(pb::generate_response::Terminal::Finish(
                pb::GenerationFinish {
                    reason: pb::FinishReason::Stop.into(),
                    stop_reason: None,
                },
            )),
            ..Default::default()
        };
        let mapped = map_generate_response(response, 0, 0, false).unwrap();
        assert_eq!(mapped.index, Some(1));
        assert_eq!(mapped.finish_reason, Some(FinishReason::Stop));
        let log_probs = mapped.log_probs.unwrap();
        assert_eq!(log_probs.len(), 1);
        assert!((log_probs[0] + 0.2).abs() < 1e-6);
        assert_eq!(mapped.tokens, Some(vec![Some("x".into())]));
        assert!(mapped.disaggregated_params.is_some());
        assert_eq!(
            mapped
                .completion_usage
                .unwrap()
                .prompt_tokens_details
                .unwrap()
                .cached_tokens,
            Some(2)
        );
    }

    #[test]
    fn response_rejects_misaligned_output_logprobs() {
        let response = pb::GenerateResponse {
            choice_index: 0,
            delta_output_ids: vec![7],
            logprobs: Some(pb::Logprobs {
                output: vec![pb::TokenLogprob {
                    logprob: -0.2,
                    token_id: 8,
                    text: Some("y".into()),
                    top_logprobs: vec![],
                }],
                prompt: vec![],
            }),
            ..Default::default()
        };
        let error = map_generate_response(response, 0, 0, false).unwrap_err();
        assert!(error.to_string().contains("does not align"));
    }

    #[test]
    fn full_width_room_is_checked() {
        let mut request = request();
        request.bootstrap_info = Some(BootstrapInfo {
            bootstrap_host: "prefill".to_string(),
            bootstrap_port: 5000,
            bootstrap_room: i64::MAX as u64 + 1,
            handoff_id: None,
        });
        assert!(
            build_generate_request(&request, "rid", DisaggregationMode::Decode, None, None)
                .is_err()
        );
    }
}
