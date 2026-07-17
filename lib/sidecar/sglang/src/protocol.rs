// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure request lowering and response conversion for SGLang's native gRPC protocol.

use std::collections::HashMap;

use dynamo_backend_common::{
    DisaggregationMode, DynamoError, LLMEngineOutput, LLMEngineOutputExt, PreprocessedRequest,
    PromptTokensDetails, StopReason, TopLogprob, usage,
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
    let input_ids = request
        .token_ids
        .iter()
        .map(|token| {
            i32::try_from(*token)
                .map_err(|_| client::invalid_arg(format!("token id {token} does not fit in i32")))
        })
        .collect::<Result<Vec<_>, _>>()?;
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

    let mut stop_token_ids = Vec::new();
    for tokens in [
        request.stop_conditions.stop_token_ids.as_ref(),
        request.stop_conditions.stop_token_ids_hidden.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        for token in tokens {
            let token = i32::try_from(*token).map_err(|_| {
                client::invalid_arg(format!("stop token id {token} does not fit in i32"))
            })?;
            if !stop_token_ids.contains(&token) {
                stop_token_ids.push(token);
            }
        }
    }

    let guided_decoding = request
        .sampling_options
        .guided_decoding
        .as_ref()
        .map(guided_decoding_to_proto)
        .transpose()?;
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
        stop: request.stop_conditions.stop.clone().unwrap_or_default(),
        stop_token_ids,
        ignore_eos: request.stop_conditions.ignore_eos,
        n: Some(if mode.is_prefill() {
            1
        } else {
            i32::from(request.sampling_options.n.unwrap_or(1))
        }),
        seed: request.sampling_options.seed,
        guided_decoding,
        require_reasoning: Some(request.require_reasoning),
        max_thinking_tokens: request.stop_conditions.max_thinking_tokens,
    };

    let output_options = &request.output_options;
    let return_logprob = !mode.is_prefill()
        && (output_options.logprobs.is_some() || output_options.prompt_logprobs.is_some());
    let top_logprobs_num = if mode.is_prefill() {
        0
    } else {
        output_options
            .logprobs
            .unwrap_or(0)
            .max(output_options.prompt_logprobs.unwrap_or(0))
    };
    let top_logprobs_num = i32::try_from(top_logprobs_num)
        .map_err(|_| client::invalid_arg("requested logprobs does not fit in i32"))?;
    let logprob_start_len = if mode.is_prefill() {
        -1
    } else {
        output_options.prompt_logprobs.map(|_| 0).unwrap_or(-1)
    };
    let routed_dp_rank = request
        .routing
        .as_ref()
        .and_then(|routing| routing.dp_rank)
        .map(i32::try_from)
        .transpose()
        .map_err(|_| client::invalid_arg("routed dp_rank does not fit in i32"))?;
    let lora_path = request
        .routing
        .as_ref()
        .and_then(|routing| routing.lora_name.clone());

    let mut trace_headers = HashMap::new();
    dynamo_runtime::logging::inject_trace_headers_into_map(&mut trace_headers);

    Ok(pb::GenerateRequest {
        input_ids,
        sampling_params: Some(sampling_params),
        stream: Some(true),
        return_logprob: Some(return_logprob),
        top_logprobs_num: Some(top_logprobs_num),
        logprob_start_len: Some(logprob_start_len),
        rid: Some(request_id.to_string()),
        lora_path,
        routing_key: request.mdc_sum.clone(),
        routed_dp_rank,
        trace_headers,
        session_id: None,
        disaggregated_params: resolve_disaggregated_params(
            request,
            mode,
            bootstrap_host,
            bootstrap_port,
        )?,
        priority: request
            .routing
            .as_ref()
            .and_then(|routing| routing.priority),
    })
}

fn validate_request(request: &PreprocessedRequest) -> Result<(), DynamoError> {
    if request.token_ids.is_empty() {
        return Err(client::invalid_arg("token_ids must not be empty"));
    }
    if request.prompt_embeds.is_some() {
        return Err(client::invalid_arg(
            "prompt_embeds are not supported by SGLang's native gRPC proto",
        ));
    }
    if request.multi_modal_data.is_some() || request.mm_processor_kwargs.is_some() {
        return Err(client::invalid_arg(
            "multimodal payloads are not supported by SGLang's native Generate RPC",
        ));
    }
    let n = request.sampling_options.n.unwrap_or(1);
    if n == 0 {
        return Err(client::invalid_arg("n must be greater than zero"));
    }
    if request.sampling_options.best_of.unwrap_or(n) != n {
        return Err(client::invalid_arg(
            "best_of is unsupported unless it equals n; SGLang does not implement beam-search selection",
        ));
    }
    if request.sampling_options.use_beam_search.unwrap_or(false) {
        return Err(client::invalid_arg(
            "beam search is not represented by SGLang's native gRPC proto",
        ));
    }
    if let Some(penalty) = request.sampling_options.length_penalty
        && (penalty - 1.0).abs() > f32::EPSILON
    {
        return Err(client::invalid_arg(
            "length_penalty is not represented by SGLang's native gRPC proto",
        ));
    }
    if request
        .sampling_options
        .include_stop_str_in_output
        .unwrap_or(false)
    {
        return Err(client::invalid_arg(
            "include_stop_str_in_output is not represented by SGLang's native gRPC proto",
        ));
    }
    if request
        .stop_conditions
        .stop_token_ids_visible
        .as_ref()
        .is_some_and(|tokens| !tokens.is_empty())
    {
        return Err(client::invalid_arg(
            "visible stop-token semantics are not represented by SGLang's native gRPC proto",
        ));
    }
    if let Some(guided) = request.sampling_options.guided_decoding.as_ref()
        && (guided
            .backend
            .as_ref()
            .is_some_and(|value| !value.is_empty())
            || guided.whitespace_pattern.is_some())
    {
        return Err(client::invalid_arg(
            "guided-decoding backend and whitespace selection are not represented by SGLang's native gRPC proto",
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
    if constraint.is_none() {
        return Err(client::invalid_arg(
            "guided decoding requires one of json, regex, grammar, choice, or structural_tag",
        ));
    }
    Ok(pb::GuidedDecoding { constraint })
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
        return bootstrap_values_to_proto(host, u64::from(port), room).map(Some);
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
    bootstrap_values_to_proto(host, port, room)
}

fn bootstrap_values_to_proto(
    host: &str,
    port: u64,
    room: u64,
) -> Result<pb::DisaggregatedParams, DynamoError> {
    if host.trim().is_empty() {
        return Err(client::invalid_arg("bootstrap_host must not be empty"));
    }
    let bootstrap_port = i32::try_from(port)
        .map_err(|_| client::invalid_arg(format!("bootstrap_port is out of range: {port}")))?;
    let bootstrap_room = i64::try_from(room).map_err(|_| {
        client::invalid_arg(format!(
            "bootstrap_room must fit SGLang's signed int64 field: {room}"
        ))
    })?;
    Ok(pb::DisaggregatedParams {
        bootstrap_host: host.to_string(),
        bootstrap_port,
        bootstrap_room,
    })
}

pub(crate) fn disaggregated_params_to_json(params: &pb::DisaggregatedParams) -> Value {
    serde_json::json!({
        "bootstrap_host": params.bootstrap_host,
        "bootstrap_port": params.bootstrap_port,
        "bootstrap_room": params.bootstrap_room,
    })
}

fn json_value_to_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        value => value.to_string(),
    }
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
        Some(Kind::ListValue(ListValue { values })) => {
            Value::Array(values.into_iter().map(prost_to_json).collect())
        }
        Some(Kind::StructValue(value)) => prost_struct_to_json(value),
    }
}

fn prost_number_to_json(value: f64) -> Value {
    const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
    if value.is_finite() && value.fract() == 0.0 && value.abs() <= MAX_SAFE_INTEGER {
        return Value::Number((value as i64).into());
    }
    serde_json::Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
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
            let reason = pb::FinishReason::try_from(finish.reason).map_err(|_| {
                client::protocol_error(format!("unknown SGLang finish reason {}", finish.reason))
            })?;
            match reason {
                pb::FinishReason::Stop => LLMEngineOutput::stop(),
                pb::FinishReason::Length => LLMEngineOutput::length(),
                pb::FinishReason::Cancelled => LLMEngineOutput::cancelled(),
                pb::FinishReason::Unspecified => {
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
                "packed_expert_ids": routed.packed_expert_ids,
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
    if let Some(pb::generate_response::Terminal::Error(error)) = response.terminal.as_ref() {
        engine_data.insert(
            "generation_error".to_string(),
            serde_json::json!({
                "code": error.code,
                "message": error.message,
                "retryable": error.retryable,
            }),
        );
    }
    mapped.engine_data = (!engine_data.is_empty()).then_some(Value::Object(engine_data));

    if let (Some(message), Some(completion_usage)) =
        (usage_message, mapped.completion_usage.as_mut())
        && let Ok(cached_tokens) = u32::try_from(message.cached_prompt_tokens)
        && cached_tokens > 0
    {
        completion_usage.prompt_tokens_details = Some(PromptTokensDetails {
            cached_tokens: Some(cached_tokens),
            ..Default::default()
        });
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
                "rank": Value::Null,
            }),
        );
        for alternative in &selected.top_logprobs {
            position
                .entry(alternative.token_id.to_string())
                .or_insert_with(|| {
                    serde_json::json!({
                        "logprob": alternative.logprob,
                        "decoded_token": alternative.text,
                        "rank": alternative.rank,
                    })
                });
        }
        positions.push(Value::Object(position));
    }
    Value::Array(positions)
}

#[cfg(test)]
mod tests {
    use crate::proto as pb;
    use dynamo_backend_common::{
        BootstrapInfo, DisaggregationMode, FinishReason, GuidedDecodingOptions, OutputOptions,
        PrefillResult, PreprocessedRequest, SamplingOptions, StopConditions,
    };
    use serde_json::json;

    use super::{build_generate_request, disaggregated_params_to_json, map_generate_response};

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
    fn request_maps_native_fields_and_full_width_room() {
        let mut request = request();
        request.bootstrap_info = Some(BootstrapInfo {
            bootstrap_host: "prefill".to_string(),
            bootstrap_port: 5000,
            bootstrap_room: i64::MAX as u64,
            handoff_id: None,
        });
        let mapped =
            build_generate_request(&request, "rid-1", DisaggregationMode::Decode, None, None)
                .unwrap();
        assert_eq!(mapped.input_ids, vec![1, 2, 3]);
        assert_eq!(mapped.rid.as_deref(), Some("rid-1"));
        assert_eq!(mapped.sampling_params.unwrap().max_new_tokens, Some(8));
        assert_eq!(
            mapped.disaggregated_params.unwrap().bootstrap_room,
            i64::MAX
        );
    }

    #[test]
    fn request_maps_typed_generation_controls() {
        let mut request = request();
        request.sampling_options.n = Some(2);
        request.sampling_options.best_of = Some(2);
        request.sampling_options.seed = Some(1234);
        request.stop_conditions.max_thinking_tokens = Some(256);
        request.require_reasoning = true;
        request.routing_mut().priority = Some(7);
        request.sampling_options.guided_decoding = Some(GuidedDecodingOptions {
            choice: Some(vec!["yes".to_string(), "no".to_string()]),
            ..Default::default()
        });

        let mapped = build_generate_request(
            &request,
            "rid-controls",
            DisaggregationMode::Aggregated,
            None,
            None,
        )
        .unwrap();
        assert_eq!(mapped.priority, Some(7));
        let sampling = mapped.sampling_params.unwrap();
        assert_eq!(sampling.n, Some(2));
        assert_eq!(sampling.seed, Some(1234));
        assert_eq!(sampling.require_reasoning, Some(true));
        assert_eq!(sampling.max_thinking_tokens, Some(256));
        assert!(matches!(
            sampling.guided_decoding.unwrap().constraint,
            Some(pb::guided_decoding::Constraint::Choice(pb::ChoiceConstraint { values }))
                if values == ["yes", "no"]
        ));
    }

    #[test]
    fn request_maps_each_guided_decoding_variant() {
        let variants = [
            GuidedDecodingOptions {
                json: Some(json!({"type": "object"})),
                ..Default::default()
            },
            GuidedDecodingOptions {
                regex: Some("[0-9]+".to_string()),
                ..Default::default()
            },
            GuidedDecodingOptions {
                grammar: Some("root ::= 'yes'".to_string()),
                ..Default::default()
            },
            GuidedDecodingOptions {
                structural_tag: Some(json!({"type": "structural_tag"})),
                ..Default::default()
            },
        ];
        for guided in variants {
            let mut request = request();
            request.sampling_options.guided_decoding = Some(guided);
            assert!(
                build_generate_request(
                    &request,
                    "rid-guided",
                    DisaggregationMode::Aggregated,
                    None,
                    None,
                )
                .unwrap()
                .sampling_params
                .unwrap()
                .guided_decoding
                .unwrap()
                .constraint
                .is_some()
            );
        }
    }

    #[test]
    fn prefill_forces_one_choice() {
        let mut request = request();
        request.sampling_options.n = Some(2);
        request.sampling_options.best_of = Some(2);
        let mapped = build_generate_request(
            &request,
            "rid-prefill",
            DisaggregationMode::Prefill,
            Some("prefill"),
            Some(5001),
        )
        .unwrap();
        assert_eq!(mapped.sampling_params.unwrap().n, Some(1));
    }

    #[test]
    fn prefill_clamps_generation_and_disables_decode_only_options() {
        let mut request = request();
        request.stop_conditions.min_tokens = Some(4);
        request.output_options = OutputOptions {
            logprobs: Some(2),
            prompt_logprobs: Some(3),
            ..Default::default()
        };
        let mapped = build_generate_request(
            &request,
            "rid-2",
            DisaggregationMode::Prefill,
            Some("prefill"),
            Some(5001),
        )
        .unwrap();
        let sampling = mapped.sampling_params.unwrap();
        assert_eq!(sampling.max_new_tokens, Some(1));
        assert_eq!(sampling.min_new_tokens, None);
        assert_eq!(mapped.return_logprob, Some(false));
        assert_eq!(mapped.top_logprobs_num, Some(0));
        assert_eq!(mapped.logprob_start_len, Some(-1));
        assert_eq!(mapped.disaggregated_params.unwrap().bootstrap_port, 5001);
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
    fn typed_response_maps_choice_usage_logprobs_and_stop() {
        let response = pb::GenerateResponse {
            delta_output_ids: vec![7],
            choice_index: 1,
            logprobs: Some(pb::Logprobs {
                output: vec![pb::TokenLogprob {
                    logprob: -0.2,
                    token_id: 7,
                    text: Some("x".into()),
                    top_logprobs: vec![pb::LogprobAlternative {
                        logprob: -0.2,
                        token_id: 7,
                        text: Some("x".into()),
                        rank: Some(1),
                    }],
                }],
                prompt: vec![pb::TokenLogprob {
                    logprob: -0.1,
                    token_id: 3,
                    text: Some("p".into()),
                    top_logprobs: vec![],
                }],
            }),
            usage: Some(pb::Usage {
                prompt_tokens: 4,
                completion_tokens: 1,
                total_tokens: 5,
                cached_prompt_tokens: 2,
            }),
            routed_experts: Some(pb::RoutedExpertMetadata {
                packed_expert_ids: vec![1, 0, 0, 0],
                shape: vec![1],
                start_position: 0,
            }),
            engine_metadata: Some(prost_types::Struct {
                fields: [(
                    "weight_version".to_string(),
                    prost_types::Value {
                        kind: Some(prost_types::value::Kind::StringValue("v2".to_string())),
                    },
                )]
                .into_iter()
                .collect(),
            }),
            terminal: Some(pb::generate_response::Terminal::Finish(
                pb::GenerationFinish {
                    reason: pb::FinishReason::Stop as i32,
                    stop_reason: Some(pb::StopReason {
                        reason: Some(pb::stop_reason::Reason::MatchedString("END".to_string())),
                    }),
                },
            )),
        };
        let mapped = map_generate_response(response, 0, 0, false).unwrap();
        assert_eq!(mapped.index, Some(1));
        assert_eq!(mapped.finish_reason, Some(FinishReason::Stop));
        assert_eq!(
            mapped.stop_reason,
            Some(dynamo_backend_common::StopReason::String("END".into()))
        );
        assert_eq!(mapped.token_ids, vec![7]);
        assert_eq!(mapped.tokens, Some(vec![Some("x".into())]));
        let log_probs = mapped.log_probs.unwrap();
        assert_eq!(log_probs.len(), 1);
        assert!((log_probs[0] + 0.2).abs() < 1e-6);
        let usage = mapped.completion_usage.unwrap();
        assert_eq!(usage.total_tokens, 5);
        assert_eq!(usage.prompt_tokens_details.unwrap().cached_tokens, Some(2));
        let engine_data = mapped.engine_data.unwrap();
        assert_eq!(engine_data["weight_version"], json!("v2"));
        assert_eq!(
            engine_data["prompt_logprobs"][1]["3"]["decoded_token"],
            json!("p")
        );
        assert_eq!(engine_data["routed_experts"]["shape"], json!([1]));
    }

    #[test]
    fn typed_error_is_a_choice_terminal_with_stable_metadata() {
        let response = pb::GenerateResponse {
            choice_index: 0,
            terminal: Some(pb::generate_response::Terminal::Error(
                pb::GenerationError {
                    code: pb::GenerationErrorCode::Unavailable as i32,
                    message: "worker unavailable".to_string(),
                    retryable: true,
                },
            )),
            ..Default::default()
        };
        let mapped = map_generate_response(response, 4, 0, false).unwrap();
        assert!(matches!(mapped.finish_reason, Some(FinishReason::Error(_))));
        assert_eq!(
            mapped.engine_data.unwrap()["generation_error"]["retryable"],
            json!(true)
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
        let error = map_generate_response(response, 0, 0, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not align"));
    }

    #[test]
    fn decode_requires_rendezvous_params() {
        assert!(
            build_generate_request(&request(), "rid-3", DisaggregationMode::Decode, None, None,)
                .is_err()
        );
    }

    #[test]
    fn room_above_signed_int64_is_rejected() {
        let mut request = request();
        request.bootstrap_info = Some(BootstrapInfo {
            bootstrap_host: "prefill".to_string(),
            bootstrap_port: 5000,
            bootstrap_room: i64::MAX as u64 + 1,
            handoff_id: None,
        });
        assert!(
            build_generate_request(&request, "rid-4", DisaggregationMode::Decode, None, None,)
                .is_err()
        );
    }
}
