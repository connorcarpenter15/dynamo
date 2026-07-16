// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native SGLang KV-event and Prometheus bridging.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use dynamo_backend_common::{DynamoError, KvEventSource, MetricsBindings, MetricsCtx};
use tokio_util::sync::CancellationToken;

use crate::client::{self, Discovery};

pub(crate) fn kv_event_sources(discovery: &Discovery) -> Result<Vec<KvEventSource>, DynamoError> {
    let mut endpoints = BTreeMap::new();
    for observable in &discovery.observability {
        let Some(endpoint) = observable
            .kv_events_zmq_endpoint
            .as_deref()
            .filter(|endpoint| !endpoint.is_empty())
        else {
            continue;
        };
        if !endpoint.starts_with("tcp://") {
            return Err(client::protocol_error(format!(
                "SGLang KV-event endpoint for DP rank {} is not routable TCP: {endpoint}",
                observable.dp_rank
            )));
        }
        if endpoints
            .insert(
                observable.dp_rank,
                (
                    endpoint.to_string(),
                    observable.kv_events_topic.clone().unwrap_or_default(),
                ),
            )
            .is_some()
        {
            return Err(client::protocol_error(format!(
                "SGLang reported duplicate KV-event endpoint for DP rank {}",
                observable.dp_rank
            )));
        }
    }
    Ok(endpoints
        .into_iter()
        .map(|(dp_rank, (endpoint, topic))| KvEventSource::Zmq {
            endpoint,
            topic,
            dp_rank,
        })
        .collect())
}

pub(crate) fn setup_metrics(
    discovery: &Discovery,
    ctx: MetricsCtx<'_>,
    cancel: CancellationToken,
) -> Result<MetricsBindings, DynamoError> {
    let urls = discovery
        .observability
        .iter()
        .filter_map(|observable| observable.metrics_url.clone())
        .filter(|url| !url.is_empty())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if urls.is_empty() {
        return Ok(MetricsBindings::default());
    }
    for url in &urls {
        let parsed = reqwest::Url::parse(url).map_err(|error| {
            client::protocol_error(format!("invalid metrics URL `{url}`: {error}"))
        })?;
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err(client::protocol_error(format!(
                "unsupported SGLang metrics URL scheme in `{url}`"
            )));
        }
    }

    let snapshot = Arc::new(RwLock::new(String::new()));
    let callback_snapshot = snapshot.clone();
    ctx.metrics.add_expfmt_callback(Arc::new(move || {
        Ok(callback_snapshot
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone())
    }));

    tokio::spawn(async move {
        let http = match reqwest::Client::builder()
            .timeout(Duration::from_secs(3))
            .build()
        {
            Ok(client) => client,
            Err(error) => {
                tracing::error!(%error, "failed to construct SGLang metrics client");
                return;
            }
        };
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                _ = cancel.cancelled() => break,
                _ = interval.tick() => {
                    let mut merged = String::new();
                    for url in &urls {
                        match http.get(url).send().await {
                            Ok(response) if response.status().is_success() => match response.text().await {
                                Ok(text) => {
                                    if !merged.is_empty() && !merged.ends_with('\n') {
                                        merged.push('\n');
                                    }
                                    merged.push_str(&text);
                                }
                                Err(error) => tracing::warn!(%url, %error, "failed to read SGLang metrics"),
                            },
                            Ok(response) => tracing::warn!(%url, status = %response.status(), "SGLang metrics returned an error"),
                            Err(error) => tracing::warn!(%url, %error, "failed to scrape SGLang metrics"),
                        }
                    }
                    if !merged.is_empty() {
                        *snapshot.write().unwrap_or_else(|poison| poison.into_inner()) = merged;
                    }
                }
            }
        }
    });

    // SGLang's exposition already contains per-rank labels. Dynamo router-load
    // snapshots continue to come from the KV-event path; fabricating snapshots
    // from lossy Prometheus text would introduce stale routing signals.
    Ok(MetricsBindings::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto as pb;

    fn discovery(observability: Vec<pb::ObservabilityEndpoint>) -> Discovery {
        Discovery {
            model_path: "model".into(),
            tokenizer_path: "model".into(),
            served_model_name: None,
            max_model_len: None,
            runtime_kind: pb::RuntimeKind::Llm,
            worker_role: pb::WorkerRole::Aggregated,
            capacity: Default::default(),
            dp_topology: Default::default(),
            bootstrap: None,
            observability,
            reasoning_parser: None,
            tool_call_parser: None,
            weight_version: None,
        }
    }

    #[test]
    fn kv_sources_preserve_rank_topic_and_endpoint() {
        let sources = kv_event_sources(&discovery(vec![pb::ObservabilityEndpoint {
            dp_rank: 7,
            metrics_url: None,
            kv_events_zmq_endpoint: Some("tcp://127.0.0.1:5557".into()),
            kv_events_topic: Some("events".into()),
        }]))
        .unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].dp_rank(), 7);
    }

    #[test]
    fn kv_sources_reject_duplicate_rank_and_non_tcp() {
        let item = pb::ObservabilityEndpoint {
            dp_rank: 0,
            metrics_url: None,
            kv_events_zmq_endpoint: Some("ipc:///tmp/events".into()),
            kv_events_topic: None,
        };
        assert!(kv_event_sources(&discovery(vec![item])).is_err());
    }
}
