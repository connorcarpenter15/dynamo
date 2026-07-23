// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CPU-only soak + CPU profile of the real vLLM sidecar driven against the
// mocker gRPC server. No GPU, model, or real engine involved.
//
// 1) Start the mocker in another shell:
//      cargo run --release -p dynamo-vllm-mocker --bin dynamo-vllm-mocker-server -- \
//        --listen 127.0.0.1:50051 --model mocker-model --max-concurrent-requests 1024 \
//        --extra-engine-args '{"speedup_ratio":0.0,"block_size":64,"num_gpu_blocks":16384,"max_num_seqs":512,"max_num_batched_tokens":16384}'
// 2) Run the soak (profiles this process only = the sidecar path):
//      cargo run --release --example soak_sidecar -- \
//        --endpoint 127.0.0.1:50051 --concurrency 64 --duration 180 --isl 1024 --osl 256

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use dynamo_backend_common::{
    GenerateContext, LLMEngine, OutputOptions, PreprocessedRequest, SamplingOptions, StopConditions,
};
use dynamo_vllm_sidecar::VllmSidecarEngine;
use futures::StreamExt;

fn arg(name: &str, default: &str) -> String {
    let args: Vec<String> = std::env::args().collect();
    for i in 0..args.len() {
        if args[i] == name {
            return args.get(i + 1).cloned().unwrap_or_else(|| default.to_string());
        }
    }
    default.to_string()
}

fn build_request(isl: usize, osl: u32) -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("mocker-model".to_string())
        .token_ids(vec![11u32; isl])
        .stop_conditions(StopConditions {
            max_tokens: Some(osl),
            ignore_eos: Some(true),
            ..Default::default()
        })
        .sampling_options(SamplingOptions {
            temperature: Some(0.0),
            ..Default::default()
        })
        .output_options(OutputOptions::default())
        .build()
        .unwrap()
}

#[tokio::main]
async fn main() {
    let endpoint = arg("--endpoint", "127.0.0.1:50051");
    let concurrency: usize = arg("--concurrency", "64").parse().unwrap();
    let duration: u64 = arg("--duration", "180").parse().unwrap();
    let isl: usize = arg("--isl", "1024").parse().unwrap();
    let osl: u32 = arg("--osl", "256").parse().unwrap();
    let connections = arg("--grpc-connections", "16");

    let engine = VllmSidecarEngine::from_args(Some(vec![
        "dynamo-vllm-sidecar".to_string(),
        "--vllm-endpoint".to_string(),
        endpoint.clone(),
        "--model-path".to_string(),
        "mocker-model".to_string(),
        "--grpc-connections".to_string(),
        connections.clone(),
        "--grpc-startup-deadline-secs".to_string(),
        "10".to_string(),
    ]))
    .expect("sidecar args")
    .0;
    engine
        .start(0)
        .await
        .expect("sidecar start (is dynamo-vllm-mocker-server up on --endpoint?)");
    let engine = Arc::new(engine);

    eprintln!(
        "soak: endpoint={endpoint} conns={connections} concurrency={concurrency} duration={duration}s isl={isl} osl={osl}"
    );

    // SIGPROF-based CPU profiler (works without perf_event access, unlike samply/perf).
    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(999)
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()
        .expect("pprof guard");

    let ok = Arc::new(AtomicU64::new(0));
    let err = Arc::new(AtomicU64::new(0));
    let toks = Arc::new(AtomicU64::new(0));
    let deadline = Instant::now() + Duration::from_secs(duration);
    let start = Instant::now();

    let mut handles = Vec::new();
    for _ in 0..concurrency {
        let engine = Arc::clone(&engine);
        let ok = Arc::clone(&ok);
        let err = Arc::clone(&err);
        let toks = Arc::clone(&toks);
        handles.push(tokio::spawn(async move {
            while Instant::now() < deadline {
                let ctx = dynamo_backend_common::testing::mock_context();
                match engine
                    .generate(build_request(isl, osl), GenerateContext::new(ctx, None))
                    .await
                {
                    Ok(mut stream) => {
                        let mut n = 0u64;
                        let mut failed = false;
                        while let Some(item) = stream.next().await {
                            match item {
                                Ok(out) => n += out.token_ids.len() as u64,
                                Err(_) => {
                                    failed = true;
                                    break;
                                }
                            }
                        }
                        if failed {
                            err.fetch_add(1, Ordering::Relaxed);
                        } else {
                            ok.fetch_add(1, Ordering::Relaxed);
                            toks.fetch_add(n, Ordering::Relaxed);
                        }
                    }
                    Err(_) => {
                        err.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }));
    }

    let progress = {
        let ok = Arc::clone(&ok);
        let err = Arc::clone(&err);
        let toks = Arc::clone(&toks);
        tokio::spawn(async move {
            while Instant::now() < deadline {
                tokio::time::sleep(Duration::from_secs(15)).await;
                let el = start.elapsed().as_secs_f64();
                let o = ok.load(Ordering::Relaxed);
                eprintln!(
                    "  t={el:5.0}s ok={o} err={} req/s={:.1} tok/s={:.0}",
                    err.load(Ordering::Relaxed),
                    o as f64 / el,
                    toks.load(Ordering::Relaxed) as f64 / el
                );
            }
        })
    };

    for h in handles {
        let _ = h.await;
    }
    progress.abort();

    let el = start.elapsed().as_secs_f64();
    let o = ok.load(Ordering::Relaxed);
    eprintln!("\n==== soak summary ====");
    eprintln!("duration_s     {el:.1}");
    eprintln!("requests_ok    {o}");
    eprintln!("requests_err   {}", err.load(Ordering::Relaxed));
    eprintln!("req_per_s      {:.1}", o as f64 / el);
    eprintln!("out_tok_per_s  {:.0}", toks.load(Ordering::Relaxed) as f64 / el);

    match guard.report().build() {
        Ok(report) => {
            let path = arg("--out", "sidecar-flamegraph.svg");
            if let Ok(file) = std::fs::File::create(&path) {
                let _ = report.flamegraph(file);
                eprintln!("flamegraph     {path}");
            }
            // Textual top-function tables so the profile is analyzable without a
            // GUI. INCLUSIVE is order-independent (a fn appears anywhere in the
            // stack); SELF uses the innermost frame.
            use std::collections::{HashMap, HashSet};
            let mut total: isize = 0;
            let mut self_c: HashMap<String, isize> = HashMap::new();
            let mut incl_c: HashMap<String, isize> = HashMap::new();
            for (frames, count) in report.data.iter() {
                total += *count;
                if let Some(sym) = frames.frames.first().and_then(|f| f.first()) {
                    *self_c.entry(format!("{sym}")).or_default() += *count;
                }
                let mut seen = HashSet::new();
                for frame in &frames.frames {
                    for sym in frame {
                        let n = format!("{sym}");
                        if seen.insert(n.clone()) {
                            *incl_c.entry(n).or_default() += *count;
                        }
                    }
                }
            }
            let top = |m: HashMap<String, isize>, label: &str| {
                let mut v: Vec<_> = m.into_iter().collect();
                v.sort_by(|a, b| b.1.cmp(&a.1));
                eprintln!("\n==== top {label} (of {total} samples) ====");
                for (name, c) in v.into_iter().take(35) {
                    eprintln!("{:6.2}%  {}", 100.0 * c as f64 / total.max(1) as f64, name);
                }
            };
            top(self_c, "SELF (leaf) functions");
            top(incl_c, "INCLUSIVE functions");
        }
        Err(e) => eprintln!("pprof report failed: {e}"),
    }
}
