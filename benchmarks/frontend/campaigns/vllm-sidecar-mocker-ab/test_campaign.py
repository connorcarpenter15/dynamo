#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused dry-run regression coverage for the locked campaign topology."""

import json
import tempfile
import unittest
from pathlib import Path

import run_campaign as campaign


class CampaignDryRunTest(unittest.TestCase):
    def test_timeout_only_records_preserve_an_all_failure_leg(self) -> None:
        """Regression: genuine saturation remains analyzable without a summary export."""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "aiperf").mkdir()
            (output / "logs").mkdir()
            (output / "system").mkdir()
            (output / "aiperf/profile_export.jsonl").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "benchmark_phase": "profiling",
                            "x_request_id": "timeout",
                        },
                        "metrics": {"error_isl": {"value": 8192, "unit": "tokens"}},
                        "error": {"type": "TimeoutError", "message": "TimeoutError()"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            for phase in ("warmup", "profiling"):
                phase_dir = output / "aiperf/phases" / phase
                phase_dir.mkdir(parents=True)
                (phase_dir / "profile_export_aiperf.json").write_text(
                    "{}", encoding="utf-8"
                )
            (output / "logs/frontend.log").write_text("", encoding="utf-8")
            (output / "config.json").write_text(
                json.dumps({"benchmark_duration": 60, "aiperf_version": "0.10.0"}),
                encoding="utf-8",
            )
            metrics = campaign.parse_leg_metrics(output)
        self.assertEqual(metrics["completed_requests"], 0)
        self.assertEqual(metrics["failed_requests"], 1)
        self.assertEqual(metrics["failed_request_fraction"], 1)
        self.assertEqual(metrics["output_throughput_tps"], 0)

    def test_frontend_tokens_are_correlated_to_profiling_request_ids(self) -> None:
        """Regression: tokenizer-added tokens must not replace server token totals."""
        with tempfile.TemporaryDirectory() as temporary:
            frontend_log = Path(temporary) / "frontend.log"
            frontend_log.write_text(
                '\x1b[3mx_request_id\x1b[0m=\x1b[0m"warmup" '
                "request completed status=success input_tokens=99 output_tokens=99\n"
                'request completed x_request_id="profile" '
                "status=success input_tokens=32 output_tokens=16\n",
                encoding="utf-8",
            )
            metrics = campaign.parse_frontend_token_metrics(frontend_log, {"profile"})
        self.assertEqual(metrics["request_count"], 1)
        self.assertEqual(metrics["total_input_tokens"], 32)
        self.assertEqual(metrics["total_output_tokens"], 16)
        self.assertEqual(metrics["output_length_min"], 16)
        self.assertEqual(metrics["output_length_max"], 16)

    def test_frontend_stream_errors_remain_analyzable_saturation_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "aiperf").mkdir()
            (output / "logs").mkdir()
            (output / "system").mkdir()
            records = [
                {
                    "metadata": {
                        "benchmark_phase": "profiling",
                        "x_request_id": request_id,
                    },
                    "metrics": {"request_latency": {"value": latency, "unit": "ms"}},
                }
                for request_id, latency in (("complete", 10), ("failed", 1000))
            ]
            (output / "aiperf/profile_export.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            (output / "logs/frontend.log").write_text(
                'request completed x_request_id="complete" status=success '
                "input_tokens=32 output_tokens=16\n"
                'request completed x_request_id="failed" status=error '
                "input_tokens=32 output_tokens=7\n",
                encoding="utf-8",
            )
            (output / "config.json").write_text(
                json.dumps({"benchmark_duration": 60, "aiperf_version": "0.10.0"}),
                encoding="utf-8",
            )
            metrics = campaign.parse_leg_metrics(output)
        self.assertEqual(metrics["completed_requests"], 1)
        self.assertEqual(metrics["failed_requests"], 1)
        self.assertEqual(metrics["server_failed_requests"], 1)
        self.assertEqual(metrics["failed_request_fraction"], 0.5)
        self.assertEqual(metrics["output_throughput_tps"], 16 / 60)
        self.assertEqual(metrics["total_output_tokens"], 23)
        self.assertEqual(metrics["completed_total_output_tokens"], 16)
        self.assertEqual(metrics["e2e_p50_ms"], 10)

    def test_topologies_share_workload_affinity_and_crossover_contract(self) -> None:
        """Regression: topology drift could invalidate the A/B result; dry-run commands expose it."""
        manifest = campaign.load_manifest()
        layout = campaign.CpuLayout(
            frontend="0-7",
            backend="8-23",
            loadgen="24-39",
            infra="40-47",
            physical_cores=48,
        )
        legs = campaign.build_resolved_plan(manifest, selected_shards=1)
        main = [leg for leg in legs if leg["phase"] == "main"]
        expected_points = len(manifest["main_matrix"]["concurrency"])
        expected_legs = expected_points * len(manifest["main_matrix"]["crossover"])
        self.assertEqual(len(main), expected_legs)
        grouped: dict[str, list[dict]] = {}
        for leg in main:
            grouped.setdefault(leg["metadata"]["point_key"], []).append(leg)
        expected_crossover = manifest["main_matrix"]["crossover"]
        self.assertEqual(len(grouped), expected_points)
        self.assertTrue(
            all(
                [leg["arm"] for leg in point] == expected_crossover
                for point in grouped.values()
            )
        )
        mocker_config_path = campaign.REPO_ROOT / manifest["fixture"]["mocker_config"]
        mocker_config = json.loads(mocker_config_path.read_text(encoding="utf-8"))
        self.assertEqual(mocker_config["engine_type"], "vllm")
        self.assertEqual(mocker_config["speedup_ratio"], 0.0)
        self.assertEqual(mocker_config["dp_size"], 1)
        self.assertEqual(mocker_config["worker_type"], "aggregated")
        self.assertEqual(mocker_config["block_size"], 64)
        self.assertFalse(mocker_config["enable_prefix_caching"])
        self.assertEqual(mocker_config["max_num_seqs"], 524288)
        self.assertEqual(manifest["transport"]["max_concurrent_requests"], 524288)
        self.assertEqual(
            manifest["transport"]["dynamo_env"]["DYN_TCP_REQUEST_TIMEOUT"], "300"
        )
        self.assertGreaterEqual(mocker_config["num_gpu_blocks"], 4_194_304)
        model_path = campaign.REPO_ROOT / manifest["fixture"]["model"]
        tokenizer = json.loads(
            (model_path / "tokenizer.json").read_text(encoding="utf-8")
        )
        self.assertGreater(len(tokenizer["model"]["vocab"]), 0)
        tokenizer_config = json.loads(
            (model_path / "tokenizer_config.json").read_text(encoding="utf-8")
        )
        self.assertTrue(tokenizer_config["chat_template"])
        model_config = json.loads(
            (model_path / "config.json").read_text(encoding="utf-8")
        )
        required_context = manifest["workload"]["max_context_length"]
        self.assertGreaterEqual(
            model_config["max_position_embeddings"], required_context
        )
        self.assertGreaterEqual(tokenizer_config["model_max_length"], required_context)

        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            binaries = {
                "vllm_sidecar": "/build/dynamo-vllm-sidecar",
                "vllm_mocker_server": "/build/dynamo-vllm-mocker-server",
            }
            point = next(iter(grouped.values()))
            direct_leg = next(leg for leg in point if leg["arm"] == "direct-mocker")
            sidecar_leg = next(leg for leg in point if leg["arm"] == "sidecar")
            direct = campaign.build_run_perf_command(
                direct_leg, manifest, layout, output_root, "dryrun", binaries
            )
            sidecar = campaign.build_run_perf_command(
                sidecar_leg, manifest, layout, output_root, "dryrun", binaries
            )
            self.assertEqual(
                campaign.flag_value(direct, "--backend-mode"), "direct-vllm-mocker"
            )
            self.assertEqual(
                campaign.flag_value(sidecar, "--backend-mode"), "vllm-sidecar-mocker"
            )
            self.assertNotIn("--vllm-sidecar-bin", direct)
            self.assertIn("--vllm-sidecar-bin", sidecar)
            self.assertNotIn("--request-rate", direct)
            self.assertNotIn("--request-rate", sidecar)
            self.assertIn("--allow-aiperf-saturation-failures", direct)
            self.assertIn("--allow-aiperf-saturation-failures", sidecar)
            self.assertIn("--aiperf-burst-phase-starts", direct)
            self.assertIn("--aiperf-burst-phase-starts", sidecar)
            self.assertTrue(manifest["workload"]["burst_phase_starts"])
            self.assertEqual(
                campaign.flag_value(direct, "--aiperf-loopback-targets"), "8"
            )
            self.assertEqual(
                campaign.flag_value(direct, "--model"),
                campaign.flag_value(direct, "--model-name"),
            )
            for flag in [
                "--model",
                "--model-name",
                "--workers",
                "--data-parallel-size",
                "--request-plane",
                "--event-plane",
                "--mocker-config",
                "--endpoint-type",
                "--frontend-cpuset",
                "--backend-cpuset",
                "--loadgen-cpuset",
                "--infra-cpuset",
                "--concurrency",
                "--aiperf-shards",
                "--require-aiperf-version",
                "--random-seed",
                "--benchmark-grace-period",
                "--request-timeout-seconds",
                "--aiperf-scenario",
                "--aiperf-public-dataset",
                "--aiperf-max-context-length",
                "--aiperf-workers",
                "--aiperf-record-processors",
                "--aiperf-loopback-targets",
                "--max-concurrent-requests",
            ]:
                self.assertEqual(
                    campaign.flag_value(direct, flag),
                    campaign.flag_value(sidecar, flag),
                    flag,
                )
            self.assertEqual(
                campaign.flag_value(direct, "--benchmark-grace-period"), "300"
            )
            self.assertEqual(
                campaign.flag_value(direct, "--request-timeout-seconds"), "300"
            )
            self.assertEqual(
                campaign.flag_value(direct, "--aiperf-scenario"),
                "inferencex-agentx-mvp",
            )
            self.assertEqual(
                campaign.flag_value(direct, "--aiperf-public-dataset"),
                "semianalysis_cc_traces_weka_062126_256k",
            )
            self.assertNotEqual(
                campaign.flag_value(direct, "--namespace"),
                campaign.flag_value(sidecar, "--namespace"),
            )


if __name__ == "__main__":
    unittest.main()
