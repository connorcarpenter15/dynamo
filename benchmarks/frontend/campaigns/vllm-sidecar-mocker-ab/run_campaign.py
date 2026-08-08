#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run and analyze the locked CPU-only vLLM sidecar A/B campaign."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CAMPAIGN_DIR = Path(__file__).resolve().parent
REPO_ROOT = CAMPAIGN_DIR.parents[3]
MANIFEST_PATH = CAMPAIGN_DIR / "manifest.json"
MANIFEST_HASH_PATH = CAMPAIGN_DIR / "manifest.sha256"
RUN_PERF_PATH = REPO_ROOT / "benchmarks/frontend/scripts/run_perf.sh"


class CampaignError(RuntimeError):
    """A locked campaign invariant was violated."""


@dataclass(frozen=True)
class CpuLayout:
    frontend: str
    backend: str
    loadgen: str
    infra: str
    physical_cores: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "frontend": self.frontend,
            "backend": self.backend,
            "loadgen": self.loadgen,
            "infra": self.infra,
            "physical_cores": self.physical_cores,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CpuLayout":
        return cls(
            frontend=value["frontend"],
            backend=value["backend"],
            loadgen=value["loadgen"],
            infra=value["infra"],
            physical_cores=int(value["physical_cores"]),
        )


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    if path == MANIFEST_PATH:
        recorded = MANIFEST_HASH_PATH.read_text(encoding="utf-8").split()[0]
        actual = sha256_file(path)
        if recorded != actual:
            raise CampaignError(f"manifest SHA-256 mismatch: {recorded} != {actual}")
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise CampaignError(
            f"unsupported manifest schema: {manifest.get('schema_version')}"
        )
    return manifest


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def run_text(command: list[str], *, check: bool = True) -> str:
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = result.stdout.strip()
    if check and result.returncode:
        raise CampaignError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{output}"
        )
    return output


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def cpu_list(cpus: Iterable[int]) -> str:
    values = sorted(set(cpus))
    if not values:
        raise CampaignError("CPU set is empty")
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def expand_cpu_list(value: str) -> set[int]:
    result: set[int] = set()
    for part in value.split(","):
        start, separator, end = part.partition("-")
        if separator:
            result.update(range(int(start), int(end) + 1))
        else:
            result.add(int(start))
    return result


def discover_cpu_layout(manifest: dict[str, Any]) -> CpuLayout:
    output = run_text(["lscpu", "--parse=CPU,CORE,SOCKET,ONLINE"])
    allowed_cpus = os.sched_getaffinity(0)
    representatives: dict[tuple[int, int], int] = {}
    for line in output.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split(",")
        if len(fields) != 4 or fields[3].strip().lower() != "y":
            continue
        cpu, core, socket = map(int, fields[:3])
        if cpu not in allowed_cpus:
            continue
        representatives.setdefault((socket, core), cpu)
    cpus = [representatives[key] for key in sorted(representatives)]
    hardware = manifest["hardware"]
    minimum = int(hardware["minimum_physical_cores"])
    if len(cpus) < minimum:
        raise CampaignError(
            f"host has {len(cpus)} physical cores; campaign requires {minimum}"
        )
    roles = hardware["roles"]
    frontend_count = int(roles["frontend_cores"])
    backend_count = int(roles["backend_cores"])
    loadgen_count = int(roles["loadgen_cores"])
    minimum_infra = int(roles["minimum_infra_cores"])
    split_1 = frontend_count
    split_2 = split_1 + backend_count
    split_3 = split_2 + loadgen_count
    if len(cpus) - split_3 < minimum_infra:
        raise CampaignError("CPU role allocation leaves too few infrastructure cores")
    layout = CpuLayout(
        frontend=cpu_list(cpus[:split_1]),
        backend=cpu_list(cpus[split_1:split_2]),
        loadgen=cpu_list(cpus[split_2:split_3]),
        infra=cpu_list(cpus[split_3:]),
        physical_cores=len(cpus),
    )
    sets = [
        expand_cpu_list(layout.frontend),
        expand_cpu_list(layout.backend),
        expand_cpu_list(layout.loadgen),
        expand_cpu_list(layout.infra),
    ]
    if any(
        left & right for index, left in enumerate(sets) for right in sets[index + 1 :]
    ):
        raise CampaignError("generated CPU sets overlap")
    return layout


def read_cpu_stat(cpus: set[int]) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    with Path("/proc/stat").open(encoding="utf-8") as handle:
        for line in handle:
            match = re.match(r"cpu(\d+)\s+(.+)", line)
            if not match:
                continue
            cpu = int(match.group(1))
            if cpu not in cpus:
                continue
            values = [int(value) for value in match.group(2).split()]
            total = sum(values)
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            result[cpu] = (total, idle)
    return result


def sample_busy_fraction(layout: CpuLayout, seconds: int) -> float:
    role_cpus = set().union(
        expand_cpu_list(layout.frontend),
        expand_cpu_list(layout.backend),
        expand_cpu_list(layout.loadgen),
        expand_cpu_list(layout.infra),
    )
    cpus = set(os.sched_getaffinity(0))
    if not role_cpus <= cpus:
        raise CampaignError("CPU role affinity changed after layout discovery")
    before = read_cpu_stat(cpus)
    time.sleep(seconds)
    after = read_cpu_stat(cpus)
    total_delta = 0
    idle_delta = 0
    for cpu in cpus:
        if cpu not in before or cpu not in after:
            raise CampaignError(f"CPU {cpu} disappeared during utilization preflight")
        total_delta += after[cpu][0] - before[cpu][0]
        idle_delta += after[cpu][1] - before[cpu][1]
    if total_delta <= 0:
        raise CampaignError("could not measure host CPU utilization")
    return 1.0 - idle_delta / total_delta


def point_key(input_tokens: int, output_tokens: int, concurrency: int) -> str:
    return f"i{input_tokens}-o{output_tokens}-c{concurrency}"


def make_leg(
    phase: str,
    leg_id: str,
    arm: str,
    input_tokens: int,
    output_tokens: int,
    concurrency: int,
    *,
    grpc_connections: int = 8,
    shards: int = 1,
    profile: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "id": leg_id,
        "arm": arm,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "concurrency": concurrency,
        "grpc_connections": grpc_connections,
        "aiperf_shards": shards,
        "profile": profile,
        "metadata": metadata or {},
    }


def high_concurrency_shards(
    manifest: dict[str, Any], concurrency: int, selected: int
) -> int:
    minimum = int(manifest["loadgen_preflight"]["four_shard_minimum_concurrency"])
    return selected if concurrency >= minimum else 1


def build_smoke_legs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    smoke = manifest["smoke"]
    return [
        make_leg(
            "smoke",
            f"smoke-{arm}",
            arm,
            smoke["input_tokens"],
            smoke["output_tokens"],
            smoke["concurrency"],
            metadata={"request_count": smoke["request_count"]},
        )
        for arm in ("direct-mocker", "sidecar")
    ]


def build_preflight_legs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    preflight = manifest["loadgen_preflight"]
    return [
        make_leg(
            "preflight",
            f"loadgen-preflight-shards-{shards}",
            preflight["arm"],
            preflight["input_tokens"],
            preflight["output_tokens"],
            preflight["concurrency"],
            shards=shards,
            metadata={"candidate_shards": shards},
        )
        for shards in preflight["candidate_shards"]
    ]


def build_main_legs(
    manifest: dict[str, Any], selected_shards: int
) -> list[dict[str, Any]]:
    matrix = manifest["main_matrix"]
    points = [
        (input_tokens, output_tokens, concurrency)
        for input_tokens in matrix["input_tokens"]
        for output_tokens in matrix["output_tokens"]
        for concurrency in matrix["concurrency"]
    ]
    random.Random(matrix["shuffle_seed"]).shuffle(points)
    legs: list[dict[str, Any]] = []
    for point_index, (input_tokens, output_tokens, concurrency) in enumerate(points):
        occurrences = {"direct-mocker": 0, "sidecar": 0}
        for crossover_index, arm in enumerate(matrix["crossover"]):
            occurrence = occurrences[arm]
            occurrences[arm] += 1
            pair_index = 0 if crossover_index < 2 else 1
            key = point_key(input_tokens, output_tokens, concurrency)
            legs.append(
                make_leg(
                    "main",
                    f"main-{point_index:02d}-{key}-x{crossover_index}-{arm}",
                    arm,
                    input_tokens,
                    output_tokens,
                    concurrency,
                    grpc_connections=manifest["transport"][
                        "production_grpc_connections"
                    ],
                    shards=high_concurrency_shards(
                        manifest, concurrency, selected_shards
                    ),
                    metadata={
                        "point_index": point_index,
                        "point_key": key,
                        "crossover_index": crossover_index,
                        "pair_index": pair_index,
                        "arm_occurrence": occurrence,
                    },
                )
            )
    return legs


def build_connection_legs(
    manifest: dict[str, Any], selected_shards: int, main_legs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    diagnostic = manifest["connection_diagnostic"]
    reuse_connections = int(diagnostic["reuse_main_connections"])
    reusable = {
        (
            leg["metadata"]["point_key"],
            leg["metadata"]["arm_occurrence"],
        ): leg["id"]
        for leg in main_legs
        if leg["arm"] == "sidecar"
    }
    legs: list[dict[str, Any]] = []
    for anchor_index, anchor in enumerate(diagnostic["anchors"]):
        key = point_key(
            anchor["input_tokens"], anchor["output_tokens"], anchor["concurrency"]
        )
        for order_index, order in enumerate(diagnostic["orders"]):
            connections = list(diagnostic["connections"])
            if order == "descending":
                connections.reverse()
            for order_position, connections_count in enumerate(connections):
                metadata: dict[str, Any] = {
                    "anchor_index": anchor_index,
                    "point_key": key,
                    "order": order,
                    "order_position": order_position,
                }
                if connections_count == reuse_connections:
                    metadata["reuse_from"] = reusable[(key, order_index)]
                legs.append(
                    make_leg(
                        "connections",
                        f"connections-a{anchor_index}-{order}-g{connections_count}",
                        "sidecar",
                        anchor["input_tokens"],
                        anchor["output_tokens"],
                        anchor["concurrency"],
                        grpc_connections=connections_count,
                        shards=high_concurrency_shards(
                            manifest, anchor["concurrency"], selected_shards
                        ),
                        metadata=metadata,
                    )
                )
    return legs


def build_profile_legs(
    manifest: dict[str, Any], selected_shards: int
) -> list[dict[str, Any]]:
    profiles = manifest["profiles"]
    common = {
        "input_tokens": profiles["input_tokens"],
        "output_tokens": profiles["output_tokens"],
        "concurrency": profiles["concurrency"],
    }
    shards = high_concurrency_shards(manifest, profiles["concurrency"], selected_shards)
    direct_runs = int(profiles["direct_mocker_runs"])
    legs = [
        make_leg(
            "profiles",
            "profile-direct-mocker"
            if direct_runs == 1
            else f"profile-direct-mocker-{index}",
            "direct-mocker",
            **common,
            shards=shards,
            profile=True,
        )
        for index in range(direct_runs)
    ]
    legs.extend(
        make_leg(
            "profiles",
            f"profile-sidecar-g{connections}",
            "sidecar",
            **common,
            grpc_connections=connections,
            shards=shards,
            profile=True,
        )
        for connections in profiles["sidecar_connections"]
    )
    return legs


def build_resolved_plan(
    manifest: dict[str, Any], selected_shards: int
) -> list[dict[str, Any]]:
    if selected_shards not in manifest["loadgen_preflight"]["candidate_shards"]:
        raise CampaignError(f"invalid locked shard count: {selected_shards}")
    main = build_main_legs(manifest, selected_shards)
    legs = (
        build_smoke_legs(manifest)
        + build_preflight_legs(manifest)
        + main
        + build_connection_legs(manifest, selected_shards, main)
        + build_profile_legs(manifest, selected_shards)
    )
    for ordinal, leg in enumerate(legs):
        leg["ordinal"] = ordinal
        leg["output_rel"] = f"legs/{ordinal:04d}-{leg['id']}"
    return legs


def flag_value(command: list[str], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


def resolve_model_identity(manifest: dict[str, Any]) -> tuple[Path, str]:
    """Use the model path as the portable native-sidecar model identity."""
    model = (REPO_ROOT / manifest["fixture"]["model"]).resolve()
    served_model_name = manifest["fixture"]["served_model_name"]
    if served_model_name != {"policy": "resolved-model-path"}:
        raise CampaignError(
            "native vLLM sidecar requires served_model_name.policy=resolved-model-path"
        )
    return model, str(model)


def build_run_perf_command(
    leg: dict[str, Any],
    manifest: dict[str, Any],
    layout: CpuLayout,
    output_root: Path,
    run_id: str,
    binaries: dict[str, str],
    attempt: int = 1,
) -> list[str]:
    model, served_model_name = resolve_model_identity(manifest)
    if manifest["tools"]["request_rate"] != "unlimited":
        raise CampaignError("campaign requires an unlimited request rate")
    mocker_config = REPO_ROOT / manifest["fixture"]["mocker_config"]
    namespace = f"vllm-ab-{run_id}-{leg['ordinal']:04d}"
    output_dir = output_root / leg["output_rel"] / f"attempt-{attempt:03d}"
    command = [
        str(RUN_PERF_PATH),
        "--model",
        str(model),
        "--model-name",
        served_model_name,
        "--workers",
        "1",
        "--data-parallel-size",
        "1",
        "--request-plane",
        manifest["transport"]["request_plane"],
        "--event-plane",
        manifest["transport"]["event_plane"],
        "--backend-mode",
        manifest["arms"][leg["arm"]]["backend_mode"],
        "--grpc-connections",
        str(leg["grpc_connections"]),
        "--grpc-port",
        str(manifest["transport"]["grpc_port"]),
        "--mocker-config",
        str(mocker_config),
        "--endpoint-type",
        manifest["transport"]["endpoint_type"],
        "--namespace",
        namespace,
        "--aiperf-shards",
        str(leg["aiperf_shards"]),
        "--aiperf-export-level",
        manifest["tools"]["aiperf_export_level"],
        "--require-aiperf-version",
        manifest["tools"]["aiperf_version"],
        "--random-seed",
        str(manifest["tools"]["random_seed"]),
        "--exact-input-token-id",
        str(manifest["fixture"]["exact_input_token_id"]),
        "--frontend-cpuset",
        layout.frontend,
        "--backend-cpuset",
        layout.backend,
        "--loadgen-cpuset",
        layout.loadgen,
        "--infra-cpuset",
        layout.infra,
        "--concurrency",
        str(leg["concurrency"]),
        "--isl",
        str(leg["input_tokens"]),
        "--osl",
        str(leg["output_tokens"]),
        "--output-dir",
        str(output_dir),
        "--tokenizer-backend",
        "hf",
        "--require-managed-infra",
    ]
    if leg["arm"] == "sidecar":
        command.extend(
            [
                "--vllm-mocker-server-bin",
                binaries["vllm_mocker_server"],
                "--vllm-sidecar-bin",
                binaries["vllm_sidecar"],
            ]
        )
    if leg["phase"] == "smoke":
        command.extend(
            [
                "--num-requests",
                str(leg["metadata"]["request_count"]),
                "--warmup-count",
                "1",
                "--capture-duration",
                "10",
            ]
        )
    else:
        timing = manifest["timing"]
        command.extend(
            [
                "--benchmark-duration",
                str(timing["measurement_seconds"]),
                "--benchmark-grace-period",
                str(timing["measurement_grace_seconds"]),
                "--request-timeout-seconds",
                str(manifest["tools"]["request_timeout_seconds"]),
                "--warmup-duration",
                str(timing["warmup_seconds"]),
                "--warmup-grace-period",
                str(timing["warmup_grace_seconds"]),
                "--aiperf-dataset-entries",
                str(manifest["tools"]["num_dataset_entries"]),
                "--allow-aiperf-timeout-failures",
                "--capture-duration",
                str(
                    timing["measurement_seconds"]
                    + timing["warmup_seconds"]
                    + timing["cooldown_seconds"]
                ),
            ]
        )
    if leg["profile"]:
        command.extend(
            ["--profile-target", manifest["profiles"]["profile_target"], "--skip-nsys"]
        )
    else:
        command.extend(
            ["--skip-nsys", "--skip-perf", "--skip-bpf", "--skip-flamegraph"]
        )
    return command


def resolve_binary(explicit: str | None, name: str, fallback: Path) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        discovered = shutil.which(name)
        path = Path(discovered).resolve() if discovered else fallback.resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise CampaignError(f"required executable not found: {path}")
    return path


def cargo_release_binary(name: str) -> Path:
    target_dir = Path(os.environ.get("CARGO_TARGET_DIR", REPO_ROOT / "target"))
    return target_dir.expanduser() / "release" / name


def verify_source(manifest: dict[str, Any]) -> dict[str, Any]:
    source = manifest["source"]
    head = run_text(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    required = source["required_ancestor"]
    ancestor = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", required, head],
        check=False,
    )
    if ancestor.returncode:
        raise CampaignError(
            f"campaign source {head} does not descend from required {required}"
        )
    status = run_text(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])
    if source["require_clean_worktree"] and status:
        raise CampaignError("campaign requires a clean committed worktree")
    return {"head": head, "required_ancestor": required, "clean": not bool(status)}


def aiperf_version(required: str) -> str:
    executable = shutil.which("aiperf")
    if not executable:
        raise CampaignError("aiperf is not installed")
    output = run_text([executable, "--version"], check=False)
    if not re.search(rf"(^|\D){re.escape(required)}(\D|$)", output):
        raise CampaignError(f"required AIPerf {required}, got {output or 'unknown'}")
    return output


def python_runtime_inventory() -> dict[str, str]:
    script = """
import importlib.util
import json

names = ["dynamo._core", "dynamo.frontend", "dynamo.mocker"]
origins = {}
for name in names:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise SystemExit(f"missing Python module: {name}")
    origins[name] = spec.origin
print(json.dumps(origins, sort_keys=True))
"""
    origins = json.loads(run_text([sys.executable, "-c", script]))
    for module in ("dynamo.frontend", "dynamo.mocker"):
        try:
            Path(origins[module]).resolve().relative_to(REPO_ROOT)
        except ValueError as error:
            raise CampaignError(
                f"{module} is not installed from the campaign worktree: {origins[module]}"
            ) from error
    return origins


def lock_manifest(output_root: Path) -> str:
    source_bytes = MANIFEST_PATH.read_bytes()
    digest = sha256_bytes(source_bytes)
    locked_manifest = output_root / "manifest.json"
    locked_hash = output_root / "manifest.sha256"
    output_root.mkdir(parents=True, exist_ok=True)
    if locked_manifest.exists() and locked_manifest.read_bytes() != source_bytes:
        raise CampaignError("output manifest differs from the repository manifest")
    if not locked_manifest.exists():
        locked_manifest.write_bytes(source_bytes)
    expected_hash_text = f"{digest}  manifest.json\n"
    if (
        locked_hash.exists()
        and locked_hash.read_text(encoding="utf-8") != expected_hash_text
    ):
        raise CampaignError("output manifest SHA-256 does not match")
    if not locked_hash.exists():
        locked_hash.write_text(expected_hash_text, encoding="utf-8")
    return digest


def prepare_environment(
    manifest: dict[str, Any],
    output_root: Path,
    sidecar_bin: str | None,
    mocker_server_bin: str | None,
) -> dict[str, Any]:
    environment_path = output_root / "environment.json"
    if environment_path.exists():
        with environment_path.open(encoding="utf-8") as handle:
            environment = json.load(handle)
        source = verify_source(manifest)
        if source != environment["source"]:
            raise CampaignError(
                "source commit or cleanliness changed since campaign initialization"
            )
        if run_text(["hostname"]) != environment["hostname"]:
            raise CampaignError("campaign resume attempted on a different host")
        current_hashes = {
            "cargo_lock": sha256_file(REPO_ROOT / "Cargo.lock"),
            "run_perf": sha256_file(RUN_PERF_PATH),
            "vllm_grpc_schema": sha256_file(REPO_ROOT / manifest["protocol"]["schema"]),
            "mocker_config": sha256_file(
                REPO_ROOT / manifest["fixture"]["mocker_config"]
            ),
            "vllm_sidecar": sha256_file(Path(environment["binaries"]["vllm_sidecar"])),
            "vllm_mocker_server": sha256_file(
                Path(environment["binaries"]["vllm_mocker_server"])
            ),
            "python_binding": sha256_file(
                Path(environment["python_modules"]["dynamo._core"])
            ),
        }
        if current_hashes != environment["hashes"]:
            raise CampaignError(
                "source, configuration, or binary hashes changed since initialization"
            )
        layout = CpuLayout.from_dict(environment["cpu_layout"])
        busy_fraction = sample_busy_fraction(
            layout, int(manifest["hardware"]["busy_sample_seconds"])
        )
        if busy_fraction > float(
            manifest["hardware"]["maximum_preflight_busy_fraction"]
        ):
            raise CampaignError(
                f"host busy fraction {busy_fraction:.3%} exceeds the locked limit"
            )
        aiperf_version(manifest["tools"]["aiperf_version"])
        return environment

    source = verify_source(manifest)
    for command in ["taskset", "lscpu", "jq", "ss", "etcd", "nats-server", "nc"]:
        if not shutil.which(command):
            raise CampaignError(f"required command is not installed: {command}")
    sidecar = resolve_binary(
        sidecar_bin,
        "dynamo-vllm-sidecar",
        cargo_release_binary("dynamo-vllm-sidecar"),
    )
    mocker_server = resolve_binary(
        mocker_server_bin,
        "dynamo-vllm-mocker-server",
        cargo_release_binary("dynamo-vllm-mocker-server"),
    )
    layout = discover_cpu_layout(manifest)
    hardware = manifest["hardware"]
    busy_fraction = sample_busy_fraction(layout, int(hardware["busy_sample_seconds"]))
    if busy_fraction > float(hardware["maximum_preflight_busy_fraction"]):
        raise CampaignError(
            f"host busy fraction {busy_fraction:.3%} exceeds "
            f"{float(hardware['maximum_preflight_busy_fraction']):.3%}"
        )
    protocol = REPO_ROOT / manifest["protocol"]["schema"]
    mocker_config = REPO_ROOT / manifest["fixture"]["mocker_config"]
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S")
    safe_environment = {
        key: os.environ[key]
        for key in [
            "CARGO_HOME",
            "CARGO_TARGET_DIR",
            "HF_HOME",
            "PIP_CACHE_DIR",
            "RUSTUP_HOME",
            "TMPDIR",
            "UV_CACHE_DIR",
            "VIRTUAL_ENV",
            "XDG_CACHE_HOME",
        ]
        if key in os.environ
    }
    python_modules = python_runtime_inventory()
    environment = {
        "created_at": utc_now(),
        "run_id": run_id,
        "hostname": run_text(["hostname"]),
        "source": source,
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "cpu_layout": layout.as_dict(),
        "preflight_busy_fraction": busy_fraction,
        "aiperf_version": aiperf_version(manifest["tools"]["aiperf_version"]),
        "tool_versions": {
            "python": run_text([sys.executable, "--version"]),
            "cargo": run_text(["cargo", "--version"]),
            "rustc": run_text(["rustc", "--version"]),
        },
        "hashes": {
            "cargo_lock": sha256_file(REPO_ROOT / "Cargo.lock"),
            "run_perf": sha256_file(RUN_PERF_PATH),
            "vllm_grpc_schema": sha256_file(protocol),
            "mocker_config": sha256_file(mocker_config),
            "vllm_sidecar": sha256_file(sidecar),
            "vllm_mocker_server": sha256_file(mocker_server),
            "python_binding": sha256_file(Path(python_modules["dynamo._core"])),
        },
        "binaries": {
            "vllm_sidecar": str(sidecar),
            "vllm_mocker_server": str(mocker_server),
        },
        "python_modules": python_modules,
        "safe_environment": safe_environment,
        "uname": run_text(["uname", "-a"]),
        "lscpu": run_text(["lscpu"]),
    }
    atomic_write_json(environment_path, environment)
    return environment


def lock_resolved_plan(output_root: Path, legs: list[dict[str, Any]]) -> None:
    path = output_root / "resolved-plan.json"
    value = {"schema_version": 1, "legs": legs}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if canonical_json(existing) != canonical_json(value):
            raise CampaignError("resolved plan differs from the already locked plan")
        return
    atomic_write_json(path, value)
    (output_root / "resolved-plan.sha256").write_text(
        f"{sha256_bytes(canonical_json(value))}  resolved-plan.json\n", encoding="utf-8"
    )


def load_state(output_root: Path) -> dict[str, Any]:
    path = output_root / "state.json"
    if not path.exists():
        return {"schema_version": 1, "legs": {}}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_state(output_root: Path, state: dict[str, Any]) -> None:
    atomic_write_json(output_root / "state.json", state)


def metric_average(data: dict[str, Any], name: str) -> float:
    value = data.get(name, {})
    return float(value.get("avg", 0.0)) if isinstance(value, dict) else 0.0


def error_count(value: Any) -> float:
    if isinstance(value, list):
        return sum(error_count(item) for item in value)
    if isinstance(value, dict):
        for key in ("count", "errors", "failed_requests"):
            if key in value and isinstance(value[key], (int, float)):
                return float(value[key])
        return sum(error_count(item) for item in value.values())
    return 0.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def pooled_record_metrics(
    paths: list[Path],
) -> tuple[dict[str, list[float]], set[str], dict[str, int]]:
    metric_names = (
        "time_to_first_token",
        "inter_token_latency",
        "request_latency",
        "output_sequence_length",
        "input_sequence_length",
    )
    pooled = {name: [] for name in metric_names}
    request_ids: set[str] = set()
    outcomes = {"completed": 0, "failed": 0}
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="strict").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise CampaignError(f"invalid JSONL at {path}:{line_number}") from error
            if record.get("metadata", {}).get("benchmark_phase") != "profiling":
                continue
            if record.get("error") is not None:
                outcomes["failed"] += 1
                continue
            outcomes["completed"] += 1
            request_id = record.get("metadata", {}).get("x_request_id")
            if not request_id:
                raise CampaignError(f"profiling record lacks x_request_id in {path}")
            request_ids.add(str(request_id))
            metrics = record.get("metrics", {})
            for name in metric_names:
                value = metrics.get(name)
                if not isinstance(value, dict) or value.get("value") is None:
                    continue
                expected_unit = "tokens" if name.endswith("sequence_length") else "ms"
                if value.get("unit") != expected_unit:
                    raise CampaignError(
                        f"unexpected {name} unit in {path}: {value.get('unit')}"
                    )
                pooled[name].append(float(value["value"]))
    return pooled, request_ids, outcomes


def parse_frontend_token_metrics(
    frontend_log: Path, request_ids: set[str]
) -> dict[str, float]:
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    request_id_pattern = re.compile(r'x_request_id="?([^"\s]+)"?')
    token_pattern = re.compile(r"\binput_tokens=(\d+)\s+output_tokens=(\d+)\b")
    matched: set[str] = set()
    input_tokens: list[int] = []
    output_tokens: list[int] = []
    for line in frontend_log.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        if "request completed" not in line:
            continue
        clean = ansi_escape.sub("", line)
        request_id_match = request_id_pattern.search(clean)
        if not request_id_match or request_id_match.group(1) not in request_ids:
            continue
        request_id = request_id_match.group(1)
        if request_id in matched:
            raise CampaignError(
                f"duplicate frontend completion for {request_id} in {frontend_log}"
            )
        token_match = token_pattern.search(clean)
        if not token_match:
            raise CampaignError(
                f"frontend completion lacks token totals for "
                f"{request_id} in {frontend_log}"
            )
        matched.add(request_id)
        input_tokens.append(int(token_match.group(1)))
        output_tokens.append(int(token_match.group(2)))
    missing = request_ids - matched
    if missing:
        raise CampaignError(
            f"frontend token totals missing for {len(missing)} profiling requests "
            f"in {frontend_log}"
        )
    return {
        "request_count": float(len(matched)),
        "total_input_tokens": float(sum(input_tokens)),
        "total_output_tokens": float(sum(output_tokens)),
        "input_length_min": float(min(input_tokens, default=0)),
        "input_length_max": float(max(input_tokens, default=0)),
        "output_length_min": float(min(output_tokens, default=0)),
        "output_length_max": float(max(output_tokens, default=0)),
    }


def parse_process_metrics(system_dir: Path) -> dict[str, Any]:
    clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    processes: dict[str, Any] = {}
    for stat_path in sorted(system_dir.glob("proc_stat_*.txt")):
        label = stat_path.stem.removeprefix("proc_stat_")
        samples: list[float] = []
        for line in stat_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if not line or line.startswith("---"):
                continue
            fields = line.split()
            if len(fields) > 14:
                samples.append((int(fields[13]) + int(fields[14])) / clock_ticks)
        if samples:
            processes[label] = {"cpu_seconds": max(0.0, samples[-1] - samples[0])}
    for status_path in sorted(system_dir.glob("proc_status_*.txt")):
        label = status_path.stem.removeprefix("proc_status_")
        entry = processes.setdefault(label, {})
        rss_values = [
            int(match.group(1))
            for match in re.finditer(
                r"^VmRSS:\s+(\d+)\s+kB$",
                status_path.read_text(encoding="utf-8", errors="replace"),
                re.MULTILINE,
            )
        ]
        entry["max_rss_kib"] = max(rss_values, default=0)
        for field in ("voluntary_ctxt_switches", "nonvoluntary_ctxt_switches"):
            values = [
                int(match.group(1))
                for match in re.finditer(
                    rf"^{field}:\s+(\d+)$",
                    status_path.read_text(encoding="utf-8", errors="replace"),
                    re.MULTILINE,
                )
            ]
            entry[field] = max(0, values[-1] - values[0]) if values else 0
    backend_labels = [label for label in processes if label != "frontend"]
    return {
        "processes": processes,
        "combined_backend": {
            "cpu_seconds": sum(
                processes[label].get("cpu_seconds", 0.0) for label in backend_labels
            ),
            "max_rss_kib_sum": sum(
                processes[label].get("max_rss_kib", 0) for label in backend_labels
            ),
            "context_switches": sum(
                processes[label].get("voluntary_ctxt_switches", 0)
                + processes[label].get("nonvoluntary_ctxt_switches", 0)
                for label in backend_labels
            ),
        },
    }


def parse_leg_metrics(output_dir: Path) -> dict[str, Any]:
    profiles = sorted((output_dir / "aiperf").rglob("profile_export_aiperf.json"))
    record_paths = sorted((output_dir / "aiperf").rglob("profile_export.jsonl"))
    if not record_paths:
        raise CampaignError(f"missing AIPerf records export in {output_dir}")
    if profiles and len(record_paths) != len(profiles):
        raise CampaignError(
            f"expected one AIPerf records export per shard in {output_dir}: "
            f"{len(record_paths)} != {len(profiles)}"
        )
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in profiles]
    records, request_ids, outcomes = pooled_record_metrics(record_paths)
    server_tokens = parse_frontend_token_metrics(
        output_dir / "logs/frontend.log", request_ids
    )
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    if shards:
        completed = sum(metric_average(shard, "request_count") for shard in shards)
        failures = sum(
            metric_average(shard, "error_request_count") for shard in shards
        )
        if failures == 0:
            failures = sum(
                error_count(shard.get("error_summary", [])) for shard in shards
            )
        durations = [metric_average(shard, "benchmark_duration") for shard in shards]
        wall_seconds = max(durations, default=0.0)
        versions = sorted(
            {str(shard.get("aiperf_version", "unknown")) for shard in shards}
        )
    else:
        completed = float(outcomes["completed"])
        failures = float(outcomes["failed"])
        wall_seconds = float(config.get("benchmark_duration") or 0.0)
        versions = [str(config.get("aiperf_version", "unknown"))]
    if completed != float(outcomes["completed"]):
        raise CampaignError(
            f"AIPerf summary completed {completed:g} requests but records contain "
            f"{outcomes['completed']} successes"
        )
    if failures != float(outcomes["failed"]):
        raise CampaignError(
            f"AIPerf summary recorded {failures:g} failures but records contain "
            f"{outcomes['failed']} errors"
        )
    if completed != server_tokens["request_count"]:
        raise CampaignError(
            f"AIPerf completed {completed:g} requests but frontend token totals "
            f"matched {server_tokens['request_count']:g}"
        )
    aiperf_total_input_tokens = sum(
        metric_average(shard, "total_isl") for shard in shards
    )
    aiperf_total_output_tokens = sum(
        metric_average(shard, "total_output_tokens") for shard in shards
    )
    aiperf_total_output_sequence_tokens = sum(
        metric_average(shard, "total_osl") for shard in shards
    )
    metric = {
        "shards": len(record_paths),
        "request_throughput_rps": completed / wall_seconds if wall_seconds else 0.0,
        "output_throughput_tps": server_tokens["total_output_tokens"] / wall_seconds
        if wall_seconds
        else 0.0,
        "completed_requests": completed,
        "failed_requests": failures,
        "failed_request_fraction": failures / (completed + failures)
        if completed + failures
        else 0.0,
        "total_input_tokens": server_tokens["total_input_tokens"],
        "total_output_tokens": server_tokens["total_output_tokens"],
        "aiperf_total_input_tokens": aiperf_total_input_tokens,
        "aiperf_total_output_tokens": aiperf_total_output_tokens,
        "aiperf_total_output_sequence_tokens": aiperf_total_output_sequence_tokens,
        "wall_seconds": wall_seconds,
        "ttft_p50_ms": percentile(records["time_to_first_token"], 0.50),
        "ttft_p99_ms": percentile(records["time_to_first_token"], 0.99),
        "itl_p50_ms": percentile(records["inter_token_latency"], 0.50),
        "itl_p99_ms": percentile(records["inter_token_latency"], 0.99),
        "e2e_p50_ms": percentile(records["request_latency"], 0.50),
        "e2e_p99_ms": percentile(records["request_latency"], 0.99),
        "input_length_min": server_tokens["input_length_min"],
        "input_length_max": server_tokens["input_length_max"],
        "output_length_min": server_tokens["output_length_min"],
        "output_length_max": server_tokens["output_length_max"],
        "aiperf_input_length_min": min(
            records["input_sequence_length"], default=0.0
        ),
        "aiperf_input_length_max": max(
            records["input_sequence_length"], default=0.0
        ),
        "aiperf_output_length_min": min(
            records["output_sequence_length"], default=0.0
        ),
        "aiperf_output_length_max": max(
            records["output_sequence_length"], default=0.0
        ),
        "aiperf_versions": versions,
    }
    metric["process_metrics"] = parse_process_metrics(output_dir / "system")
    grpc_check = output_dir / "system/grpc_connection_check.json"
    metric["grpc_connection_check"] = (
        json.loads(grpc_check.read_text(encoding="utf-8"))
        if grpc_check.exists()
        else None
    )
    return metric


def validate_leg_output(
    output_dir: Path, leg: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    config_path = output_dir / "config.json"
    models_path = output_dir / "system/models.json"
    cleanup_path = output_dir / "system/cleanup.json"
    process_health_path = output_dir / "system/process_health.json"
    for required in (config_path, models_path, cleanup_path, process_health_path):
        if not required.exists():
            raise CampaignError(f"missing required artifact: {required}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("backend_mode") != manifest["arms"][leg["arm"]]["backend_mode"]:
        raise CampaignError(f"backend mode mismatch in {config_path}")
    if config.get("source_dirty"):
        raise CampaignError("timed leg ran from a dirty source tree")
    model, served_model_name = resolve_model_identity(manifest)
    expected_config = {
        "model": str(model),
        "model_name": served_model_name,
        "request_plane": manifest["transport"]["request_plane"],
        "event_plane": manifest["transport"]["event_plane"],
        "endpoint_type": manifest["transport"]["endpoint_type"],
        "grpc_connections": leg["grpc_connections"],
        "aiperf_shards": leg["aiperf_shards"],
        "aiperf_export_level": manifest["tools"]["aiperf_export_level"],
        "random_seed": manifest["tools"]["random_seed"],
        "exact_input_token_id": manifest["fixture"]["exact_input_token_id"],
        "benchmark_grace_period": manifest["timing"][
            "measurement_grace_seconds"
        ],
        "request_timeout_seconds": manifest["tools"]["request_timeout_seconds"],
        "warmup_grace_period": manifest["timing"]["warmup_grace_seconds"],
        "aiperf_dataset_entries": manifest["tools"]["num_dataset_entries"],
        "allow_aiperf_timeout_failures": manifest["tools"][
            "allow_aiperf_timeout_failures"
        ],
        "concurrency": leg["concurrency"],
        "isl": leg["input_tokens"],
        "osl": leg["output_tokens"],
    }
    mismatches = {
        key: {"expected": expected, "observed": config.get(key)}
        for key, expected in expected_config.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise CampaignError(f"locked leg config mismatch: {mismatches}")
    expected_mocker_hash = sha256_file(REPO_ROOT / manifest["fixture"]["mocker_config"])
    if config.get("mocker_config_sha256") != expected_mocker_hash:
        raise CampaignError(f"Mocker config hash mismatch in {config_path}")
    accepted_timeouts = bool(config.get("aiperf_timeout_failures_accepted"))
    if config.get("aiperf_failed") and not (
        manifest["tools"]["allow_aiperf_timeout_failures"] and accepted_timeouts
    ):
        raise CampaignError(f"an AIPerf shard failed in {output_dir}")
    if accepted_timeouts and not config.get("aiperf_failed"):
        raise CampaignError("AIPerf timeout acceptance was set without a failed shard")
    models = json.loads(models_path.read_text(encoding="utf-8"))
    model_ids = {entry.get("id") for entry in models.get("data", [])}
    if served_model_name not in model_ids:
        raise CampaignError(f"expected model registration is absent in {models_path}")
    cleanup = json.loads(cleanup_path.read_text(encoding="utf-8"))
    if not cleanup.get("clean"):
        raise CampaignError(f"process cleanup was incomplete in {output_dir}")
    process_health = json.loads(process_health_path.read_text(encoding="utf-8"))
    if not process_health.get("valid"):
        raise CampaignError(f"a frontend or backend process exited in {output_dir}")
    metrics = parse_leg_metrics(output_dir)
    expected_version = manifest["tools"]["aiperf_version"]
    if metrics["aiperf_versions"] != [expected_version]:
        raise CampaignError(
            f"AIPerf export version mismatch: {metrics['aiperf_versions']} != {[expected_version]}"
        )
    if leg["arm"] == "sidecar":
        check = metrics["grpc_connection_check"]
        if not check or not check.get("valid"):
            raise CampaignError(f"invalid sidecar connection pool in {output_dir}")
        if int(check["observed"]) != int(leg["grpc_connections"]):
            raise CampaignError(
                f"sidecar socket count does not match the locked leg in {output_dir}"
            )
    if metrics["completed_requests"]:
        expected_input = float(leg["input_tokens"])
        expected_output = float(leg["output_tokens"])
        if (
            metrics["input_length_min"] != expected_input
            or metrics["input_length_max"] != expected_input
        ):
            raise CampaignError(
                f"input length was not exactly {expected_input:g}: "
                f"{metrics['input_length_min']:g}..{metrics['input_length_max']:g}"
            )
        if (
            metrics["output_length_min"] != expected_output
            or metrics["output_length_max"] != expected_output
        ):
            raise CampaignError(
                f"output length was not exactly {expected_output:g}: "
                f"{metrics['output_length_min']:g}..{metrics['output_length_max']:g}"
            )
    if leg["phase"] == "smoke":
        expected_tokens = float(manifest["smoke"]["expected_output_tokens"])
        if (
            metrics["output_length_min"] != expected_tokens
            or metrics["output_length_max"] != expected_tokens
        ):
            raise CampaignError(
                f"smoke output length was not exactly {expected_tokens:g}: "
                f"{metrics['output_length_min']:g}..{metrics['output_length_max']:g}"
            )
        if metrics["failed_requests"]:
            raise CampaignError("smoke request failed")
    if leg["profile"]:
        profile_artifacts = [
            path
            for path in (output_dir / "perf").glob("*cpu_flamegraph*")
            if path.is_file() and path.stat().st_size > 0
        ]
        if not profile_artifacts:
            raise CampaignError(f"CPU profile artifact is absent in {output_dir}")
    return metrics


def leg_fingerprint(leg: dict[str, Any], manifest_sha256: str) -> str:
    return sha256_bytes(
        canonical_json({"manifest_sha256": manifest_sha256, "leg": leg})
    )


def execute_leg(
    leg: dict[str, Any],
    manifest: dict[str, Any],
    manifest_sha256: str,
    environment: dict[str, Any],
    output_root: Path,
    state: dict[str, Any],
    *,
    retry_failed: bool,
) -> None:
    entries = state["legs"]
    existing = entries.get(leg["id"])
    fingerprint = leg_fingerprint(leg, manifest_sha256)
    if existing and existing.get("fingerprint") != fingerprint:
        raise CampaignError(f"locked leg changed: {leg['id']}")
    if existing and existing.get("status") in {"complete", "reused"}:
        return
    if (
        existing
        and existing.get("status") in {"failed", "running"}
        and not retry_failed
    ):
        raise CampaignError(
            f"leg {leg['id']} previously {existing['status']}; use --retry-failed to repeat it unchanged"
        )

    reuse_from = leg["metadata"].get("reuse_from")
    if reuse_from:
        source = entries.get(reuse_from)
        if source and source.get("status") in {"complete", "reused"}:
            entries[leg["id"]] = {
                "status": "reused",
                "fingerprint": fingerprint,
                "source_leg": reuse_from,
                "output_dir": source["output_dir"],
                "metrics": source["metrics"],
                "finished_at": utc_now(),
            }
            save_state(output_root, state)
            return
        raise CampaignError(
            f"leg {leg['id']} requires valid reusable main leg {reuse_from}"
        )

    attempt_number = len(existing.get("attempts", [])) + 1 if existing else 1
    command = build_run_perf_command(
        leg,
        manifest,
        CpuLayout.from_dict(environment["cpu_layout"]),
        output_root,
        environment["run_id"],
        environment["binaries"],
        attempt=attempt_number,
    )
    output_dir = Path(flag_value(command, "--output-dir"))
    attempt = {
        "attempt": attempt_number,
        "started_at": utc_now(),
        "command": command,
        "command_sha256": sha256_bytes(canonical_json(command)),
        "output_dir": str(output_dir),
    }
    attempts = list(existing.get("attempts", [])) if existing else []
    attempts.append(attempt)
    entries[leg["id"]] = {
        "status": "running",
        "fingerprint": fingerprint,
        "leg": leg,
        "attempts": attempts,
    }
    save_state(output_root, state)
    log_path = (
        output_root
        / "driver-logs"
        / f"{leg['ordinal']:04d}-{leg['id']}-attempt-{attempt_number:03d}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"started_at={attempt['started_at']}\n")
        log.write(f"command={json.dumps(command)}\n")
        log.flush()
        result = subprocess.run(
            command, check=False, stdout=log, stderr=subprocess.STDOUT
        )
    attempt["finished_at"] = utc_now()
    attempt["exit_code"] = result.returncode
    if result.returncode:
        entries[leg["id"]]["status"] = "failed"
        entries[leg["id"]]["failure"] = f"run_perf.sh exited {result.returncode}"
        save_state(output_root, state)
        raise CampaignError(
            f"leg {leg['id']} failed with exit code {result.returncode}; see {log_path}"
        )
    try:
        metrics = validate_leg_output(output_dir, leg, manifest)
    except Exception as error:
        entries[leg["id"]]["status"] = "failed"
        entries[leg["id"]]["failure"] = str(error)
        save_state(output_root, state)
        raise
    entries[leg["id"]].update(
        {
            "status": "complete",
            "output_dir": str(output_dir),
            "metrics": metrics,
            "finished_at": utc_now(),
        }
    )
    save_state(output_root, state)


def derive_shard_decision(
    manifest: dict[str, Any], output_root: Path, state: dict[str, Any]
) -> int:
    candidates = manifest["loadgen_preflight"]["candidate_shards"]
    results: dict[int, float] = {}
    for shards in candidates:
        entry = state["legs"].get(f"loadgen-preflight-shards-{shards}")
        if not entry or entry.get("status") != "complete":
            raise CampaignError(
                "both load-generator preflight legs must complete first"
            )
        results[int(shards)] = float(entry["metrics"]["output_throughput_tps"])
    baseline = results[int(candidates[0])]
    if baseline <= 0:
        raise CampaignError("single-shard preflight produced no output throughput")
    improvement = results[int(candidates[1])] / baseline - 1.0
    threshold = float(manifest["loadgen_preflight"]["improvement_threshold"])
    selected = int(candidates[1]) if improvement > threshold else int(candidates[0])
    decision = {
        "created_at": utc_now(),
        "throughput_tps": {str(key): value for key, value in results.items()},
        "improvement_fraction": improvement,
        "threshold_fraction": threshold,
        "selected_high_concurrency_shards": selected,
    }
    decision_path = output_root / "decisions.json"
    if decision_path.exists():
        existing = json.loads(decision_path.read_text(encoding="utf-8"))
        existing.pop("created_at", None)
        comparable = dict(decision)
        comparable.pop("created_at", None)
        if canonical_json(existing) != canonical_json(comparable):
            raise CampaignError(
                "load-generator decision differs from the locked decision"
            )
    else:
        atomic_write_json(decision_path, decision)
    return selected


def load_shard_decision(output_root: Path) -> int:
    path = output_root / "decisions.json"
    if not path.exists():
        raise CampaignError("run the preflight phase before the locked sweep")
    return int(
        json.loads(path.read_text(encoding="utf-8"))["selected_high_concurrency_shards"]
    )


def execute_phase(
    phase: str,
    legs: list[dict[str, Any]],
    manifest: dict[str, Any],
    manifest_sha256: str,
    environment: dict[str, Any],
    output_root: Path,
    state: dict[str, Any],
    retry_failed: bool,
) -> None:
    cooldown = int(manifest["timing"]["cooldown_seconds"])
    phase_legs = [leg for leg in legs if leg["phase"] == phase]
    for index, leg in enumerate(phase_legs):
        before = state["legs"].get(leg["id"], {}).get("status")
        execute_leg(
            leg,
            manifest,
            manifest_sha256,
            environment,
            output_root,
            state,
            retry_failed=retry_failed,
        )
        after = state["legs"].get(leg["id"], {}).get("status")
        if (
            before not in {"complete", "reused"}
            and after == "complete"
            and index + 1 < len(phase_legs)
        ):
            time.sleep(cooldown)


def run_campaign(args: argparse.Namespace) -> None:
    manifest = load_manifest()
    output_root = args.output_root.expanduser().resolve()
    try:
        output_root.relative_to(REPO_ROOT)
    except ValueError:
        pass
    else:
        raise CampaignError("campaign output must be outside the Git worktree")
    environment = prepare_environment(
        manifest, output_root, args.vllm_sidecar_bin, args.vllm_mocker_server_bin
    )
    manifest_sha256 = lock_manifest(output_root)
    state = load_state(output_root)

    if args.phase in {"smoke", "all"}:
        provisional = build_resolved_plan(manifest, 1)
        execute_phase(
            "smoke",
            provisional,
            manifest,
            manifest_sha256,
            environment,
            output_root,
            state,
            args.retry_failed,
        )
    if args.phase in {"preflight", "all"}:
        provisional = build_resolved_plan(manifest, 1)
        execute_phase(
            "preflight",
            provisional,
            manifest,
            manifest_sha256,
            environment,
            output_root,
            state,
            args.retry_failed,
        )
        selected_shards = derive_shard_decision(manifest, output_root, state)
    else:
        selected_shards = (
            load_shard_decision(output_root) if args.phase not in {"smoke"} else 1
        )

    if args.phase == "smoke":
        return
    legs = build_resolved_plan(manifest, selected_shards)
    lock_resolved_plan(output_root, legs)
    phases = (
        ["main", "connections", "profiles"] if args.phase == "all" else [args.phase]
    )
    for phase in phases:
        if phase == "preflight":
            continue
        execute_phase(
            phase,
            legs,
            manifest,
            manifest_sha256,
            environment,
            output_root,
            state,
            args.retry_failed,
        )
    if args.phase == "all":
        analyze_campaign(output_root)


def median(entries: list[dict[str, Any]], field: str) -> float:
    values = [float(entry[field]) for entry in entries]
    return statistics.median(values) if values else 0.0


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_ratio_chart(
    path: Path,
    rows: list[dict[str, Any]],
    field: str,
    title: str,
    threshold: float,
) -> None:
    width, height = 1100, 620
    left, right, top, bottom = 90, 40, 70, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [float(row[field]) for row in rows if math.isfinite(float(row[field]))]
    y_min = min([0.75, threshold - 0.05] + values)
    y_max = max([1.25, threshold + 0.05] + values)
    padding = max(0.05, (y_max - y_min) * 0.1)
    y_min -= padding
    y_max += padding
    concurrency_values = sorted({int(row["concurrency"]) for row in rows})
    x_positions = {
        value: left + index * plot_width / max(1, len(concurrency_values) - 1)
        for index, value in enumerate(concurrency_values)
    }

    def y_position(value: float) -> float:
        return top + (y_max - value) * plot_height / (y_max - y_min)

    colors = ["#2f6fed", "#d1495b", "#2a9d8f", "#8f5bd7"]
    shapes = sorted(
        {(int(row["input_tokens"]), int(row["output_tokens"])) for row in rows}
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="22">{title}</text>',
    ]
    for tick_index in range(6):
        value = y_min + tick_index * (y_max - y_min) / 5
        y = y_position(value)
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#dddddd"/>'
        )
        lines.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.2f}</text>'
        )
    threshold_y = y_position(threshold)
    lines.append(
        f'<line x1="{left}" y1="{threshold_y:.1f}" x2="{width - right}" y2="{threshold_y:.1f}" stroke="#333333" stroke-dasharray="7,5"/>'
    )
    for concurrency, x in x_positions.items():
        lines.append(
            f'<text x="{x:.1f}" y="{height - bottom + 28}" text-anchor="middle" font-family="sans-serif" font-size="12">{concurrency}</text>'
        )
    for shape_index, shape in enumerate(shapes):
        color = colors[shape_index % len(colors)]
        shape_rows = sorted(
            [
                row
                for row in rows
                if (int(row["input_tokens"]), int(row["output_tokens"])) == shape
            ],
            key=lambda row: int(row["concurrency"]),
        )
        points = " ".join(
            f"{x_positions[int(row['concurrency'])]:.1f},{y_position(float(row[field])):.1f}"
            for row in shape_rows
            if math.isfinite(float(row[field]))
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
        for row in shape_rows:
            value = float(row[field])
            if math.isfinite(value):
                lines.append(
                    f'<circle cx="{x_positions[int(row["concurrency"])]:.1f}" cy="{y_position(value):.1f}" r="4" fill="{color}"/>'
                )
        legend_y = top + shape_index * 24
        lines.append(
            f'<rect x="{width - right - 175}" y="{legend_y - 10}" width="14" height="14" fill="{color}"/>'
        )
        lines.append(
            f'<text x="{width - right - 155}" y="{legend_y + 2}" font-family="sans-serif" font-size="13">{shape[0]}×{shape[1]}</text>'
        )
    lines.extend(
        [
            f'<text x="{left + plot_width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="15">Concurrency</text>',
            f'<text x="22" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 22 {top + plot_height / 2})" font-family="sans-serif" font-size="15">Sidecar / direct</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_campaign(output_root: Path) -> None:
    manifest_path = output_root / "manifest.json"
    plan_path = output_root / "resolved-plan.json"
    if not manifest_path.exists() or not plan_path.exists():
        raise CampaignError(
            "campaign manifest and resolved plan are required for analysis"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    legs = json.loads(plan_path.read_text(encoding="utf-8"))["legs"]
    state = load_state(output_root)
    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    individual_rows: list[dict[str, Any]] = []
    for leg in legs:
        entry = state["legs"].get(leg["id"], {})
        metrics = entry.get("metrics", {})
        process_capture = metrics.get("process_metrics", {})
        process_metrics = process_capture.get("combined_backend", {})
        row = {
            "leg_id": leg["id"],
            "phase": leg["phase"],
            "status": entry.get("status", "not-run"),
            "arm": leg["arm"],
            "input_tokens": leg["input_tokens"],
            "output_tokens": leg["output_tokens"],
            "concurrency": leg["concurrency"],
            "grpc_connections": leg["grpc_connections"],
            "aiperf_shards": leg["aiperf_shards"],
            "output_throughput_tps": metrics.get("output_throughput_tps", 0.0),
            "request_throughput_rps": metrics.get("request_throughput_rps", 0.0),
            "ttft_p50_ms": metrics.get("ttft_p50_ms", 0.0),
            "ttft_p99_ms": metrics.get("ttft_p99_ms", 0.0),
            "itl_p50_ms": metrics.get("itl_p50_ms", 0.0),
            "itl_p99_ms": metrics.get("itl_p99_ms", 0.0),
            "e2e_p50_ms": metrics.get("e2e_p50_ms", 0.0),
            "e2e_p99_ms": metrics.get("e2e_p99_ms", 0.0),
            "completed_requests": metrics.get("completed_requests", 0.0),
            "failed_requests": metrics.get("failed_requests", 0.0),
            "failed_request_fraction": metrics.get("failed_request_fraction", 0.0),
            "total_input_tokens": metrics.get("total_input_tokens", 0.0),
            "total_output_tokens": metrics.get("total_output_tokens", 0.0),
            "aiperf_total_input_tokens": metrics.get(
                "aiperf_total_input_tokens", 0.0
            ),
            "aiperf_total_output_tokens": metrics.get(
                "aiperf_total_output_tokens", 0.0
            ),
            "aiperf_total_output_sequence_tokens": metrics.get(
                "aiperf_total_output_sequence_tokens", 0.0
            ),
            "backend_cpu_seconds": process_metrics.get("cpu_seconds", 0.0),
            "backend_max_rss_kib_sum": process_metrics.get("max_rss_kib_sum", 0),
            "backend_context_switches": process_metrics.get("context_switches", 0),
            "output_dir": entry.get("output_dir", ""),
        }
        for label, values in process_capture.get("processes", {}).items():
            row[f"process_{label}_cpu_seconds"] = values.get("cpu_seconds", 0.0)
            row[f"process_{label}_max_rss_kib"] = values.get("max_rss_kib", 0)
            row[f"process_{label}_voluntary_context_switches"] = values.get(
                "voluntary_ctxt_switches", 0
            )
            row[f"process_{label}_nonvoluntary_context_switches"] = values.get(
                "nonvoluntary_ctxt_switches", 0
            )
        row.update(
            {
                f"meta_{key}": value
                for key, value in leg["metadata"].items()
                if key != "reuse_from"
            }
        )
        individual_rows.append(row)
    individual_fields = sorted({field for row in individual_rows for field in row})
    write_csv(results_dir / "individual.csv", individual_rows, individual_fields)
    atomic_write_json(results_dir / "individual.json", individual_rows)

    flags = manifest["flags"]
    main_rows = [
        row
        for row in individual_rows
        if row["phase"] == "main" and row["status"] in {"complete", "reused"}
    ]
    pair_groups: dict[tuple[int, int, int, int], dict[str, dict[str, Any]]] = {}
    for row in main_rows:
        key = (
            int(row["input_tokens"]),
            int(row["output_tokens"]),
            int(row["concurrency"]),
            int(row["meta_pair_index"]),
        )
        pair_groups.setdefault(key, {})[row["arm"]] = row
    comparison_metrics = {
        "output_throughput_tps": "output_throughput_ratio",
        "request_throughput_rps": "request_throughput_ratio",
        "ttft_p50_ms": "ttft_p50_ratio",
        "ttft_p99_ms": "ttft_p99_ratio",
        "itl_p50_ms": "itl_p50_ratio",
        "itl_p99_ms": "itl_p99_ratio",
        "e2e_p50_ms": "e2e_p50_ratio",
        "e2e_p99_ms": "e2e_p99_ratio",
        "backend_cpu_seconds": "backend_cpu_ratio",
        "backend_max_rss_kib_sum": "backend_max_rss_ratio",
        "backend_context_switches": "backend_context_switch_ratio",
    }
    paired_rows: list[dict[str, Any]] = []
    for key in sorted(pair_groups):
        arms = pair_groups[key]
        if set(arms) != {"direct-mocker", "sidecar"}:
            continue
        direct = arms["direct-mocker"]
        sidecar = arms["sidecar"]
        pair_row = {
            "input_tokens": key[0],
            "output_tokens": key[1],
            "concurrency": key[2],
            "pair_index": key[3],
            "direct_leg_id": direct["leg_id"],
            "sidecar_leg_id": sidecar["leg_id"],
            "direct_failed_request_fraction": direct["failed_request_fraction"],
            "sidecar_failed_request_fraction": sidecar["failed_request_fraction"],
        }
        for metric_name, ratio_name in comparison_metrics.items():
            direct_value = float(direct[metric_name])
            sidecar_value = float(sidecar[metric_name])
            pair_row[f"direct_{metric_name}"] = direct_value
            pair_row[f"sidecar_{metric_name}"] = sidecar_value
            pair_row[ratio_name] = ratio(sidecar_value, direct_value)
        paired_rows.append(pair_row)
    paired_fields = list(paired_rows[0]) if paired_rows else []
    write_csv(results_dir / "paired.csv", paired_rows, paired_fields)
    atomic_write_json(results_dir / "paired.json", paired_rows)

    grouped_pairs: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in paired_rows:
        key = (
            int(row["input_tokens"]),
            int(row["output_tokens"]),
            int(row["concurrency"]),
        )
        grouped_pairs.setdefault(key, []).append(row)
    paired_medians: list[dict[str, Any]] = []
    flagged_points: list[dict[str, Any]] = []
    for key in sorted(grouped_pairs):
        pairs = grouped_pairs[key]
        if len(pairs) != 2:
            continue
        failure_fraction = median(pairs, "sidecar_failed_request_fraction")
        output_ratio = median(pairs, "output_throughput_ratio")
        ttft_ratio = median(pairs, "ttft_p99_ratio")
        point_flags = []
        if math.isfinite(output_ratio) and output_ratio < 1.0 - float(
            flags["throughput_loss_fraction"]
        ):
            point_flags.append("throughput_loss")
        if math.isfinite(ttft_ratio) and ttft_ratio > 1.0 + float(
            flags["p99_ttft_increase_fraction"]
        ):
            point_flags.append("p99_ttft_increase")
        if failure_fraction > float(flags["failed_request_fraction"]):
            point_flags.append("failed_requests")
        row = {
            "input_tokens": key[0],
            "output_tokens": key[1],
            "concurrency": key[2],
            "sidecar_failed_request_fraction_median": failure_fraction,
            "flags": ";".join(point_flags),
        }
        for metric_name, ratio_name in comparison_metrics.items():
            row[f"direct_{metric_name}_median"] = median(pairs, f"direct_{metric_name}")
            row[f"sidecar_{metric_name}_median"] = median(
                pairs, f"sidecar_{metric_name}"
            )
            row[ratio_name] = median(pairs, ratio_name)
        paired_medians.append(row)
        if point_flags:
            flagged_points.append(row)
    median_fields = list(paired_medians[0]) if paired_medians else []
    write_csv(results_dir / "paired-medians.csv", paired_medians, median_fields)
    atomic_write_json(results_dir / "paired-medians.json", paired_medians)
    atomic_write_json(results_dir / "flagged-points.json", flagged_points)

    connection_rows = [
        row
        for row in individual_rows
        if row["phase"] == "connections" and row["status"] in {"complete", "reused"}
    ]
    connection_fields = list(connection_rows[0]) if connection_rows else []
    write_csv(
        results_dir / "connection-diagnostic.csv", connection_rows, connection_fields
    )
    diagnostic_summary: list[dict[str, Any]] = []
    for anchor in manifest["connection_diagnostic"]["anchors"]:
        anchor_rows = [
            row
            for row in connection_rows
            if int(row["input_tokens"]) == int(anchor["input_tokens"])
            and int(row["output_tokens"]) == int(anchor["output_tokens"])
            and int(row["concurrency"]) == int(anchor["concurrency"])
        ]
        rows_8 = [row for row in anchor_rows if int(row["grpc_connections"]) == 8]
        rows_16 = [row for row in anchor_rows if int(row["grpc_connections"]) == 16]
        if not rows_8 or not rows_16:
            continue
        throughput_8 = median(rows_8, "output_throughput_tps")
        throughput_16 = median(rows_16, "output_throughput_tps")
        failures_8 = median(rows_8, "failed_request_fraction")
        failures_16 = median(rows_16, "failed_request_fraction")
        throughput_improvement = ratio(throughput_16, throughput_8) - 1.0
        failure_reduction = failures_8 - failures_16
        insufficient = (
            throughput_improvement
            > float(flags["pool_throughput_improvement_fraction"])
            or failure_reduction
            > float(flags["pool_failure_reduction_percentage_points"]) / 100.0
        )
        diagnostic_summary.append(
            {
                **anchor,
                "throughput_improvement_fraction_16_vs_8": throughput_improvement,
                "failure_reduction_fraction_16_vs_8": failure_reduction,
                "eight_connections_insufficient": insufficient,
            }
        )
    atomic_write_json(results_dir / "connection-diagnostic.json", diagnostic_summary)
    if paired_medians:
        write_ratio_chart(
            results_dir / "throughput-ratio.svg",
            paired_medians,
            "output_throughput_ratio",
            "vLLM sidecar/direct output-throughput ratio",
            1.0 - float(flags["throughput_loss_fraction"]),
        )
        write_ratio_chart(
            results_dir / "ttft-p99-ratio.svg",
            paired_medians,
            "ttft_p99_ratio",
            "vLLM sidecar/direct p99 TTFT ratio",
            1.0 + float(flags["p99_ttft_increase_fraction"]),
        )

    complete_main = len(main_rows)
    expected_main = len([leg for leg in legs if leg["phase"] == "main"])
    pool_insufficient = any(
        row["eight_connections_insufficient"] for row in diagnostic_summary
    )
    report_lines = [
        "# CPU-Only vLLM Sidecar A/B Campaign",
        "",
        "## Completion",
        "",
        f"- Main legs: {complete_main}/{expected_main}",
        f"- Matched pairs: {len(paired_rows)}/56",
        f"- Paired-median points: {len(paired_medians)}/28",
        f"- Flagged points: {len(flagged_points)}",
        f"- Eight-connection pool insufficient: {'yes' if pool_insufficient else 'no'}",
        "",
        "## Flagged points",
        "",
    ]
    if flagged_points:
        report_lines.extend(
            f"- {row['input_tokens']}×{row['output_tokens']}/c{row['concurrency']}: {row['flags']}"
            for row in flagged_points
        )
    else:
        report_lines.append("- None.")
    report_lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "These results isolate Dynamo request-plane plus native vLLM gRPC sidecar overhead against the same CPU Mocker scheduler. They do not measure vLLM EngineCore, model execution, GPU kernels, KV transfer, real-engine startup, or OpenEngine.",
            "",
            "Individual runs are in `individual.csv`; matched comparisons are in `paired.csv`; point-level paired medians are in `paired-medians.csv`; connection-pool diagnostics are in `connection-diagnostic.csv` and `connection-diagnostic.json`.",
        ]
    )
    (results_dir / "report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )


def plan_campaign(args: argparse.Namespace) -> None:
    manifest = load_manifest()
    legs = build_resolved_plan(manifest, args.high_concurrency_shards)
    value = {"schema_version": 1, "legs": legs}
    if args.output:
        atomic_write_json(args.output.expanduser().resolve(), value)
    else:
        json.dump(value, sys.stdout, indent=2)
        sys.stdout.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser(
        "plan", help="render the deterministic resolved plan without running it"
    )
    plan.add_argument("--high-concurrency-shards", type=int, choices=[1, 4], default=1)
    plan.add_argument("--output", type=Path)
    plan.set_defaults(func=plan_campaign)

    run = subparsers.add_parser("run", help="execute a locked campaign phase")
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument(
        "--phase",
        choices=["smoke", "preflight", "main", "connections", "profiles", "all"],
        default="all",
    )
    run.add_argument("--vllm-sidecar-bin")
    run.add_argument("--vllm-mocker-server-bin")
    run.add_argument("--retry-failed", action="store_true")
    run.set_defaults(func=run_campaign)

    analyze = subparsers.add_parser(
        "analyze", help="aggregate completed campaign artifacts"
    )
    analyze.add_argument("--output-root", type=Path, required=True)
    analyze.set_defaults(
        func=lambda args: analyze_campaign(args.output_root.expanduser().resolve())
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except CampaignError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
