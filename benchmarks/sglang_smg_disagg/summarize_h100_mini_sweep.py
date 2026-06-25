#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path


METRIC_MAP = {
    "Request Throughput (requests/sec)": "request_rps",
    "Output Token Throughput (tokens/sec)": "output_tps",
    "Time To First Token (ms)": "ttft",
    "Request Latency (ms)": "latency",
    "Inter Token Latency (ms)": "itl",
    "Output Sequence Length (tokens)": "osl",
    "Input Sequence Length (tokens)": "isl",
}


def read_csv(path: Path) -> dict:
    out = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row.get("Metric", "")
            key = METRIC_MAP.get(metric)
            if not key:
                continue
            if key in {"request_rps", "output_tps"}:
                out[key] = float(row["avg"])
            else:
                for stat in ("avg", "p50", "p99"):
                    value = row.get(stat)
                    if value not in (None, ""):
                        out[f"{key}_{stat}"] = float(value)
    return out


def load_metadata(point_dir: Path) -> dict:
    with (point_dir / "metadata.json").open() as f:
        return json.load(f)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} SWEEP_DIR", file=sys.stderr)
        return 2

    sweep_dir = Path(sys.argv[1])
    rows = []
    for point_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        metadata_path = point_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        meta = load_metadata(point_dir)
        for backend in ("smg", "legacy", "unified"):
            csv_path = point_dir / "artifacts" / backend / "profile_export_aiperf.csv"
            if not csv_path.exists():
                rows.append(
                    {
                        "point": meta["point"],
                        "workload": meta["point"].split("_isl", 1)[0],
                        "backend": backend,
                        "status": "missing",
                    }
                )
                continue
            row = {
                "point": meta["point"],
                "workload": meta["point"].split("_isl", 1)[0],
                "isl": meta["isl"],
                "osl": meta["osl"],
                "concurrency": meta["concurrency"],
                "requests": meta["requests"],
                "backend": backend,
                "status": "ok",
            }
            row.update(read_csv(csv_path))
            rows.append(row)

    out_json = sweep_dir / "summary.json"
    out_csv = sweep_dir / "summary.csv"
    out_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    fieldnames = [
        "point",
        "workload",
        "isl",
        "osl",
        "concurrency",
        "requests",
        "backend",
        "status",
        "request_rps",
        "output_tps",
        "ttft_p50",
        "ttft_p99",
        "latency_p50",
        "latency_p99",
        "itl_p50",
        "itl_p99",
        "isl_avg",
        "osl_avg",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"wrote {out_csv}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
