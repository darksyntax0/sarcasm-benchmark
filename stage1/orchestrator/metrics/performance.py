"""Latency, throughput, and GPU-memory aggregation for one model's run."""
import numpy as np
from datetime import datetime


def _parse_iso(ts: str):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")


def compute_performance_metrics(results: list, run_meta: dict = None) -> dict:
    latencies = [r["latency_seconds"] for r in results if r.get("latency_seconds") is not None]
    non_timeout_latencies = [
        r["latency_seconds"] for r in results
        if r.get("latency_seconds") is not None and r.get("error") != "timeout"
    ]
    timeout_count = sum(1 for r in results if r.get("error") == "timeout")
    total = len(results)

    mem_allocated = [r["gpu_mem_allocated_mb"] for r in results if r.get("gpu_mem_allocated_mb")]
    mem_reserved = [r["gpu_mem_reserved_mb"] for r in results if r.get("gpu_mem_reserved_mb")]

    latency_stats = None
    if non_timeout_latencies:
        arr = np.array(non_timeout_latencies)
        latency_stats = {
            "mean_sec": float(np.mean(arr)),
            "median_sec": float(np.median(arr)),
            "p95_sec": float(np.percentile(arr, 95)),
            "min_sec": float(np.min(arr)),
            "max_sec": float(np.max(arr)),
        }

    wall_clock_seconds = None
    if run_meta and run_meta.get("started_at") and run_meta.get("ended_at"):
        wall_clock_seconds = (
            _parse_iso(run_meta["ended_at"]) - _parse_iso(run_meta["started_at"])
        ).total_seconds()

    throughput = None
    if wall_clock_seconds and wall_clock_seconds > 0:
        throughput = total / (wall_clock_seconds / 60.0)  # samples per minute

    return {
        "latency": latency_stats,
        "timeout_count": timeout_count,
        "timeout_rate": timeout_count / total if total else None,
        "wall_clock_seconds": wall_clock_seconds,
        "throughput_samples_per_min": throughput,
        "gpu_mem_allocated_mb": {
            "mean": float(np.mean(mem_allocated)) if mem_allocated else None,
            "max": float(np.max(mem_allocated)) if mem_allocated else None,
        },
        "gpu_mem_reserved_mb": {
            "mean": float(np.mean(mem_reserved)) if mem_reserved else None,
            "max": float(np.max(mem_reserved)) if mem_reserved else None,
        },
    }
