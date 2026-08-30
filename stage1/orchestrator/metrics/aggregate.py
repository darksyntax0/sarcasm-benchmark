"""Loads one model's results + ground truth + run_meta, computes all metrics, writes the
per-model summary JSON, and builds the cross-model comparison table."""
import json
import os
import sys
from datetime import datetime, timezone

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd

from bench_common import io_utils
from orchestrator.metrics.classification import compute_classification_metrics
from orchestrator.metrics.explanation_similarity import compute_explanation_similarity
from orchestrator.metrics.performance import compute_performance_metrics


def aggregate_model(model_name: str, results_path: str, ground_truth_path: str,
                     run_meta_path: str = None, dataset_size: int = None) -> dict:
    results = io_utils.read_jsonl(results_path)
    gt_records = io_utils.read_jsonl(ground_truth_path)
    gt_category_lookup = {g["sample_id"]: g["gt_category"] for g in gt_records}
    gt_explanation_lookup = {g["sample_id"]: g["gt_explanation"] for g in gt_records}

    run_meta = None
    if run_meta_path and os.path.exists(run_meta_path):
        with open(run_meta_path) as f:
            run_meta = json.load(f)

    classification = compute_classification_metrics(results, gt_category_lookup)
    similarity = compute_explanation_similarity(results, gt_explanation_lookup)
    performance = compute_performance_metrics(results, run_meta)

    summary = {
        "model_name": model_name,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_size": dataset_size if dataset_size is not None else len(results),
        "accuracy": classification["accuracy"],
        "macro_f1": classification["macro_f1"],
        "weighted_f1": classification["weighted_f1"],
        "per_category_metrics": classification["per_class"],
        "confusion_matrix": classification["confusion_matrix"],
        "labels": classification["labels"],
        "explanation_similarity": {
            k: v for k, v in similarity.items() if k != "per_sample"
        },
        "inference_time": performance["latency"],
        "throughput_samples_per_min": performance["throughput_samples_per_min"],
        "wall_clock_seconds": performance["wall_clock_seconds"],
        "memory_usage": {
            "gpu_mem_allocated_mb": performance["gpu_mem_allocated_mb"],
            "gpu_mem_reserved_mb": performance["gpu_mem_reserved_mb"],
        },
        "total_attempted": classification["total_attempted"],
        "total_valid_predictions": classification["total_valid_predictions"],
        "timeout_count": classification["timeout_count"],
        "error_count": classification["error_count"],
        "parse_error_count": classification["parse_error_count"],
        "unparseable_count": classification["unparseable_count"],
        "unparseable_rate": classification["unparseable_rate"],
    }
    return summary


def write_model_metrics(summary: dict, output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)


def build_comparison_table(model_summaries: list) -> "pd.DataFrame":
    rows = []
    for s in model_summaries:
        rows.append(
            {
                "model_name": s["model_name"],
                "dataset_size": s["dataset_size"],
                "accuracy": s["accuracy"],
                "macro_f1": s["macro_f1"],
                "weighted_f1": s["weighted_f1"],
                "mean_explanation_similarity": s["explanation_similarity"].get("mean"),
                "mean_latency_sec": (s["inference_time"] or {}).get("mean_sec"),
                "throughput_samples_per_min": s["throughput_samples_per_min"],
                "timeout_rate": s["timeout_count"] / s["total_attempted"] if s["total_attempted"] else None,
                "unparseable_rate": s["unparseable_rate"],
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--run-meta", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = aggregate_model(args.model_name, args.results, args.ground_truth, args.run_meta)
    write_model_metrics(summary, args.output)
    print(json.dumps(summary, indent=2))
