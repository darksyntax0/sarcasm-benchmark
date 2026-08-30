"""Renders per-model and cross-model markdown reports with matplotlib/seaborn figures.

Two caveats are stated explicitly wherever the relevant numbers are shown (not just once in a
footnote): confidence is a self-reported generative-model number, not a calibrated probability;
explanation-similarity is an embedding-cosine proxy for semantic overlap, not a correctness
judgment.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

CONFIDENCE_CAVEAT = (
    "*Confidence is a self-reported number from the model's own generated JSON output, not a "
    "calibrated probability -- these models are not classifiers with a trained confidence head.*"
)
SIMILARITY_CAVEAT = (
    "*Explanation similarity is an embedding cosine-similarity proxy for semantic overlap with "
    "the ground-truth explanation, not a correctness judgment -- wording can differ while the "
    "underlying reasoning is right, or vice versa.*"
)


def _confusion_matrix_figure(cm, labels, model_name, out_path):
    if cm is None:
        return None
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"{model_name} — confusion matrix")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _per_class_f1_figure(per_class, model_name, out_path):
    if not per_class:
        return None
    labels = list(per_class.keys())
    f1s = [per_class[l]["f1"] for l in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, f1s, color="#4C72B0")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title(f"{model_name} — per-category F1")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _latency_histogram(per_sample_latencies, model_name, out_path):
    if not per_sample_latencies:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(per_sample_latencies, bins=15, color="#55A868")
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Count")
    ax.set_title(f"{model_name} — inference latency distribution")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def generate_model_report(model_name, metrics: dict, results: list, figures_dir: str, out_dir: str) -> str:
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    cm_path = os.path.join(figures_dir, f"{model_name}_confusion_matrix.png")
    _confusion_matrix_figure(metrics.get("confusion_matrix"), metrics.get("labels"), model_name, cm_path)

    f1_path = os.path.join(figures_dir, f"{model_name}_per_class_f1.png")
    _per_class_f1_figure(metrics.get("per_category_metrics"), model_name, f1_path)

    latencies = [r["latency_seconds"] for r in results if r.get("latency_seconds") is not None and r.get("error") != "timeout"]
    lat_path = os.path.join(figures_dir, f"{model_name}_latency_hist.png")
    _latency_histogram(latencies, model_name, lat_path)

    lines = [f"# {model_name} — Stage 1 Report\n"]
    lines.append(f"- Dataset size: {metrics['dataset_size']}")
    lines.append(f"- Accuracy: {metrics['accuracy']}")
    lines.append(f"- Macro F1: {metrics['macro_f1']}")
    lines.append(f"- Weighted F1: {metrics['weighted_f1']}")
    lines.append(f"- Timeout count: {metrics['timeout_count']} / {metrics['total_attempted']}")
    lines.append(f"- Parse error count: {metrics['parse_error_count']}")
    lines.append(f"- Unparseable rate: {metrics['unparseable_rate']}\n")

    lines.append("## Per-category metrics\n")
    lines.append("| Category | Precision | Recall | F1 | Support |")
    lines.append("|---|---|---|---|---|")
    for cat, m in (metrics.get("per_category_metrics") or {}).items():
        lines.append(f"| {cat} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['support']} |")

    lines.append(f"\n![confusion matrix]({os.path.relpath(cm_path, out_dir)})")
    lines.append(f"\n![per-class F1]({os.path.relpath(f1_path, out_dir)})")
    if lat_path:
        lines.append(f"\n![latency histogram]({os.path.relpath(lat_path, out_dir)})")

    lines.append("\n## Explanation similarity\n")
    sim = metrics.get("explanation_similarity") or {}
    lines.append(f"- Mean: {sim.get('mean')}  (std: {sim.get('std')}, compared: {sim.get('compared_count')}, skipped: {sim.get('skipped_count')})")
    lines.append(f"\n{SIMILARITY_CAVEAT}\n")
    lines.append(f"\n{CONFIDENCE_CAVEAT}\n")

    report_path = os.path.join(out_dir, f"{model_name}_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path


def generate_comparison_report(comparison_df, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    lines = ["# Stage 1 — Model Comparison\n"]
    lines.append(comparison_df.to_markdown(index=False))
    lines.append(f"\n{SIMILARITY_CAVEAT}\n")
    lines.append(f"\n{CONFIDENCE_CAVEAT}\n")
    report_path = os.path.join(out_dir, "comparison_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path
