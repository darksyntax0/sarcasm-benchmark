"""Classification metrics: accuracy, per-class precision/recall/F1, confusion matrix.

Runs in the benchmark-orchestrator env (needs scikit-learn).
"""
import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from bench_common.constants import VALID_CATEGORIES

LABELS = list(VALID_CATEGORIES) + ["OTHER"]

# Observed near-miss spellings that are unambiguously the same category (not a fuzzy matcher --
# only add entries actually seen in real model output, e.g. Audio-Reasoner consistently says
# "SARCASTIC" (adjective) instead of "SARCASM" (the requested noun form) in ~8/18 pilot samples).
_CATEGORY_ALIASES = {
    "SARCASTIC": "SARCASM",
}


def _normalize_category(value):
    """Generative models won't perfectly conform to the requested exact casing/spelling --
    normalize defensively before matching against LABELS."""
    if not value or not isinstance(value, str):
        return None
    v = value.strip().upper().replace(" ", "_")
    v = _CATEGORY_ALIASES.get(v, v)
    if v in LABELS:
        return v
    return None


def compute_classification_metrics(results: list, gt_lookup: dict) -> dict:
    """results: list of result dicts (one model's results.jsonl, already loaded).
    gt_lookup: {sample_id: gt_category}.

    Rows with an unparseable/None predicted_category are excluded from the accuracy/F1
    denominator but counted separately as unparseable_count/unparseable_rate -- a model that
    frequently fails to produce a usable category is a real finding, not noise to discard.
    """
    y_true = []
    y_pred = []
    timeout_count = 0
    error_count = 0
    parse_error_count = 0
    unparseable_count = 0
    total_attempted = 0

    for r in results:
        sample_id = r.get("sample_id")
        if sample_id not in gt_lookup:
            continue
        total_attempted += 1

        if r.get("error") == "timeout":
            timeout_count += 1
            unparseable_count += 1
            continue
        if r.get("error"):
            error_count += 1
            unparseable_count += 1
            continue
        if r.get("parse_error"):
            parse_error_count += 1
            unparseable_count += 1
            continue

        pred = _normalize_category(r.get("predicted_category"))
        if pred is None:
            unparseable_count += 1
            continue

        y_true.append(gt_lookup[sample_id])
        y_pred.append(pred)

    total_valid_predictions = len(y_true)

    if total_valid_predictions == 0:
        return {
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "per_class": {},
            "confusion_matrix": None,
            "labels": LABELS,
            "total_attempted": total_attempted,
            "total_valid_predictions": 0,
            "timeout_count": timeout_count,
            "error_count": error_count,
            "parse_error_count": parse_error_count,
            "unparseable_count": unparseable_count,
            "unparseable_rate": (unparseable_count / total_attempted) if total_attempted else None,
        }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )
    macro_f1 = sum(f1) / len(f1)
    total_support = sum(support)
    weighted_f1 = (
        sum(f * s for f, s in zip(f1, support)) / total_support if total_support else 0.0
    )
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / total_valid_predictions
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(LABELS)
    }

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": LABELS,
        "total_attempted": total_attempted,
        "total_valid_predictions": total_valid_predictions,
        "timeout_count": timeout_count,
        "error_count": error_count,
        "parse_error_count": parse_error_count,
        "unparseable_count": unparseable_count,
        "unparseable_rate": unparseable_count / total_attempted if total_attempted else None,
    }
