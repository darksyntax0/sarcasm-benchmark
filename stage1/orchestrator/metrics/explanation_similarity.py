"""Explanation-similarity metric: embedding cosine similarity between each model's predicted
explanation and the ground-truth explanation. This is a semantic-overlap PROXY, not a
correctness judgment -- a model can be highly similar in wording while reasoning about the wrong
thing, or dissimilar in wording while capturing the same idea. Report this caveat wherever the
number is surfaced (see report/generate_report.py).
"""
import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

_model_singleton = {"model": None}


def _get_model(model_name: str = "all-MiniLM-L6-v2"):
    if _model_singleton["model"] is None:
        from sentence_transformers import SentenceTransformer

        _model_singleton["model"] = SentenceTransformer(model_name)
    return _model_singleton["model"]


def compute_explanation_similarity(results: list, gt_lookup_explanations: dict,
                                    model_name: str = "all-MiniLM-L6-v2") -> dict:
    """gt_lookup_explanations: {sample_id: gt_explanation}."""
    pairs = []
    for r in results:
        sample_id = r.get("sample_id")
        pred_explanation = r.get("explanation")
        gt_explanation = gt_lookup_explanations.get(sample_id)
        if sample_id in gt_lookup_explanations and pred_explanation and gt_explanation:
            pairs.append((sample_id, pred_explanation, gt_explanation))

    skipped_count = len(results) - len(pairs)

    if not pairs:
        return {
            "mean": None, "median": None, "std": None, "min": None, "max": None,
            "per_sample": [], "skipped_count": skipped_count, "compared_count": 0,
        }

    st_model = _get_model(model_name)
    pred_texts = [p[1] for p in pairs]
    gt_texts = [p[2] for p in pairs]

    pred_emb = st_model.encode(pred_texts, convert_to_numpy=True, normalize_embeddings=True)
    gt_emb = st_model.encode(gt_texts, convert_to_numpy=True, normalize_embeddings=True)
    sims = np.sum(pred_emb * gt_emb, axis=1)  # cosine similarity, both sides pre-normalized

    per_sample = [
        {"sample_id": pairs[i][0], "similarity": float(sims[i])} for i in range(len(pairs))
    ]

    return {
        "mean": float(np.mean(sims)),
        "median": float(np.median(sims)),
        "std": float(np.std(sims)),
        "min": float(np.min(sims)),
        "max": float(np.max(sims)),
        "percentile_25": float(np.percentile(sims, 25)),
        "percentile_75": float(np.percentile(sims, 75)),
        "per_sample": per_sample,
        "skipped_count": skipped_count,
        "compared_count": len(pairs),
    }
