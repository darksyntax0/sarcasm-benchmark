#!/usr/bin/env python3
"""Unified scorer for all five stages.

Scoring convention (as in the paper): predictions that are absent, unparseable, or
outside the label inventory are scored as INCORRECT rather than excluded, so every
condition is evaluated on an identical item set and format adherence is not conflated
with task accuracy.
"""
import argparse, json, os, sys
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalise import normalize, CATS, INVALID


def load_gt(path):
    gt = {}
    for line in open(path):
        r = json.loads(line)
        gt[r["sample_id"]] = r.get("gt_category") or r.get("category")
    return gt


def score_run(results_path, gt):
    pred = {}
    for line in open(results_path):
        r = json.loads(line)
        pred[r["sample_id"]] = normalize(r.get("predicted_category"))
    ids = sorted(gt)
    y_true = [gt[i] for i in ids]
    y_pred = [pred.get(i, INVALID) for i in ids]          # missing item == incorrect
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    _, _, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(CATS), average=None, zero_division=0)
    return {
        "n_items": len(ids),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(np.mean(f1)), 4),
        "per_category_f1": {c: round(float(x), 4) for c, x in zip(CATS, f1)},
        "support": {c: int(s) for c, s in zip(CATS, sup)},
        "invalid_predictions": sum(1 for v in y_pred if v == INVALID),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True,
                    help="directory containing <model>/results.jsonl")
    ap.add_argument("--ground-truth", required=True, help="ground truth jsonl")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    gt = load_gt(args.ground_truth)
    out = {}
    for model in sorted(os.listdir(args.results_root)):
        p = os.path.join(args.results_root, model, "results.jsonl")
        if os.path.isfile(p):
            out[model] = score_run(p, gt)

    hdr = f"{'model':20s}{'acc':>8s}{'macroF1':>9s}{'invalid':>9s}"
    print(hdr); print("-" * len(hdr))
    for m, s in out.items():
        print(f"{m:20s}{s['accuracy']:8.3f}{s['macro_f1']:9.3f}{s['invalid_predictions']:9d}")
    if out:
        print("-" * len(hdr))
        print(f"{'MEAN':20s}{np.mean([s['accuracy'] for s in out.values()]):8.3f}"
              f"{np.mean([s['macro_f1'] for s in out.values()]):9.3f}")
    if args.json_out:
        json.dump(out, open(args.json_out, "w"), indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
