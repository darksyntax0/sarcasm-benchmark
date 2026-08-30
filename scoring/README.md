# Scoring

`normalise.py` implements the three-step label normalisation described in the paper
(exact match -> morphological variant -> unambiguous single-edit typo; multi-label
hedges and out-of-inventory concepts are *not* mapped and score as incorrect).

`score_all.py` computes accuracy, macro-F1 and per-category F1 for every stage and
model, reproducing the paper's main and per-category tables.

Scoring reads only the stored `results.jsonl` files, so it requires no GPU and can be
re-run freely as normalisation rules evolve.

```bash
python scoring/score_all.py \
    --results-root <stage results dir> \
    --ground-truth <ground_truth_full.jsonl>
```
