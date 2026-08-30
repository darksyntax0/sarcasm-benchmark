"""Stage 2 sample access -- stdlib only, importable unmodified by every model conda env.

A Stage 2 sample carries BOTH clips + BOTH transcripts + ground truth:
    context.wav / context.txt   (preceding utterance)
    target.wav  / target.txt    (utterance to analyze)
    sarcasm_analysis.json       (ground truth: category/sarcasm_source/sarcasm_target/explanation)

This module NEVER imports torch/transformers so it can be imported in any of the 5 model envs.
It is Stage-2-only and does not touch Stage 1 code or outputs.
"""
import json
import os

SAMPLES_ROOT = os.environ.get("SARCASM_DATA_ROOT", "data/samples")

# Fixed, deterministic validation subset (small, real dataset samples). Chosen from the head of
# the valid-sample list so the set is reproducible and stable across runs/models.
VALIDATION_IDS = ["-C1L4vuYaeM_000", "-C1L4vuYaeM_001"]


def _read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def load_sample(sample_id, root=SAMPLES_ROOT):
    d = os.path.join(root, sample_id)
    ctx_txt = _read_text(os.path.join(d, "context.txt"))
    tgt_txt = _read_text(os.path.join(d, "target.txt"))
    gt = None
    gt_path = os.path.join(d, "sarcasm_analysis.json")
    if os.path.isfile(gt_path):
        with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
            gt = json.load(f)
    return {
        "sample_id": sample_id,
        "context_audio": os.path.join(d, "context.wav"),
        "target_audio": os.path.join(d, "target.wav"),
        "context_text": ctx_txt,
        "target_text": tgt_txt,
        "ground_truth": gt,
    }


def iter_valid_samples(root=SAMPLES_ROOT):
    """Yield every sample_id that has all five Stage-2 artifacts present and both transcripts
    non-empty. Single source of truth for the Stage-2 sample count."""
    need = ["context.wav", "context.txt", "target.wav", "target.txt", "sarcasm_analysis.json"]
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        if not all(os.path.isfile(os.path.join(d, f)) for f in need):
            continue
        if not _read_text(os.path.join(d, "context.txt")):
            continue
        if not _read_text(os.path.join(d, "target.txt")):
            continue
        yield name


def load_validation_samples():
    return [load_sample(sid) for sid in VALIDATION_IDS]


if __name__ == "__main__":
    ids = list(iter_valid_samples())
    print(f"valid Stage-2 samples: {len(ids)}")
    print("validation subset:", VALIDATION_IDS)
    for s in load_validation_samples():
        print(f"  {s['sample_id']}: ctx='{s['context_text'][:40]}...' tgt='{s['target_text'][:40]}...' gt={s['ground_truth']['category']}")
