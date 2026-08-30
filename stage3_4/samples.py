"""Stage 3/4 sample access -- stdlib only, importable unmodified by every model conda env.

CRITICAL for fair cross-stage comparison: Stages 3 & 4 use the EXACT SAME 867-sample split as
Stages 1 & 2. A sample is valid iff it has all five artifacts present and both transcripts
non-empty -- identical predicate to Stage 2's loader -- so Stage1/2/3/4 all evaluate the same ids:
    context.wav / context.txt / target.wav / target.txt / sarcasm_analysis.json

Stage 3 uses only target.txt. Stage 4 uses context.txt + target.txt. No audio is ever read here
(the audio files must merely EXIST, to keep the split identical to the audio stages).
"""
import json
import os

SAMPLES_ROOT = os.environ.get("SARCASM_DATA_ROOT", "data/samples")
_NEED = ["context.wav", "context.txt", "target.wav", "target.txt", "sarcasm_analysis.json"]


def _read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def iter_valid_samples(root=SAMPLES_ROOT):
    """Same predicate as Stage 2 -> the same 867 ids, so the dataset split is unchanged."""
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        if not all(os.path.isfile(os.path.join(d, f)) for f in _NEED):
            continue
        if not _read_text(os.path.join(d, "context.txt")):
            continue
        if not _read_text(os.path.join(d, "target.txt")):
            continue
        yield name


def load_sample(sample_id, root=SAMPLES_ROOT):
    d = os.path.join(root, sample_id)
    gt = None
    gp = os.path.join(d, "sarcasm_analysis.json")
    if os.path.isfile(gp):
        with open(gp, "r", encoding="utf-8", errors="replace") as f:
            gt = json.load(f)
    return {
        "sample_id": sample_id,
        "context_text": _read_text(os.path.join(d, "context.txt")),
        "target_text": _read_text(os.path.join(d, "target.txt")),
        "ground_truth": gt,
    }


if __name__ == "__main__":
    ids = list(iter_valid_samples())
    print(f"valid samples (== Stage 1/2 split): {len(ids)}")
    s = load_sample(ids[0])
    print("example:", s["sample_id"], "| ctx:", s["context_text"][:50], "| tgt:", s["target_text"][:50],
          "| gt:", s["ground_truth"]["category"])
