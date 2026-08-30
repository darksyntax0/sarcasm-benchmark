"""Stage 1 dataset loader: walks benchmark/samples/ and determines which sample directories are
actually usable (target.wav + non-empty target.txt + valid sarcasm_analysis.json present).

This is the single source of truth for "how many valid Stage 1 samples exist" -- nothing else
in this framework hardcodes that number.
"""
import json
import os

from .constants import REQUIRED_GT_KEYS, VALID_CATEGORIES
from .schema import GroundTruthRecord, ManifestRecord


def load_stage1_samples(samples_root: str):
    """Walk samples_root. For each sample directory, require:
      - target.wav present
      - target.txt present AND non-empty after strip() (catches both missing-file and the
        real-world empty-file case)
      - sarcasm_analysis.json present, valid JSON, has all REQUIRED_GT_KEYS, category value in
        VALID_CATEGORIES

    context.wav / context.txt are never opened -- Stage 2 concern only.

    Returns (manifest_records, ground_truth_records, validity_report) where validity_report is
    a dict of counts explaining exactly why each excluded sample was excluded.
    """
    report = {
        "total_dirs": 0,
        "missing_wav": 0,
        "missing_or_empty_txt": 0,
        "missing_json": 0,
        "invalid_json_parse": 0,
        "invalid_json_schema": 0,
        "invalid_category_value": 0,
        "sample_id_mismatch": 0,
        "valid_count": 0,
    }

    manifest_records = []
    ground_truth_records = []

    if not os.path.isdir(samples_root):
        raise FileNotFoundError(f"samples_root does not exist: {samples_root}")

    for entry in sorted(os.listdir(samples_root)):
        sample_dir = os.path.join(samples_root, entry)
        if not os.path.isdir(sample_dir):
            continue
        report["total_dirs"] += 1

        wav_path = os.path.join(sample_dir, "target.wav")
        txt_path = os.path.join(sample_dir, "target.txt")
        json_path = os.path.join(sample_dir, "sarcasm_analysis.json")

        if not os.path.isfile(wav_path):
            report["missing_wav"] += 1
            continue

        if not os.path.isfile(txt_path):
            report["missing_or_empty_txt"] += 1
            continue
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            target_text = f.read().strip()
        if not target_text:
            report["missing_or_empty_txt"] += 1
            continue

        if not os.path.isfile(json_path):
            report["missing_json"] += 1
            continue
        with open(json_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
        try:
            gt = json.loads(raw)
        except json.JSONDecodeError:
            report["invalid_json_parse"] += 1
            continue
        if not isinstance(gt, dict) or not all(k in gt for k in REQUIRED_GT_KEYS):
            report["invalid_json_schema"] += 1
            continue
        if gt["category"] not in VALID_CATEGORIES:
            report["invalid_category_value"] += 1
            continue
        if gt.get("sample_id") != entry:
            report["sample_id_mismatch"] += 1
            # Warn, don't exclude -- proceed using the directory name as the authoritative id.

        manifest_records.append(
            ManifestRecord(sample_id=entry, target_wav_path=wav_path, target_text=target_text)
        )
        ground_truth_records.append(
            GroundTruthRecord(
                sample_id=entry,
                gt_category=gt["category"],
                gt_sarcasm_source=gt.get("sarcasm_source", ""),
                gt_sarcasm_target=gt.get("sarcasm_target", ""),
                gt_explanation=gt.get("explanation", ""),
            )
        )
        report["valid_count"] += 1

    return manifest_records, ground_truth_records, report


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SARCASM_DATA_ROOT", "data/samples")
    manifest, gts, rep = load_stage1_samples(root)
    print(json.dumps(rep, indent=2))
    print(f"\nFirst 3 manifest records:")
    for m in manifest[:3]:
        print(" ", m.to_dict())
    print(f"\nFirst 3 ground truth records:")
    for g in gts[:3]:
        print(" ", g.to_dict())
