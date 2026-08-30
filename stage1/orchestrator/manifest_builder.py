"""Builds manifest.jsonl (fed to model subprocesses, no ground truth) and ground_truth.jsonl
(orchestrator-only) from the real dataset. Runs in the benchmark-orchestrator env."""
import logging
import os
import random
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench_common.dataset import load_stage1_samples
from bench_common import io_utils

logger = logging.getLogger(__name__)


def build_full_manifest(samples_root: str):
    """Calls bench_common.dataset.load_stage1_samples, logs the validity report, returns
    (manifest_dicts, ground_truth_dicts, validity_report)."""
    manifest_records, gt_records, report = load_stage1_samples(samples_root)
    logger.info("Stage 1 dataset validity report: %s", report)
    manifest_dicts = [m.to_dict() for m in manifest_records]
    gt_dicts = [
        {
            "sample_id": g.sample_id,
            "gt_category": g.gt_category,
            "gt_sarcasm_source": g.gt_sarcasm_source,
            "gt_sarcasm_target": g.gt_sarcasm_target,
            "gt_explanation": g.gt_explanation,
        }
        for g in gt_records
    ]
    return manifest_dicts, gt_dicts, report


def build_pilot_manifest(full_manifest: list, n: int = 18, seed: int = 42) -> list:
    """Deterministic/reproducible pilot subset -- fixed seed so re-running the pilot picks the
    SAME samples for apples-to-apples comparison across models and across framework revisions."""
    rng = random.Random(seed)
    return rng.sample(full_manifest, min(n, len(full_manifest)))


def write_manifest_and_gt(manifest_dicts: list, gt_dicts: list, manifest_path: str, gt_path: str):
    if os.path.exists(manifest_path):
        os.remove(manifest_path)
    if os.path.exists(gt_path):
        os.remove(gt_path)
    for m in manifest_dicts:
        io_utils.append_jsonl(manifest_path, m)
    for g in gt_dicts:
        io_utils.append_jsonl(gt_path, g)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    full_manifest, gt_dicts, report = build_full_manifest(
        os.environ.get("SARCASM_DATA_ROOT", "data/samples")
    )
    print("full manifest size:", len(full_manifest))
    pilot = build_pilot_manifest(full_manifest)
    print("pilot manifest size:", len(pilot))
    print("pilot sample_ids:", [r["sample_id"] for r in pilot])
