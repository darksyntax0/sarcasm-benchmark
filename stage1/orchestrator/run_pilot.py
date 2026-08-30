"""Phase A: pilot run. Validates that all 5 models run end-to-end, JSON parsing/extraction
works, and per-sample timing/memory measurement works, on a small deterministic subset -- BEFORE
committing GPU-hours to the full 867-sample run. Does not auto-chain into Phase B; run_full.py
must be invoked explicitly after reviewing this pilot's results.
"""
import argparse
import logging
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench_common import io_utils
from orchestrator.config import load_model_registry
from orchestrator.manifest_builder import build_full_manifest, build_pilot_manifest, write_manifest_and_gt
from orchestrator.subprocess_runner import run_models_scheduled
from orchestrator.metrics.aggregate import aggregate_model, write_model_metrics, build_comparison_table
from orchestrator.report.generate_report import generate_model_report, generate_comparison_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAMPLES_ROOT = os.environ.get("SARCASM_DATA_ROOT", "data/samples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=os.path.join(_ROOT, "outputs"))
    parser.add_argument("--pilot-size", type=int, default=18)
    parser.add_argument("--pilot-seed", type=int, default=42)
    parser.add_argument("--sequential", action="store_true", help="run models one at a time instead of parallel across idle GPUs")
    parser.add_argument("--stall-minutes", type=int, default=15)
    parser.add_argument("--models", nargs="*", default=None, help="subset of model names to run (default: all 5)")
    args = parser.parse_args()

    output_root = args.output_root
    manifests_dir = os.path.join(output_root, "manifests")
    results_root = os.path.join(output_root, "results")
    metrics_dir = os.path.join(output_root, "metrics")
    reports_dir = os.path.join(output_root, "reports")
    for d in (manifests_dir, results_root, metrics_dir, reports_dir):
        os.makedirs(d, exist_ok=True)

    logger.info("Building full manifest to sample the pilot subset from...")
    full_manifest, gt_dicts, validity_report = build_full_manifest(SAMPLES_ROOT)
    logger.info("Validity report: %s", validity_report)

    pilot_manifest = build_pilot_manifest(full_manifest, n=args.pilot_size, seed=args.pilot_seed)
    pilot_ids = {r["sample_id"] for r in pilot_manifest}
    pilot_gt = [g for g in gt_dicts if g["sample_id"] in pilot_ids]

    manifest_path = os.path.join(manifests_dir, "pilot_manifest.jsonl")
    gt_path = os.path.join(manifests_dir, "pilot_ground_truth.jsonl")
    write_manifest_and_gt(pilot_manifest, pilot_gt, manifest_path, gt_path)
    logger.info("Pilot manifest: %d samples -> %s", len(pilot_manifest), manifest_path)

    specs = load_model_registry()
    if args.models:
        specs = {k: v for k, v in specs.items() if k in args.models}
    logger.info("Running models: %s (parallel=%s)", list(specs.keys()), not args.sequential)

    pilot_results_root = os.path.join(results_root, "pilot")
    os.makedirs(pilot_results_root, exist_ok=True)
    run_results = run_models_scheduled(
        list(specs.values()), manifest_path, pilot_results_root,
        parallel=not args.sequential, stall_minutes=args.stall_minutes,
    )

    gt_lookup = {g["sample_id"]: g["gt_category"] for g in pilot_gt}
    figures_dir = os.path.join(reports_dir, "figures")
    summaries = []
    for rr in run_results:
        results_path = rr.results_path
        run_meta_path = os.path.join(pilot_results_root, rr.model_name, f"{rr.model_name}_run_meta.json")
        summary = aggregate_model(rr.model_name, results_path, gt_path, run_meta_path, dataset_size=len(pilot_manifest))
        summary_path = os.path.join(metrics_dir, f"{rr.model_name}_pilot_metrics.json")
        write_model_metrics(summary, summary_path)
        results = io_utils.read_jsonl(results_path)
        generate_model_report(rr.model_name, summary, results, figures_dir, reports_dir)
        summaries.append(summary)
        logger.info(
            "[%s] accuracy=%s unparseable_rate=%s timeout_count=%s",
            rr.model_name, summary["accuracy"], summary["unparseable_rate"], summary["timeout_count"],
        )

    comparison_df = build_comparison_table(summaries)
    comparison_csv = os.path.join(metrics_dir, "pilot_comparison_summary.csv")
    comparison_df.to_csv(comparison_csv, index=False)
    generate_comparison_report(comparison_df, reports_dir)
    logger.info("Wrote comparison table -> %s", comparison_csv)
    print("\n" + comparison_df.to_string(index=False))

    print(
        "\nPHASE A COMPLETE. Review results under:\n"
        f"  {pilot_results_root}/<model>/results.jsonl (raw per-sample output)\n"
        f"  {metrics_dir}/*_pilot_metrics.json\n"
        f"  {comparison_csv}\n"
        "Run run_full.py explicitly (with --i-reviewed-the-pilot) once satisfied -- this does NOT auto-chain."
    )


if __name__ == "__main__":
    main()
