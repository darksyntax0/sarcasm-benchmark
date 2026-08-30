#!/usr/bin/env python3
"""Stage 2 orchestrator (self-contained; does not import or modify Stage 1).

Builds a Stage-2 manifest (both clips + both transcripts) + ground_truth, then runs each model's
inference script in its own conda env as a long-lived subprocess pinned to an idle GPU. Results go
to results/stage2/<model>/results.jsonl -- SEPARATE from Stage 1's outputs.

Usage:
  python run_stage2.py --build-manifest-only
  python run_stage2.py --limit 3                      # small validation subset, all models
  python run_stage2.py --limit 3 --models qwen2_audio # subset of models
  python run_stage2.py --full --i-reviewed-the-validation   # full 867-sample run
"""
import argparse, json, os, subprocess, sys, time
from datetime import datetime, timezone

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from samples import iter_valid_samples, load_sample

RESULTS_ROOT = os.path.join(HERE, "results", "stage2")
MANIFEST_DIR = os.path.join(HERE, "results", "manifests")
POLL = 20


def _utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_manifest(limit=None):
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    ids = list(iter_valid_samples())
    if limit:
        ids = ids[:limit]
    man_path = os.path.join(MANIFEST_DIR, f"manifest{'_pilot' if limit else '_full'}.jsonl")
    gt_path = os.path.join(MANIFEST_DIR, f"ground_truth{'_pilot' if limit else '_full'}.jsonl")
    with open(man_path, "w") as mf, open(gt_path, "w") as gf:
        for sid in ids:
            s = load_sample(sid)
            mf.write(json.dumps({"sample_id": sid, "context_audio": s["context_audio"],
                                 "target_audio": s["target_audio"], "context_text": s["context_text"],
                                 "target_text": s["target_text"]}) + "\n")
            gt = s["ground_truth"]
            gf.write(json.dumps({"sample_id": sid, "gt_category": gt.get("category"),
                                 "gt_sarcasm_source": gt.get("sarcasm_source"),
                                 "gt_sarcasm_target": gt.get("sarcasm_target"),
                                 "gt_explanation": gt.get("explanation")}) + "\n")
    print(f"manifest: {len(ids)} samples -> {man_path}")
    return man_path, gt_path, len(ids)


def idle_gpus(min_free_mb=20000, max_util=30, exclude=()):
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    cands = []
    for line in out.strip().splitlines():
        idx, used, total, util = [x.strip() for x in line.split(",")]
        idx, used, total, util = int(idx), float(used), float(total), float(util)
        if (total - used) >= min_free_mb and util <= max_util and idx not in exclude:
            cands.append((idx, total - used))
    cands.sort(key=lambda x: -x[1])
    return [c[0] for c in cands]


def load_registry():
    with open(os.path.join(HERE, "configs", "stage2_models.yaml")) as f:
        return yaml.safe_load(f)["models"]


def launch(name, spec, manifest, gpu):
    model_dir = os.path.join(RESULTS_ROOT, name)
    os.makedirs(model_dir, exist_ok=True)
    results = os.path.join(model_dir, "results.jsonl")
    log = os.path.join(model_dir, "run.log")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("TMPDIR", os.environ.get("TMPDIR", "/tmp"))
    
    cmd = [os.path.expandvars(spec["interpreter_path"]), os.path.join(HERE, spec["script"]),
           "--manifest", manifest, "--output", results,
           "--timeout-seconds", str(spec["timeout_seconds"]),
           "--max-new-tokens", str(spec["max_new_tokens"])]
    lf = open(log, "a")
    print(f"[{name}] launch on GPU {gpu} -> {results}", flush=True)
    p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return {"name": name, "proc": p, "lf": lf, "results": results, "gpu": gpu,
            "started": _utcnow(), "last_size": -1, "last_prog": time.monotonic()}


def run(models, manifest, parallel=True, stall_min=20, min_free_mb=20000):
    reg = load_registry()
    pending = [(n, reg[n]) for n in models]
    running, reserved, done = {}, set(), []

    def try_launch():
        still = []
        for name, spec in pending:
            gps = idle_gpus(min_free_mb=min_free_mb, exclude=tuple(reserved))
            if not gps:
                still.append((name, spec)); continue
            g = gps[0]
            running[name] = launch(name, spec, manifest, g)
            reserved.add(g)
            if not parallel:
                break
        pending[:] = still

    try_launch()
    while pending or running:
        time.sleep(POLL)
        for name in list(running):
            st = running[name]; p = st["proc"]; ret = p.poll()
            cur = os.path.getsize(st["results"]) if os.path.exists(st["results"]) else 0
            if ret is None:
                if cur != st["last_size"]:
                    st["last_size"] = cur; st["last_prog"] = time.monotonic(); continue
                if time.monotonic() - st["last_prog"] > stall_min * 60:
                    print(f"[{name}] STALL >{stall_min}m, killing", flush=True)
                    p.kill(); p.wait(); ret = p.returncode
                else:
                    continue
            st["lf"].close()
            n_written = sum(1 for _ in open(st["results"])) if os.path.exists(st["results"]) else 0
            meta = {"model": name, "gpu": st["gpu"], "started": st["started"], "ended": _utcnow(),
                    "exit_code": ret, "results_written": n_written}
            json.dump(meta, open(os.path.join(RESULTS_ROOT, name, "run_meta.json"), "w"), indent=2)
            print(f"[{name}] done exit={ret} written={n_written}", flush=True)
            done.append(meta); reserved.discard(st["gpu"]); del running[name]
        if pending:
            try_launch()
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-manifest-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="run only first N valid samples (small test)")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--full", action="store_true", help="run the complete dataset")
    ap.add_argument("--i-reviewed-the-validation", action="store_true",
                    help="required tripwire for --full")
    ap.add_argument("--min-free-mb", type=int, default=20000)
    args = ap.parse_args()

    reg = load_registry()
    models = args.models or list(reg.keys())
    for m in models:
        if m not in reg:
            sys.exit(f"unknown model: {m}")

    if args.full and not args.limit:
        if not args.i_reviewed_the_validation:
            sys.exit("Refusing full run without --i-reviewed-the-validation")
        manifest, gt, n = build_manifest(limit=None)
    else:
        limit = args.limit or 3
        manifest, gt, n = build_manifest(limit=limit)

    if args.build_manifest_only:
        return
    print(f"Running Stage 2: models={models} samples={n} parallel={not args.sequential}")
    done = run(models, manifest, parallel=not args.sequential, min_free_mb=args.min_free_mb)
    print("\nSUMMARY:")
    for d in done:
        print(f"  {d['model']:16s} exit={d['exit_code']} written={d['results_written']}/{n} gpu={d['gpu']}")


if __name__ == "__main__":
    main()
