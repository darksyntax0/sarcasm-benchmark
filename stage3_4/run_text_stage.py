#!/usr/bin/env python3
"""Stage 3/4 orchestrator (self-contained; does not import or modify Stage 1/2).

Builds a text manifest (sample_id + context_text + target_text) + ground_truth for the SAME 867
split, then runs each text model in its own conda env as a long-lived subprocess pinned to an idle
GPU. Results go to results/stage{3,4}/<model>/results.jsonl -- separate per stage and per model.

Usage:
  python run_text_stage.py --stage 3 --build-manifest-only
  python run_text_stage.py --stage 3 --limit 5                    # small validation subset
  python run_text_stage.py --stage 4 --limit 5 --models qwen2_5_7b
  python run_text_stage.py --stage 3 --full --i-reviewed-the-validation
"""
import argparse, json, os, subprocess, sys, time
from datetime import datetime, timezone

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from samples import iter_valid_samples, load_sample

MANIFEST_DIR = os.path.join(HERE, "results", "manifests")
POLL = 15


def _utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_manifest(limit=None):
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    ids = list(iter_valid_samples())
    if limit:
        ids = ids[:limit]
    tag = "pilot" if limit else "full"
    man = os.path.join(MANIFEST_DIR, f"manifest_{tag}.jsonl")
    gt = os.path.join(MANIFEST_DIR, f"ground_truth_{tag}.jsonl")
    with open(man, "w") as mf, open(gt, "w") as gf:
        for sid in ids:
            s = load_sample(sid)
            mf.write(json.dumps({"sample_id": sid, "context_text": s["context_text"],
                                 "target_text": s["target_text"]}) + "\n")
            g = s["ground_truth"]
            gf.write(json.dumps({"sample_id": sid, "gt_category": g.get("category"),
                                 "gt_sarcasm_source": g.get("sarcasm_source"),
                                 "gt_sarcasm_target": g.get("sarcasm_target"),
                                 "gt_explanation": g.get("explanation")}) + "\n")
    print(f"manifest: {len(ids)} samples -> {man}")
    return man, gt, len(ids)


def idle_gpus(min_free_mb=18000, max_util=40, exclude=()):
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
    with open(os.path.join(HERE, "configs", "text_models.yaml")) as f:
        return yaml.safe_load(f)["models"]


def launch(name, spec, stage, manifest, gpu):
    results_root = os.path.join(HERE, "results", f"stage{stage}")
    model_dir = os.path.join(results_root, name)
    os.makedirs(model_dir, exist_ok=True)
    results = os.path.join(model_dir, "results.jsonl")
    log = os.path.join(model_dir, "run.log")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("TMPDIR", os.environ.get("TMPDIR", "/tmp"))
    
    # All 7 models are pre-cached; load offline so gated repos (Llama-3.1) don't hit a hub 403
    # and runs are hub-independent / reproducible.
    env["HF_HUB_OFFLINE"] = "1"
    cmd = [os.path.expandvars(spec["interpreter_path"]), os.path.join(HERE, spec["script"]),
           "--stage", str(stage), "--manifest", manifest, "--output", results,
           "--timeout-seconds", str(spec["timeout_seconds"]),
           "--max-new-tokens", str(spec["max_new_tokens"])]
    lf = open(log, "a")
    print(f"[{name}] stage{stage} launch on GPU {gpu} -> {results}", flush=True)
    p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return {"name": name, "proc": p, "lf": lf, "results": results, "gpu": gpu,
            "started": _utcnow(), "last_size": -1, "last_prog": time.monotonic()}


def run(models, stage, manifest, parallel=True, stall_min=15, min_free_mb=18000):
    reg = load_registry()
    pending = [(n, reg[n]) for n in models]
    running, reserved, done = {}, set(), []
    results_root = os.path.join(HERE, "results", f"stage{stage}")

    def try_launch():
        still = []
        for name, spec in pending:
            gps = idle_gpus(min_free_mb=min_free_mb, exclude=tuple(reserved))
            if not gps:
                still.append((name, spec)); continue
            running[name] = launch(name, spec, stage, manifest, gps[0])
            reserved.add(gps[0])
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
                    print(f"[{name}] STALL, killing", flush=True); p.kill(); p.wait(); ret = p.returncode
                else:
                    continue
            st["lf"].close()
            n = sum(1 for _ in open(st["results"])) if os.path.exists(st["results"]) else 0
            meta = {"model": name, "stage": stage, "gpu": st["gpu"], "started": st["started"],
                    "ended": _utcnow(), "exit_code": ret, "results_written": n}
            json.dump(meta, open(os.path.join(results_root, name, "run_meta.json"), "w"), indent=2)
            print(f"[{name}] done exit={ret} written={n}", flush=True)
            done.append(meta); reserved.discard(st["gpu"]); del running[name]
        if pending:
            try_launch()
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[3, 4])
    ap.add_argument("--build-manifest-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--i-reviewed-the-validation", action="store_true")
    ap.add_argument("--min-free-mb", type=int, default=18000)
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
        manifest, gt, n = build_manifest(limit=args.limit or 5)

    if args.build_manifest_only:
        return
    print(f"Stage {args.stage}: models={models} samples={n} parallel={not args.sequential}")
    done = run(models, args.stage, manifest, parallel=not args.sequential, min_free_mb=args.min_free_mb)
    print("\nSUMMARY:")
    for d in done:
        print(f"  {d['model']:16s} stage{d['stage']} exit={d['exit_code']} written={d['results_written']}/{n} gpu={d['gpu']}")


if __name__ == "__main__":
    main()
