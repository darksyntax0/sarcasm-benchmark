#!/usr/bin/env python3
"""Pre-flight check: verifies environment variables, conda interpreters, and dataset
layout WITHOUT loading a model or touching a GPU. Run this before any stage."""
import os, sys, subprocess

REQUIRED_ENVS = ["qwen2-audio-env", "audio-flamingo", "phi4-env", "audio-reasoner",
                 "phi4_mini_text", "benchmark-orchestrator"]
REQUIRED_ENVS_CONDA3 = ["voxtral"]
ok = True

def check(label, cond, hint=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    if not cond:
        ok = False
        if hint:
            print(f"         -> {hint}")
    return cond

print("Environment variables")
conda_root  = os.environ.get("CONDA_ROOT")
conda3_root = os.environ.get("CONDA3_ROOT", conda_root)
data_root   = os.environ.get("SARCASM_DATA_ROOT")
ckpt        = os.environ.get("AUDIO_REASONER_CKPT")
check("CONDA_ROOT set", bool(conda_root), "export CONDA_ROOT=$HOME/miniconda")
check("SARCASM_DATA_ROOT set", bool(data_root),
      "export SARCASM_DATA_ROOT=/path/to/conversational-sarcasm-benchmark/samples")
if os.environ.get("HF_HOME"):
    print("  [PASS] HF_HOME set")
else:
    print("  [WARN] HF_HOME not set - weights will download to ~/.cache (large)")

print("\nConda interpreters")
if conda_root:
    for e in REQUIRED_ENVS:
        p = os.path.join(conda_root, "envs", e, "bin", "python")
        check(f"{e}", os.path.isfile(p), f"missing: {p}  (see envs/{e}.yml)")
if conda3_root:
    for e in REQUIRED_ENVS_CONDA3:
        p = os.path.join(conda3_root, "envs", e, "bin", "python")
        check(f"{e} (CONDA3_ROOT)", os.path.isfile(p), f"missing: {p}")

print("\nDataset layout")
if data_root and os.path.isdir(data_root):
    dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    check(f"dataset root readable ({len(dirs)} sample dirs)", len(dirs) > 0)
    if dirs:
        s = os.path.join(data_root, sorted(dirs)[0])
        for f in ["target.wav", "target.txt", "context.wav", "context.txt",
                  "sarcasm_analysis.json"]:
            check(f"sample artifact: {f}", os.path.isfile(os.path.join(s, f)))
else:
    check("dataset root exists", False,
          "download from https://huggingface.co/datasets/darksyntax0/conversational-sarcasm-benchmark")

print("\nAudio-Reasoner checkpoint (Stages 1, 2, 2.5 only)")
check("AUDIO_REASONER_CKPT present", bool(ckpt) and os.path.isdir(ckpt or ""),
      "download zhifeixie/Audio-Reasoner and set AUDIO_REASONER_CKPT")

print("\nGPU")
try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
    for line in out.stdout.strip().splitlines():
        print(f"         {line}")
    check("nvidia-smi available", out.returncode == 0)
except Exception as e:
    check("nvidia-smi available", False, str(e))

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED - see hints above"))
sys.exit(0 if ok else 1)
