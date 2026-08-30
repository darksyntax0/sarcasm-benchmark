"""Shared Stage-2 inference driver, imported by each infer/<model>.py.

Same robustness contract as Stage 1's _model_infer_base (load model once, loop all samples,
per-sample SIGALRM timeout, append+flush+fsync each result, resume from prior results), but:
  - each sample carries TWO audios (context + target) + TWO transcripts,
  - JSON parsing uses the vendored stage2/json_extract.py,
  - result records use the Stage-2 prediction schema (stage2_prompt.REQUIRED_PRED_KEYS).
Stage 1 code is never imported or modified.
"""
import argparse, json, os, re, signal, sys, time
from contextlib import contextmanager
from datetime import datetime, timezone

_STAGE2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _STAGE2 not in sys.path:
    sys.path.insert(0, _STAGE2)

import json_extract
from stage2_prompt import REQUIRED_PRED_KEYS, FINAL_JSON_MARKER


class PerSampleTimeout(Exception): pass
class AudioInputError(Exception): pass


@contextmanager
def sample_alarm(timeout_seconds: int):
    def _handler(signum, frame):
        raise PerSampleTimeout(f"inference exceeded {timeout_seconds}s")
    prev = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def gpu_mem_stats():
    try:
        import torch
        if torch.cuda.is_available():
            return {"gpu_mem_allocated_mb": torch.cuda.memory_allocated() / 1048576,
                    "gpu_mem_reserved_mb": torch.cuda.memory_reserved() / 1048576}
    except Exception:
        pass
    return {"gpu_mem_allocated_mb": None, "gpu_mem_reserved_mb": None}


def read_jsonl(path):
    if not os.path.exists(path): return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: out.append(json.loads(line))
    return out


def append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())


_Q = r"['\"]"  # single or double quote


def _field(text, key, val_pat):
    m = re.search(rf"{_Q}{key}{_Q}\s*:\s*{val_pat}", text)
    return m.group(1) if m else None


def salvage_stage2_fields(text):
    """Last-resort recovery when full-JSON parse fails: regex-extract the SCALAR prediction fields
    from Python-dict-repr / malformed output (observed with Qwen2-Audio, which emits single-quoted
    dicts whose trailing explanation/reasoning_steps contain nested apostrophes that break both
    json and ast, while is_sarcastic/predicted_category/sarcasm_source/sarcasm_target/confidence
    remain clean and unambiguous). Returns a dict with whatever was recovered, or None if not even
    predicted_category is present. Explanation is best-effort (may stop at an inner quote)."""
    cat = _field(text, "predicted_category", rf"{_Q}([^'\"]+){_Q}")
    if not cat:
        return None
    iss = _field(text, "is_sarcastic", r"(true|false|True|False)")
    if iss is not None:
        iss = iss.lower() == "true"
    conf = _field(text, "confidence", r"([0-9]*\.?[0-9]+)")
    return {
        "is_sarcastic": iss,
        "predicted_category": cat,
        "sarcasm_source": _field(text, "sarcasm_source", rf"{_Q}([^'\"]*){_Q}"),
        "sarcasm_target": _field(text, "sarcasm_target", rf"{_Q}([^'\"]*){_Q}"),
        "confidence": float(conf) if conf else None,
        "explanation": _field(text, "explanation", rf"{_Q}([^'\"]{{0,400}})"),
        "reasoning_steps": None,
    }


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--timeout-seconds", type=int, required=True)
    ap.add_argument("--max-new-tokens", type=int, required=True)
    ap.add_argument("--no-resume", action="store_true")
    return ap


def _utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_stage2_loop(model_name, manifest_records, infer_fn, output_path, timeout_seconds,
                    resume=True, json_marker=None):
    """infer_fn(record) -> raw_text. record has: sample_id, context_audio, target_audio,
    context_text, target_text."""
    done = {r["sample_id"] for r in read_jsonl(output_path)} if resume else set()
    if done:
        print(f"[{model_name}] resume: {len(done)} already done", flush=True)
    total = len(manifest_records)
    for i, rec in enumerate(manifest_records):
        sid = rec["sample_id"]
        if sid in done:
            continue
        t0 = time.monotonic()
        result = {"sample_id": sid, "model_name": model_name,
                  "is_sarcastic": None, "predicted_category": None, "sarcasm_source": None,
                  "sarcasm_target": None, "confidence": None, "explanation": None,
                  "reasoning_steps": None, "raw_output": "", "parse_error": None,
                  "salvaged": False, "audio_input_error": None, "error": None,
                  "latency_seconds": None, "timestamp": _utcnow()}
        try:
            with sample_alarm(timeout_seconds):
                raw = infer_fn(rec)
            result["raw_output"] = raw
            parsed, err = json_extract.extract_json(raw, prefer_marker=json_marker)
            if parsed:
                for k in REQUIRED_PRED_KEYS:
                    result[k] = parsed.get(k)
            else:
                # Full-JSON parse failed -> field-level salvage of the scalar prediction fields.
                salv = salvage_stage2_fields(raw)
                if salv:
                    for k in REQUIRED_PRED_KEYS:
                        result[k] = salv.get(k)
                    result["salvaged"] = True
                else:
                    result["parse_error"] = err
        except PerSampleTimeout:
            result["error"] = "timeout"
        except AudioInputError as e:
            result["audio_input_error"] = str(e)
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
        result["latency_seconds"] = time.monotonic() - t0
        result.update(gpu_mem_stats())
        append_jsonl(output_path, result)
        print(f"[{model_name}] ({i+1}/{total}) {sid} {result['latency_seconds']:.1f}s "
              f"err={result['error']} audio_err={bool(result['audio_input_error'])} "
              f"parse_err={bool(result['parse_error'])}", flush=True)
