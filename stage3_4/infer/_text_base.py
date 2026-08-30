"""Shared Stage 3/4 (text-only) inference driver, imported by each infer/<model>.py.

Same robustness/record contract as Stage 2 (load model once, loop, per-sample SIGALRM timeout,
append+flush+fsync each row, resume from written ids, robust JSON parse + field salvage), plus a
full per-row reproducibility record (model name/path, conda env, transformers version, dtype,
generation params, stage). Prompt construction is shared here (identical wording across models =
fair comparison); each model only supplies chat-templating + generation.
Stage 1/2 code is never imported or modified.
"""
import argparse, json, os, re, signal, sys, time
from contextlib import contextmanager
from datetime import datetime, timezone

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json_extract
from text_prompt import REQUIRED_PRED_KEYS, build_prompt


class PerSampleTimeout(Exception): pass


@contextmanager
def sample_alarm(timeout_seconds: int):
    def _h(sig, frm):
        raise PerSampleTimeout(f"inference exceeded {timeout_seconds}s")
    prev = signal.signal(signal.SIGALRM, _h)
    signal.alarm(timeout_seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def gpu_mem():
    try:
        import torch
        if torch.cuda.is_available():
            return {"gpu_mem_allocated_mb": torch.cuda.memory_allocated() / 1048576,
                    "gpu_mem_reserved_mb": torch.cuda.memory_reserved() / 1048576}
    except Exception:
        pass
    return {"gpu_mem_allocated_mb": None, "gpu_mem_reserved_mb": None}


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())


_Q = r"['\"]"


def _field(text, key, val_pat):
    m = re.search(rf"{_Q}{key}{_Q}\s*:\s*{val_pat}", text)
    return m.group(1) if m else None


def salvage_fields(text):
    """Last-resort scalar-field recovery from dict-repr / malformed output (same approach proven
    in Stage 2 for weak-JSON models)."""
    cat = _field(text, "predicted_category", rf"{_Q}([^'\"]+){_Q}")
    if not cat:
        return None
    iss = _field(text, "is_sarcastic", r"(true|false|True|False)")
    if iss is not None:
        iss = iss.lower() == "true"
    conf = _field(text, "confidence", r"([0-9]*\.?[0-9]+)")
    return {"is_sarcastic": iss, "predicted_category": cat,
            "sarcasm_source": _field(text, "sarcasm_source", rf"{_Q}([^'\"]*){_Q}"),
            "sarcasm_target": _field(text, "sarcasm_target", rf"{_Q}([^'\"]*){_Q}"),
            "confidence": float(conf) if conf else None,
            "explanation": _field(text, "explanation", rf"{_Q}([^'\"]{{0,400}})"),
            "reasoning_steps": None}


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[3, 4])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--timeout-seconds", type=int, required=True)
    ap.add_argument("--max-new-tokens", type=int, required=True)
    ap.add_argument("--no-resume", action="store_true")
    return ap


def _utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_text_loop(stage, model_name, manifest_records, chat_generate, output_path,
                  timeout_seconds, static_meta, resume=True, json_marker=None):
    """chat_generate(prompt_str) -> raw_text. static_meta: dict of reproducibility fields recorded
    on every row (model_path, conda_env, transformers_version, dtype, gen params)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    done = {r["sample_id"] for r in read_jsonl(output_path)} if resume else set()
    if done:
        print(f"[{model_name}][stage{stage}] resume: {len(done)} done", flush=True)
    total = len(manifest_records)
    for i, rec in enumerate(manifest_records):
        sid = rec["sample_id"]
        if sid in done:
            continue
        t0 = time.monotonic()
        result = {"sample_id": sid, "model_name": model_name, "stage": stage,
                  "is_sarcastic": None, "predicted_category": None, "sarcasm_source": None,
                  "sarcasm_target": None, "confidence": None, "explanation": None,
                  "reasoning_steps": None, "raw_output": "", "parse_error": None,
                  "salvaged": False, "error": None, "latency_seconds": None,
                  "timestamp": _utcnow()}
        result.update(static_meta)
        try:
            prompt = build_prompt(stage, rec.get("context_text", ""), rec["target_text"],
                                  use_final_json_marker=bool(json_marker))
            with sample_alarm(timeout_seconds):
                raw = chat_generate(prompt)
            result["raw_output"] = raw
            parsed, err = json_extract.extract_json(raw, prefer_marker=json_marker)
            if parsed:
                for k in REQUIRED_PRED_KEYS:
                    result[k] = parsed.get(k)
            else:
                salv = salvage_fields(raw)
                if salv:
                    for k in REQUIRED_PRED_KEYS:
                        result[k] = salv.get(k)
                    result["salvaged"] = True
                else:
                    result["parse_error"] = err
        except PerSampleTimeout:
            result["error"] = "timeout"
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
        result["latency_seconds"] = time.monotonic() - t0
        result.update(gpu_mem())
        append_jsonl(output_path, result)
        print(f"[{model_name}][stage{stage}] ({i+1}/{total}) {sid} {result['latency_seconds']:.1f}s "
              f"err={result['error']} parse_err={bool(result['parse_error'])} salv={result['salvaged']}",
              flush=True)


def make_chat_generate(model, tokenizer, gen_kwargs, system_prompt=None, template_kwargs=None):
    """Generic chat-templated generation shared by most text models. system_prompt=None means the
    model has no system role (e.g. Gemma-2) -> the instruction goes entirely in the user turn.
    template_kwargs lets a model pass e.g. enable_thinking=False (Qwen3)."""
    import torch
    template_kwargs = template_kwargs or {}

    def chat_generate(prompt):
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True, **template_kwargs)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(gen[0], skip_special_tokens=True)

    return chat_generate
