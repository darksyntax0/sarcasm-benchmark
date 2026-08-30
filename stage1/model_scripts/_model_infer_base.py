"""Shared driver imported directly by each of the 5 *_infer.py scripts (same directory, no
sys.path trick needed). Each model script loads its model ONCE at module scope, defines
infer_fn(record) -> raw_text_output, and calls run_model_loop(...) -- this is what makes the
whole benchmark run without reloading a multi-GB model per sample.
"""
import argparse
import signal
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench_common import io_utils, json_extract
from bench_common.constants import REQUIRED_PRED_KEYS


class PerSampleTimeout(Exception):
    pass


class AudioInputError(Exception):
    pass


@contextmanager
def sample_alarm(timeout_seconds: int):
    """signal.alarm(timeout_seconds) wrapped tightly around one inference call. Cancels the
    alarm in a finally block on both success and exception so it never leaks into the next
    sample's timing. Unix-only (fine -- this always runs on the Linux cluster)."""

    def _handler(signum, frame):
        raise PerSampleTimeout(f"inference exceeded {timeout_seconds}s")

    previous_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def assert_audio_present(inputs, key_candidates, min_numel: int = 1) -> None:
    """Generic guard against the exact silent-audio-drop failure mode hit with Qwen2-Audio's
    `audios=` vs `audio=` kwarg rename: check that at least one of key_candidates exists in the
    processor output with a non-trivial tensor. Raises AudioInputError if not -- this must be a
    loud, distinct failure, never a silently-ignored one."""
    for key in key_candidates:
        if key in inputs:
            tensor = inputs[key]
            numel = tensor.numel() if hasattr(tensor, "numel") else len(tensor)
            if numel >= min_numel:
                return
    raise AudioInputError(
        f"none of the expected audio keys {key_candidates} present with data in processor "
        f"output (got keys: {list(inputs.keys()) if hasattr(inputs, 'keys') else type(inputs)})"
    )


def get_gpu_mem_stats() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "gpu_mem_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                "gpu_mem_reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
            }
    except ImportError:
        pass
    return {"gpu_mem_allocated_mb": None, "gpu_mem_reserved_mb": None}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="path to manifest .jsonl")
    parser.add_argument("--output", required=True, help="path to results .jsonl (appended to)")
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--no-resume", action="store_true", help="ignore existing results, rerun all")
    return parser


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_model_loop(model_name, manifest_records, infer_fn, output_path, timeout_seconds, resume=True,
                    json_marker=None):
    """Core loop reused identically by all 5 scripts. `infer_fn(record: dict) -> str` must build
    the prompt+audio, call the model's generate/infer, and return the raw decoded text -- all
    timing/memory/error handling/JSON-extraction/resume logic lives here, once."""
    done_ids = io_utils.read_processed_sample_ids(output_path) if resume else set()
    if done_ids:
        print(f"[{model_name}] resuming: {len(done_ids)} samples already in {output_path}", flush=True)

    total = len(manifest_records)
    for i, record in enumerate(manifest_records):
        sample_id = record["sample_id"]
        if sample_id in done_ids:
            continue

        t0 = time.monotonic()
        result = {
            "sample_id": sample_id,
            "model_name": model_name,
            "predicted_category": None,
            "confidence": None,
            "explanation": None,
            "reasoning_steps": None,
            "raw_output": "",
            "parse_error": None,
            "audio_input_error": None,
            "error": None,
            "latency_seconds": None,
            "timestamp": _utcnow_iso(),
        }

        try:
            with sample_alarm(timeout_seconds):
                raw = infer_fn(record)
            result["raw_output"] = raw
            parsed, err = json_extract.extract_json(raw, prefer_marker=json_marker)
            if parsed:
                for key in REQUIRED_PRED_KEYS:
                    result[key] = parsed.get(key)
            else:
                result["parse_error"] = err
        except PerSampleTimeout:
            result["error"] = "timeout"
        except AudioInputError as e:
            result["audio_input_error"] = str(e)
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"

        result["latency_seconds"] = time.monotonic() - t0
        result.update(get_gpu_mem_stats())
        io_utils.append_jsonl(output_path, result)
        print(
            f"[{model_name}] ({i + 1}/{total}) {sample_id} done in "
            f"{result['latency_seconds']:.1f}s (error={result['error']}, "
            f"audio_input_error={bool(result['audio_input_error'])}, "
            f"parse_error={bool(result['parse_error'])})",
            flush=True,
        )
