#!/usr/bin/env python3
"""Stage 2 inference: Audio-Reasoner (ms-swift, context+target audio in one request).

Multi-audio is tensor-verified for this checkpoint (ms-swift qwen2_audio template encodes 2
audios, input_features rows==2 -- confirmed in validation). CoT output is coaxed toward parseable
JSON via a trailing FINAL_JSON: marker (mirrors the Stage-1 approach for this model)."""
import os, re, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from swift.llm import InferRequest, PtEngine, RequestConfig
from stage2_prompt import build_task_text, FINAL_JSON_MARKER, VALID_CATEGORIES
import json_extract
from _stage2_base import build_arg_parser, run_stage2_loop, read_jsonl, AudioInputError

CHECKPOINT = os.environ.get("AUDIO_REASONER_CKPT", "checkpoints/Audio-Reasoner-7B")
SYSTEM = ("You are an audio deep-thinking model. Upon receiving a question, please respond in two "
          "parts: <THINK> and <RESPONSE>. The <THINK> section should be further divided into four "
          "parts: <PLANNING>, <CAPTION>, <REASONING>, and <SUMMARY>.")
_SECT = {n: re.compile(rf"<{n.upper()}>(.*?)</{n.upper()}>", re.DOTALL) for n in ("planning","caption","reasoning","summary")}

print("[audio_reasoner] loading...", flush=True)
engine = PtEngine(CHECKPOINT, max_batch_size=1, model_type="qwen2_audio")
print("[audio_reasoner] ready", flush=True)


def _fallback_json(raw):
    """Synthesize a best-effort record from the CoT structure when Audio-Reasoner doesn't emit a
    parseable FINAL_JSON line. category is matched from SUMMARY/REASONING; is_sarcastic is derived
    from it (any of the 6 sarcasm categories -> true, explicit non-sarcasm cue -> false, else null).
    source/target are left null (honest) unless a clear 'source/target is X' phrase is present."""
    import json as _j
    sect = {n: (m.group(1).strip() if (m := p.search(raw)) else "") for n, p in _SECT.items()}
    blob = f"{sect['summary']} {sect['reasoning']}".upper()
    cat = next((c for c in VALID_CATEGORIES if c in blob), None)
    is_sarc = True if cat else (False if ("NOT SARCAS" in blob or "NO SARCAS" in blob) else None)
    src = (m.group(1).strip() if (m := re.search(r"sarcasm source is ([^.,\n]+)", raw, re.I)) else None)
    tgt = (m.group(1).strip() if (m := re.search(r"(?:sarcasm )?target is ([^.,\n]+)", raw, re.I)) else None)
    steps = [sect[k] for k in ("planning","caption","reasoning","summary") if sect[k]]
    return _j.dumps({"is_sarcastic": is_sarc, "predicted_category": cat, "sarcasm_source": src,
                     "sarcasm_target": tgt, "confidence": None,
                     "explanation": sect["summary"] or sect["reasoning"], "reasoning_steps": steps})


def make_infer_fn(max_new_tokens):
    cfg = RequestConfig(max_tokens=max_new_tokens, temperature=0, stream=False)
    def infer_fn(rec):
        for p in (rec["context_audio"], rec["target_audio"]):
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                raise AudioInputError(f"audio missing/empty: {p}")
        task = build_task_text(rec["context_text"], rec["target_text"], use_final_json_marker=True)
        prompt = ("CONTEXT: the first audio clip. TARGET: the second audio clip.\n\n" + task)
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [
                        {"type": "audio", "audio": rec["context_audio"]},
                        {"type": "audio", "audio": rec["target_audio"]},
                        {"type": "text", "text": prompt}]}]
        raw = engine.infer([InferRequest(messages=messages)], cfg)[0].choices[0].message.content
        if json_extract.extract_json(raw, prefer_marker=FINAL_JSON_MARKER)[0] is None:
            raw = raw + f"\n{FINAL_JSON_MARKER} " + _fallback_json(raw)
        return raw
    return infer_fn


if __name__ == "__main__":
    a = build_arg_parser().parse_args()
    run_stage2_loop("audio_reasoner", read_jsonl(a.manifest), make_infer_fn(a.max_new_tokens),
                    a.output, a.timeout_seconds, resume=not a.no_resume, json_marker=FINAL_JSON_MARKER)
