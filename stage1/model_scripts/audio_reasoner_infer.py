"""Stage 1 inference script for zhifeixie/Audio-Reasoner. Run only with the audio-reasoner env
interpreter. Uses ms-swift's PtEngine (not raw transformers) -- most bespoke of the 5 scripts.

Audio-Reasoner's system prompt trains it to always wrap output in a fixed
<THINK><PLANNING>...<CAPTION>...<REASONING>...<SUMMARY></THINK><RESPONSE>...</RESPONSE>
structure. Asking it for bare JSON doesn't override that -- instead we ask it to append a
FINAL_JSON: marker line after </RESPONSE>, and if that's missing or unparseable, fall back to
synthesizing predicted fields from the THINK subsections themselves.
"""
import re
import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from swift.llm import InferRequest, PtEngine, RequestConfig

from bench_common import io_utils, json_extract, prompt
from bench_common.constants import FINAL_JSON_MARKER
from _model_infer_base import build_arg_parser, run_model_loop

CHECKPOINT_PATH = os.environ.get("AUDIO_REASONER_CKPT", "checkpoints/Audio-Reasoner-7B")

SYSTEM_PROMPT = (
    "You are an audio deep-thinking model. Upon receiving a question, please respond in two "
    "parts: <THINK> and <RESPONSE>. The <THINK> section should be further divided into four "
    "parts: <PLANNING>, <CAPTION>, <REASONING>, and <SUMMARY>."
)

_SECTION_RE = {
    "planning": re.compile(r"<PLANNING>(.*?)</PLANNING>", re.DOTALL),
    "caption": re.compile(r"<CAPTION>(.*?)</CAPTION>", re.DOTALL),
    "reasoning": re.compile(r"<REASONING>(.*?)</REASONING>", re.DOTALL),
    "summary": re.compile(r"<SUMMARY>(.*?)</SUMMARY>", re.DOTALL),
}

print("[audio_reasoner] loading PtEngine...", flush=True)
engine = PtEngine(CHECKPOINT_PATH, max_batch_size=1, model_type="qwen2_audio")
print("[audio_reasoner] model loaded.", flush=True)


def _synthesize_fallback_json(raw: str) -> str:
    """If the model didn't emit a usable FINAL_JSON line, synthesize a best-effort JSON object
    from its THINK subsections so the sample isn't wasted purely to a parse_error. This is a
    fallback, not a substitute for the model's own JSON -- it's clearly a lower-confidence path
    and the resulting predicted_category will be None if PLANNING/SUMMARY don't mention one of
    the exact category strings.
    """
    from bench_common.constants import VALID_CATEGORIES

    sections = {}
    for name, pattern in _SECTION_RE.items():
        m = pattern.search(raw)
        sections[name] = m.group(1).strip() if m else ""

    summary_and_reasoning = f"{sections.get('summary', '')} {sections.get('reasoning', '')}".upper()
    predicted = None
    for cat in VALID_CATEGORIES:
        if cat in summary_and_reasoning:
            predicted = cat
            break

    reasoning_steps = [sections[k] for k in ("planning", "caption", "reasoning", "summary") if sections.get(k)]
    explanation = sections.get("summary") or sections.get("reasoning") or ""

    import json as _json

    return _json.dumps(
        {
            "predicted_category": predicted,
            "confidence": None,
            "explanation": explanation,
            "reasoning_steps": reasoning_steps,
        }
    )


def make_infer_fn(max_new_tokens: int):
    request_config = RequestConfig(max_tokens=max_new_tokens, temperature=0.1, stream=False)

    def infer_fn(record: dict) -> str:
        # ms-swift's InferRequest has no processor-output dict to inspect, so this is a weaker
        # pre-flight check than the other 4 scripts' tensor-shape assertion -- documented as an
        # intentional asymmetry, not a bug.
        if not os.path.exists(record["target_wav_path"]) or os.path.getsize(record["target_wav_path"]) == 0:
            from _model_infer_base import AudioInputError

            raise AudioInputError(f"audio file missing or empty: {record['target_wav_path']}")

        text_prompt = prompt.build_stage1_prompt(record["target_text"], use_final_json_marker=True)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": record["target_wav_path"]},
                    {"type": "text", "text": text_prompt},
                ],
            },
        ]
        resp_list = engine.infer([InferRequest(messages=messages)], request_config)
        raw = resp_list[0].choices[0].message.content

        # If the marker path won't yield usable JSON, append a synthesized fallback JSON line so
        # run_model_loop's shared extractor (called with prefer_marker=FINAL_JSON_MARKER after
        # this function returns) still finds something -- keeps all JSON-extraction logic in one
        # place rather than duplicating it here.
        parsed, _ = json_extract.extract_json(raw, prefer_marker=FINAL_JSON_MARKER)
        if parsed is None:
            raw = raw + f"\n{FINAL_JSON_MARKER} " + _synthesize_fallback_json(raw)
        return raw

    return infer_fn


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    manifest_records = io_utils.read_jsonl(args.manifest)
    run_model_loop(
        model_name="audio_reasoner",
        manifest_records=manifest_records,
        infer_fn=make_infer_fn(args.max_new_tokens),
        output_path=args.output,
        timeout_seconds=args.timeout_seconds,
        resume=not args.no_resume,
        json_marker=FINAL_JSON_MARKER,
    )
