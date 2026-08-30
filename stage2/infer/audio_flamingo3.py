#!/usr/bin/env python3
"""Stage 2 inference: Audio-Flamingo-3 -- REDUCED-MODALITY VARIANT.

AF3's installed processor hard-blocks >1 audio per request (validate_inputs enforces
len(text)==len(audio); chat templates flatten to one prompt). Native two-audio Stage 2 is
therefore impossible for this model. Per the chosen design, AF3 receives the TARGET audio plus
the CONTEXT as transcription text only. This limitation is stated in the prompt (via
context_audio_available=False) and must be reported as a caveat in any Stage-2 comparison:
AF3's Stage-2 numbers are NOT directly comparable to the four native two-audio models."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
from stage2_prompt import build_task_text
from _stage2_base import build_arg_parser, run_stage2_loop, read_jsonl, AudioInputError

MODEL_ID = "nvidia/audio-flamingo-3-hf"
print("[audio_flamingo3] loading (reduced-modality: target audio + text context)...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
model.eval()
print("[audio_flamingo3] ready", flush=True)


def make_infer_fn(max_new_tokens):
    def infer_fn(rec):
        # Only the TARGET audio; context supplied as text (context_audio_available=False).
        task = build_task_text(rec["context_text"], rec["target_text"], context_audio_available=False)
        conv = [{"role": "user", "content": [
            {"type": "text", "text": "TARGET utterance audio:"},
            {"type": "audio", "path": rec["target_audio"]},
            {"type": "text", "text": task},
        ]}]
        inputs = processor.apply_chat_template(conv, tokenize=True, add_generation_prompt=True,
                                               return_dict=True).to(model.device, dtype=model.dtype)
        rows = inputs["input_features"].shape[0] if "input_features" in inputs else -1
        if rows != 1:
            raise AudioInputError(f"expected 1 (target) audio row, got {rows}")
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return infer_fn


if __name__ == "__main__":
    a = build_arg_parser().parse_args()
    run_stage2_loop("audio_flamingo3", read_jsonl(a.manifest), make_infer_fn(a.max_new_tokens),
                    a.output, a.timeout_seconds, resume=not a.no_resume)
