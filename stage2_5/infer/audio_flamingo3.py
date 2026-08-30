#!/usr/bin/env python3
"""Stage 2.5: Audio-Flamingo-3 — TARGET audio + TARGET text + CONTEXT text.

Note: this is the SAME condition AF3 ran in Stage 2 (its processor admits only one audio),
so AF3's Stage 2 and Stage 2.5 results should agree closely — an internal consistency check.
Unlike Stage 2, here the condition is uniform across all five models, so AF3 needs no caveat."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
from stage2_5_prompt import build_stage2_5_prompt
from _stage2_5_base import build_arg_parser, run_stage2_5_loop, read_jsonl, AudioInputError

MODEL_ID = "nvidia/audio-flamingo-3-hf"
print("[audio_flamingo3] loading...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
model.eval()
print("[audio_flamingo3] ready", flush=True)

def make_infer_fn(max_new_tokens):
    def infer_fn(rec):
        task = build_stage2_5_prompt(rec["context_text"], rec["target_text"])
        conv = [{"role": "user", "content": [
            {"type": "text", "text": "TARGET utterance audio:"},
            {"type": "audio", "path": rec["target_audio"]},
            {"type": "text", "text": task},
        ]}]
        inputs = processor.apply_chat_template(conv, tokenize=True, add_generation_prompt=True,
                                               return_dict=True).to(model.device, dtype=model.dtype)
        rows = inputs["input_features"].shape[0] if "input_features" in inputs else -1
        if rows < 1:
            raise AudioInputError(f"no audio features for target clip (rows={rows})")
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return infer_fn

if __name__ == "__main__":
    a = build_arg_parser().parse_args()
    run_stage2_5_loop("audio_flamingo3", read_jsonl(a.manifest), make_infer_fn(a.max_new_tokens),
                      a.output, a.timeout_seconds, resume=not a.no_resume)
