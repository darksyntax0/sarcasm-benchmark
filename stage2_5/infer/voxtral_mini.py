#!/usr/bin/env python3
"""Stage 2.5: Voxtral-Mini-3B-2507 — TARGET audio + TARGET text + CONTEXT text."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration
from stage2_5_prompt import build_stage2_5_prompt
from _stage2_5_base import build_arg_parser, run_stage2_5_loop, read_jsonl, AudioInputError

MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
print("[voxtral_mini] loading...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = VoxtralForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")
model.eval()
print("[voxtral_mini] ready", flush=True)

def make_infer_fn(max_new_tokens):
    def infer_fn(rec):
        task = build_stage2_5_prompt(rec["context_text"], rec["target_text"])
        conv = [{"role": "user", "content": [
            {"type": "text", "text": "TARGET utterance audio:"},
            {"type": "audio", "path": rec["target_audio"]},
            {"type": "text", "text": task},
        ]}]
        inputs = processor.apply_chat_template(conv).to(model.device, dtype=torch.bfloat16)
        # Voxtral chunks a clip into 30s windows -> rows >= 1 for a single clip.
        if "input_features" not in inputs or inputs["input_features"].shape[0] < 1:
            raise AudioInputError("no audio features produced for target clip")
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        return processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return infer_fn

if __name__ == "__main__":
    a = build_arg_parser().parse_args()
    run_stage2_5_loop("voxtral_mini", read_jsonl(a.manifest), make_infer_fn(a.max_new_tokens),
                      a.output, a.timeout_seconds, resume=not a.no_resume)
