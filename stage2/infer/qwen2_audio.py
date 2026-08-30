#!/usr/bin/env python3
"""Stage 2 inference: Qwen2-Audio-7B-Instruct (context+target audio in one request)."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import librosa, torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from stage2_prompt import build_task_text
from _stage2_base import build_arg_parser, run_stage2_loop, read_jsonl, AudioInputError

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
print("[qwen2_audio] loading...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
model.eval()
SR = processor.feature_extractor.sampling_rate
print("[qwen2_audio] ready", flush=True)


def make_infer_fn(max_new_tokens):
    def infer_fn(rec):
        task = build_task_text(rec["context_text"], rec["target_text"])
        conv = [{"role": "user", "content": [
            {"type": "text", "text": "CONTEXT audio:"},
            {"type": "audio", "audio_url": rec["context_audio"]},
            {"type": "text", "text": "TARGET audio:"},
            {"type": "audio", "audio_url": rec["target_audio"]},
            {"type": "text", "text": task},
        ]}]
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios = [librosa.load(rec["context_audio"], sr=SR)[0], librosa.load(rec["target_audio"], sr=SR)[0]]
        inputs = processor(text=text, audio=audios, return_tensors="pt", padding=True).to(model.device)
        if inputs.input_features.shape[0] != 2:
            raise AudioInputError(f"expected 2 audio feature rows, got {inputs.input_features.shape[0]}")
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens)
        return processor.batch_decode(gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return infer_fn


if __name__ == "__main__":
    a = build_arg_parser().parse_args()
    run_stage2_loop("qwen2_audio", read_jsonl(a.manifest), make_infer_fn(a.max_new_tokens),
                    a.output, a.timeout_seconds, resume=not a.no_resume)
