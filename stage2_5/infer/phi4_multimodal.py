#!/usr/bin/env python3
"""Stage 2.5: Phi-4-multimodal-instruct — TARGET audio + TARGET text + CONTEXT text."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import librosa, torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from stage2_5_prompt import build_stage2_5_prompt
from _stage2_5_base import build_arg_parser, run_stage2_5_loop, read_jsonl, AudioInputError

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
print("[phi4_multimodal] loading...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16,
    _attn_implementation="eager", low_cpu_mem_usage=False).to("cuda")
model.eval()
GEN_CFG = GenerationConfig.from_pretrained(MODEL_ID)
print("[phi4_multimodal] ready", flush=True)

def make_infer_fn(max_new_tokens):
    def infer_fn(rec):
        task = build_stage2_5_prompt(rec["context_text"], rec["target_text"])
        prompt = f"<|user|>TARGET audio: <|audio_1|>\n\n{task}<|end|><|assistant|>"
        tgt = librosa.load(rec["target_audio"], sr=16000)[0]
        inputs = processor(text=prompt, audios=[(tgt, 16000)], return_tensors="pt").to("cuda")
        n = inputs["audio_embed_sizes"].shape[0] if "audio_embed_sizes" in inputs else -1
        if n != 1:
            raise AudioInputError(f"expected exactly 1 audio embed (target), got {n}")
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, generation_config=GEN_CFG, do_sample=False)
        return processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return infer_fn

if __name__ == "__main__":
    a = build_arg_parser().parse_args()
    run_stage2_5_loop("phi4_multimodal", read_jsonl(a.manifest), make_infer_fn(a.max_new_tokens),
                      a.output, a.timeout_seconds, resume=not a.no_resume)
