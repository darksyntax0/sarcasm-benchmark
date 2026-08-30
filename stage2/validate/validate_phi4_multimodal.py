#!/usr/bin/env python3
"""Stage 2 validation: Phi-4-multimodal-instruct, TWO audios (context+target) in one request.

Mirrors sarcasm_ICL/run_phi4_multimodal.py with minimal fixes:
  - candidate default --attn flash_attention_2, but flash-attn is NOT installed -> use "eager".
  - load audio via librosa at 16 kHz (float32) instead of sf.read (float64, no resample), matching
    the Stage-1-proven path; Phi-4's processor kwarg `audios=[(wav,sr),...]` (plural) is correct.
Adds a tensor-level assertion that BOTH clips reached the model (audio_embed_sizes has 2 entries).
"""
import json, os, sys, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from samples import load_validation_samples

OUT = os.path.join(os.path.dirname(__file__), "out", "phi4_multimodal.json")
MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
USER, ASSISTANT, END = "<|user|>", "<|assistant|>", "<|end|>"

verdict = {"model": "phi4_multimodal", "script_loads": False, "two_audios_accepted": False,
           "audio_verification": None, "text_accepted": False, "generation_works": False,
           "per_sample": [], "notes": "", "error": None}

def build_prompt(ct, tt):
    body = ("You are analyzing a spoken conversation for sarcasm.\n\n"
            "CONTEXT (preceding utterance) audio: <|audio_1|>\n"
            f"Context transcription: {ct}\n\n"
            "TARGET (utterance to analyze) audio: <|audio_2|>\n"
            f"Target transcription: {tt}\n\n"
            "Analyze the target utterance using the context. Determine whether the target contains sarcasm and explain your reasoning.")
    return f"{USER}{body}{END}{ASSISTANT}"

def audio_count(inputs):
    for k in ("audio_embed_sizes", "input_audio_embeds"):
        if k in inputs and hasattr(inputs[k], "shape") and inputs[k].ndim >= 1:
            return int(inputs[k].shape[0])
    return -1

try:
    import librosa, torch
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    # FIX: eager attention (flash-attn not installed); explicit fp16 + .to(cuda) (Stage-1 proven)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16,
        _attn_implementation="eager", low_cpu_mem_usage=False).to("cuda")
    model.eval()
    gen_cfg = GenerationConfig.from_pretrained(MODEL_ID)
    verdict["script_loads"] = True

    for s in load_validation_samples():
        # FIX: librosa @16k float32 (candidate used sf.read -> float64, no resample)
        ctx, _ = librosa.load(s["context_audio"], sr=16000)
        tgt, _ = librosa.load(s["target_audio"], sr=16000)
        audios = [(ctx, 16000), (tgt, 16000)]
        prompt = build_prompt(s["context_text"], s["target_text"])
        inputs = processor(text=prompt, audios=audios, return_tensors="pt").to("cuda")
        n_aud = audio_count(inputs)
        verdict["audio_verification"] = f"audio-embed count={n_aud} (keys={[k for k in inputs.keys()]})"
        if n_aud == 2:
            verdict["two_audios_accepted"] = True
        verdict["text_accepted"] = int(inputs["input_ids"].shape[1]) > 0
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=200, generation_config=gen_cfg, do_sample=False)
        resp = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        verdict["generation_works"] = len(resp.strip()) > 0
        verdict["per_sample"].append({"sample_id": s["sample_id"], "audio_count": n_aud,
                                      "response_len": len(resp), "response_head": resp[:300]})
    verdict["notes"] = "Phi-4: 2 placeholders <|audio_1|>/<|audio_2|> + 2 (wav,sr) tuples; audio_embed_sizes==2 confirms both clips."
except Exception as e:
    verdict["error"] = f"{type(e).__name__}: {e}"
    traceback.print_exc()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(verdict, open(OUT, "w"), indent=2)
print(json.dumps({k: verdict[k] for k in ["model","script_loads","two_audios_accepted","text_accepted","generation_works","error"]}, indent=2))
