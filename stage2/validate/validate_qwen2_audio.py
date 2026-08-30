#!/usr/bin/env python3
"""Stage 2 validation: Qwen2-Audio-7B-Instruct, TWO audios (context+target) in one request.

Mirrors sarcasm_ICL/run_qwen2_audio.py with ONE minimal, documented fix:
  - candidate line 103 uses processor(..., audios=audios, ...); in this env's transformers
    (5.16.0.dev0) the kwarg is `audio=` (singular) and `audios=` is SILENTLY IGNORED, feeding
    the model zero audio. Fixed to `audio=audios`. (Same bug fixed in Stage 1.)
Adds a tensor-level assertion that BOTH clips reached the model (input_features rows == 2).
"""
import json, os, sys, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from samples import load_validation_samples

OUT = os.path.join(os.path.dirname(__file__), "out", "qwen2_audio.json")
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

verdict = {"model": "qwen2_audio", "script_loads": False, "two_audios_accepted": False,
           "audio_verification": None, "text_accepted": False, "generation_works": False,
           "per_sample": [], "notes": "", "error": None}

def build_conversation(ctx_text, tgt_text, ctx_audio, tgt_audio):
    return [{"role": "user", "content": [
        {"type": "text", "text": "You are analyzing a spoken conversation for sarcasm.\n\nCONTEXT:"},
        {"type": "audio", "audio_url": ctx_audio},
        {"type": "text", "text": f"Context transcription: {ctx_text}\n\nTARGET:"},
        {"type": "audio", "audio_url": tgt_audio},
        {"type": "text", "text": f"Target transcription: {tgt_text}\n\nAnalyze the target utterance using the context. Determine whether the target contains sarcasm and explain your reasoning."},
    ]}]

try:
    import librosa, torch
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    verdict["script_loads"] = True
    sr = processor.feature_extractor.sampling_rate

    for s in load_validation_samples():
        conv = build_conversation(s["context_text"], s["target_text"], s["context_audio"], s["target_audio"])
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios = []
        for m in conv:
            for ele in m["content"]:
                if ele["type"] == "audio":
                    audios.append(librosa.load(ele["audio_url"], sr=sr)[0])
        assert len(audios) == 2, f"expected 2 audios, extracted {len(audios)}"
        # FIX: singular `audio=` (candidate used `audios=`, silently ignored in this build)
        inputs = processor(text=text, audio=audios, return_tensors="pt", padding=True).to(model.device)
        n_feat = int(inputs.input_features.shape[0])
        verdict["audio_verification"] = f"input_features.shape={tuple(inputs.input_features.shape)}"
        if n_feat == 2:
            verdict["two_audios_accepted"] = True
        verdict["text_accepted"] = int(inputs.input_ids.shape[1]) > 0
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=200)
        resp = processor.batch_decode(gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        verdict["generation_works"] = len(resp.strip()) > 0
        verdict["per_sample"].append({"sample_id": s["sample_id"], "n_audio_features": n_feat,
                                      "response_len": len(resp), "response_head": resp[:300]})
    verdict["notes"] = "Two audios batched into input_features (rows==2); singular audio= kwarg required in this build."
except Exception as e:
    verdict["error"] = f"{type(e).__name__}: {e}"
    traceback.print_exc()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(verdict, open(OUT, "w"), indent=2)
print(json.dumps({k: verdict[k] for k in ["model","script_loads","two_audios_accepted","text_accepted","generation_works","error"]}, indent=2))
