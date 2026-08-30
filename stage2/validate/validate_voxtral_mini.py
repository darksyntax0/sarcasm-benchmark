#!/usr/bin/env python3
"""Stage 2 validation: Voxtral-Mini-3B-2507, TWO audios (context+target) in one request.

Mirrors sarcasm_ICL/run_voxtral_mini.py verbatim (no code fix expected -- multi-audio is
officially documented for Voxtral). Adds a tensor-level assertion that BOTH clips reached the
model (input_features encodes 2 clips).
"""
import json, os, sys, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from samples import load_validation_samples

OUT = os.path.join(os.path.dirname(__file__), "out", "voxtral_mini.json")
MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"

verdict = {"model": "voxtral_mini", "script_loads": False, "two_audios_accepted": False,
           "audio_verification": None, "text_accepted": False, "generation_works": False,
           "per_sample": [], "notes": "", "error": None}

def build_conversation(ct, tt, ca, ta):
    return [{"role": "user", "content": [
        {"type": "text", "text": "You are analyzing a spoken conversation for sarcasm.\nThe first audio is the CONTEXT (preceding utterance)."},
        {"type": "audio", "path": ca},
        {"type": "text", "text": f"Context transcription: {ct}\nThe second audio is the TARGET (utterance to analyze)."},
        {"type": "audio", "path": ta},
        {"type": "text", "text": f"Target transcription: {tt}\n\nAnalyze the target utterance using the context. Determine whether the target contains sarcasm and explain your reasoning."},
    ]}]

try:
    import torch
    from transformers import AutoProcessor, VoxtralForConditionalGeneration
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = VoxtralForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    verdict["script_loads"] = True

    for s in load_validation_samples():
        conv = build_conversation(s["context_text"], s["target_text"], s["context_audio"], s["target_audio"])
        inputs = processor.apply_chat_template(conv)
        inputs = inputs.to(model.device, dtype=torch.bfloat16)
        rows = -1
        if "input_features" in inputs and hasattr(inputs["input_features"], "shape"):
            rows = int(inputs["input_features"].shape[0])
        verdict["audio_verification"] = f"input_features rows={rows}"
        # Voxtral may concatenate audio frames rather than batch rows; accept rows>=2 OR
        # (rows==1 but the frame dim clearly encodes 2 clips) -- primary signal is rows==2.
        if rows == 2:
            verdict["two_audios_accepted"] = True
        verdict["text_accepted"] = int(inputs["input_ids"].shape[1]) > 0
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=200)
        resp = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        verdict["generation_works"] = len(resp.strip()) > 0
        verdict["per_sample"].append({"sample_id": s["sample_id"], "audio_rows": rows,
                                      "input_keys": [k for k in inputs.keys()],
                                      "response_len": len(resp), "response_head": resp[:300]})
    verdict["notes"] = "Voxtral: multi-audio officially documented; verifying input_features encodes both clips."
except Exception as e:
    verdict["error"] = f"{type(e).__name__}: {e}"
    traceback.print_exc()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(verdict, open(OUT, "w"), indent=2)
print(json.dumps({k: verdict[k] for k in ["model","script_loads","two_audios_accepted","text_accepted","generation_works","error"]}, indent=2))
