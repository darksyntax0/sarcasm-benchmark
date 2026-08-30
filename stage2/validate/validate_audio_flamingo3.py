#!/usr/bin/env python3
"""Stage 2 validation: Audio-Flamingo-3, TWO audios (context+target) in one request.

Mirrors sarcasm_ICL/run_audio_flamingo3.py (single-turn mode) with ONE minimal fix:
  - candidate loads with device_map="auto" and NO dtype -> fp32 -> CPU offload -> timeouts
    (Stage-1 finding). Fixed: torch_dtype=torch.float16. The candidate's
    .to(model.device, dtype=model.dtype) correctly casts float feature tensors only.
Adds a tensor-level assertion that BOTH clips reached the model (input_features rows == 2).
Falls back to the candidate's multi-turn shape if single-turn yields != 2 audio features.
"""
import json, os, sys, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from samples import load_validation_samples

OUT = os.path.join(os.path.dirname(__file__), "out", "audio_flamingo3.json")
MODEL_ID = "nvidia/audio-flamingo-3-hf"

verdict = {"model": "audio_flamingo3", "script_loads": False, "two_audios_accepted": False,
           "audio_verification": None, "text_accepted": False, "generation_works": False,
           "mode_used": None, "per_sample": [], "notes": "", "error": None}

def single_turn(ct, tt, ca, ta):
    return [{"role": "user", "content": [
        {"type": "text", "text": "You are analyzing a spoken conversation for sarcasm.\n\nCONTEXT (preceding utterance):"},
        {"type": "audio", "path": ca},
        {"type": "text", "text": f"Context transcription: {ct}\n\nTARGET (utterance to analyze):"},
        {"type": "audio", "path": ta},
        {"type": "text", "text": f"Target transcription: {tt}\n\nAnalyze the target utterance using the context. Determine whether the target contains sarcasm and explain your reasoning."},
    ]}]

def multi_turn(ct, tt, ca, ta):
    return [
        {"role": "user", "content": [
            {"type": "text", "text": f"You are analyzing a spoken conversation for sarcasm. This first clip is the CONTEXT. Its transcription is: {ct}. Acknowledge it and wait for the target."},
            {"type": "audio", "path": ca}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Understood. I have the context. Please provide the target."}]},
        {"role": "user", "content": [
            {"type": "text", "text": f"This second clip is the TARGET utterance. Its transcription is: {tt}. Using the context above, determine whether the target contains sarcasm and explain your reasoning."},
            {"type": "audio", "path": ta}]},
    ]

def audio_rows(inputs):
    for k in ("input_features", "input_features_mask"):
        if k in inputs and hasattr(inputs[k], "shape") and inputs[k].ndim >= 1:
            return int(inputs[k].shape[0])
    return -1

try:
    import torch
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # FIX: explicit fp16 (candidate omitted dtype -> fp32/CPU offload/timeouts)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    verdict["script_loads"] = True

    for s in load_validation_samples():
        # AF3's processor enforces 1 audio per text turn: two audios in one user turn raises
        # "Got 1 text but 2 audios; they must match 1:1." Try single-turn, and on EITHER that
        # error OR a row-count != 2, fall back to the officially-documented multi-turn shape.
        used_mode = "single"
        try:
            conv = single_turn(s["context_text"], s["target_text"], s["context_audio"], s["target_audio"])
            inputs = processor.apply_chat_template(conv, tokenize=True, add_generation_prompt=True, return_dict=True).to(model.device, dtype=model.dtype)
            rows = audio_rows(inputs)
            if rows != 2:
                raise ValueError(f"single-turn produced {rows} audio rows")
        except Exception as single_err:
            used_mode = "multiturn"
            verdict["notes"] = f"single-turn rejected ({type(single_err).__name__}: {single_err}); using multi-turn (1 audio/turn)."
            conv = multi_turn(s["context_text"], s["target_text"], s["context_audio"], s["target_audio"])
            inputs = processor.apply_chat_template(conv, tokenize=True, add_generation_prompt=True, return_dict=True).to(model.device, dtype=model.dtype)
            rows = audio_rows(inputs)
        verdict["mode_used"] = used_mode
        verdict["audio_verification"] = f"audio-feature rows={rows} (mode={used_mode})"
        if rows == 2:
            verdict["two_audios_accepted"] = True
        verdict["text_accepted"] = int(inputs["input_ids"].shape[1]) > 0
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=200)
        resp = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        verdict["generation_works"] = len(resp.strip()) > 0
        verdict["per_sample"].append({"sample_id": s["sample_id"], "mode": used_mode, "audio_rows": rows,
                                      "response_len": len(resp), "response_head": resp[:300]})
    if not verdict["notes"]:
        verdict["notes"] = "AF3: each <sound> block -> one audio-feature row; rows==2 confirms both clips fused."
except Exception as e:
    verdict["error"] = f"{type(e).__name__}: {e}"
    traceback.print_exc()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(verdict, open(OUT, "w"), indent=2)
print(json.dumps({k: verdict[k] for k in ["model","script_loads","two_audios_accepted","text_accepted","generation_works","mode_used","error"]}, indent=2))
