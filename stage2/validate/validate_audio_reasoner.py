#!/usr/bin/env python3
"""Stage 2 validation: Audio-Reasoner (ms-swift), TWO audios (context+target) in one request.

Mirrors sarcasm_ICL/run_audio_reasoner.py with minimal fixes:
  - default checkpoint is the HF id; use the LOCAL Stage-1 checkpoint to match Stage 1 and avoid
    re-download.
  - non-streaming RequestConfig for a clean single-shot response.
Multi-audio here is INHERITED from the Qwen2-Audio backbone and is NOT independently benchmarked
for Audio-Reasoner. We (a) try to tensor-verify the ms-swift template encodes 2 audios
(input_features rows==2), and (b) confirm the 2-audio CoT inference runs end-to-end. If the
tensor check cannot be introspected from PtEngine, that is reported honestly, not assumed.
"""
import os
import json, os, sys, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from samples import load_validation_samples

OUT = os.path.join(os.path.dirname(__file__), "out", "audio_reasoner.json")
CHECKPOINT = os.environ.get("AUDIO_REASONER_CKPT", "checkpoints/Audio-Reasoner-7B")
SYSTEM = ("You are an audio deep-thinking model. Upon receiving a question, please respond in two "
          "parts: <THINK> and <RESPONSE>. The <THINK> section should be further divided into four "
          "parts: <PLANNING>, <CAPTION>, <REASONING>, and <SUMMARY>.")

verdict = {"model": "audio_reasoner", "script_loads": False, "two_audios_accepted": False,
           "audio_verification": None, "text_accepted": False, "generation_works": False,
           "per_sample": [], "notes": "", "error": None}

def get_message(ctx_audio, tgt_audio, prompt):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "audio", "audio": ctx_audio},
                {"type": "audio", "audio": tgt_audio},
                {"type": "text", "text": prompt}]}]

try:
    from swift.llm import InferRequest, PtEngine, RequestConfig
    engine = PtEngine(CHECKPOINT, max_batch_size=1, model_type="qwen2_audio")
    verdict["script_loads"] = True
    req_cfg = RequestConfig(max_tokens=900, temperature=0, stream=False)

    for s in load_validation_samples():
        prompt = ("You are analyzing a spoken conversation for sarcasm.\n\n"
                  "CONTEXT: the first audio clip.\n"
                  f"Context transcription: {s['context_text']}\n\n"
                  "TARGET: the second audio clip.\n"
                  f"Target transcription: {s['target_text']}\n\n"
                  "Analyze the target utterance using the context. Determine whether the target "
                  "contains sarcasm and explain your reasoning.")
        messages = get_message(s["context_audio"], s["target_audio"], prompt)
        req = InferRequest(messages=messages)

        # (a) best-effort tensor introspection of the ms-swift template
        if verdict["audio_verification"] is None:
            try:
                tmpl = getattr(engine, "default_template", None) or getattr(engine, "template", None)
                enc = tmpl.encode(req) if tmpl is not None else None
                feats = None
                if isinstance(enc, dict):
                    feats = enc.get("input_features")
                if feats is not None and hasattr(feats, "shape"):
                    rows = int(feats.shape[0])
                    verdict["audio_verification"] = f"template input_features rows={rows}"
                    if rows == 2:
                        verdict["two_audios_accepted"] = True
                else:
                    verdict["audio_verification"] = "PtEngine template did not expose input_features; multi-audio inherited from Qwen2-Audio (separately tensor-verified)"
            except Exception as ie:
                verdict["audio_verification"] = f"template introspection unavailable ({type(ie).__name__}); multi-audio inherited from Qwen2-Audio"

        resp_list = engine.infer([req], req_cfg)
        resp = resp_list[0].choices[0].message.content
        verdict["text_accepted"] = True
        verdict["generation_works"] = len(resp.strip()) > 0
        has_cot = "<THINK>" in resp or "<RESPONSE>" in resp or "<SUMMARY>" in resp
        verdict["per_sample"].append({"sample_id": s["sample_id"], "has_cot_structure": has_cot,
                                      "response_len": len(resp), "response_head": resp[:400]})
    if not verdict["two_audios_accepted"]:
        verdict["notes"] = ("Ran 2-audio CoT end-to-end; ms-swift did not raise an audio-count "
                            "mismatch. Tensor-level 2-audio verification not exposed by PtEngine -- "
                            "multi-audio is INHERITED from the Qwen2-Audio processor (tensor-verified "
                            "separately in the qwen2_audio validator), NOT independently benchmarked "
                            "for Audio-Reasoner.")
    else:
        verdict["notes"] = "ms-swift template encoded 2 audios (input_features rows==2); CoT inference ran end-to-end."
except Exception as e:
    verdict["error"] = f"{type(e).__name__}: {e}"
    traceback.print_exc()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(verdict, open(OUT, "w"), indent=2)
print(json.dumps({k: verdict[k] for k in ["model","script_loads","two_audios_accepted","text_accepted","generation_works","error"]}, indent=2))
