#!/usr/bin/env python3
"""
Standalone multi-audio inference: Voxtral-Mini-3B-2507 (mistralai/Voxtral-Mini-3B-2507)
=======================================================================================
Task: sarcasm analysis of a TARGET utterance given a CONTEXT utterance.

Source this script is based on (verbatim API, verified 2026-08-21):
  - Official HF Transformers model doc:
      https://huggingface.co/docs/transformers/model_doc/voxtral
    It contains a dedicated "multi-audio + text instruction" example: a single user
    message with TWO {"type":"audio","path":...} blocks followed by one text block.
    Canonical call (NOTE: there is NO separate `audios=` kwarg -- apply_chat_template
    returns the full input dict directly):
        inputs = processor.apply_chat_template(conversation)
        inputs = inputs.to(model.device, dtype=torch.bfloat16)
        outputs = model.generate(**inputs, max_new_tokens=500)
        processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

Multi-audio representation:
  Content blocks {"type":"audio","path":...}|{"type":"audio","url":...} plus
  {"type":"text","text":...}. The Mistral-common tokenizer inserts audio tokens for
  each clip; multiple audios per message AND multi-turn audio are both officially
  supported. apply_chat_template handles feature extraction internally, so you must
  NOT call processor(text=..., audios=...) yourself (that path does not accept audio).

Dependencies:
  pip install "transformers>=4.54" "mistral-common[audio]" accelerate torch soundfile
Hardware: ~9.5 GB VRAM (bf16). Up to ~40 min audio for understanding, 32k ctx.
"""

import argparse
import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration

MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"


def build_conversation(context_text, target_text, context_audio, target_audio):
    """Single user turn with both audios then the text instruction."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "You are analyzing a spoken conversation for sarcasm.\n"
                         "The first audio is the CONTEXT (preceding utterance)."},
                {"type": "audio", "path": context_audio},
                {"type": "text",
                 "text": f"Context transcription: {context_text}\n"
                         "The second audio is the TARGET (utterance to analyze)."},
                {"type": "audio", "path": target_audio},
                {"type": "text",
                 "text": (f"Target transcription: {target_text}\n\n"
                          "Analyze the target utterance using the context. "
                          "Determine whether the target contains sarcasm and "
                          "explain your reasoning.")},
            ],
        }
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-audio", default="context.wav")
    ap.add_argument("--context-text", default="context.txt")
    ap.add_argument("--target-audio", default="target.wav")
    ap.add_argument("--target-text", default="target.txt")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    context_text = open(args.context_text, encoding="utf-8").read().strip()
    target_text = open(args.target_text, encoding="utf-8").read().strip()

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = VoxtralForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )

    conversation = build_conversation(
        context_text, target_text, args.context_audio, args.target_audio
    )

    # apply_chat_template returns a ready-to-generate BatchFeature (ids + audio feats)
    inputs = processor.apply_chat_template(conversation)
    inputs = inputs.to(model.device, dtype=torch.bfloat16)

    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    response = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0]

    print("=" * 80)
    print(response)
    print("=" * 80)


if __name__ == "__main__":
    main()
