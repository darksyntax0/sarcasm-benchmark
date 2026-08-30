#!/usr/bin/env python3
"""
Standalone multi-audio inference: Qwen2-Audio-7B-Instruct
=========================================================
Task: sarcasm analysis of a TARGET utterance given a CONTEXT utterance,
passing BOTH audios + BOTH transcriptions in a SINGLE inference request.

Source this script is based on (verbatim API, verified 2026-08-21):
  - Official HF model card / GitHub README:
      https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
      https://github.com/QwenLM/Qwen2-Audio
    The card demonstrates a conversation containing TWO audio blocks and the
    canonical extraction loop:
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(librosa.load(ele["audio_url"], sr=...)[0])
        inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
  - Multi-audio validity is enforced by the processor itself
    (transformers/models/qwen2_audio/processing_qwen2_audio.py): it raises if the
    number of <|AUDIO|> tokens != number of audios, i.e. multiple audios per prompt
    is a first-class feature.

Multi-audio representation:
  ChatML. Each {"type": "audio"} content block is rendered by the chat template as
      Audio N: <|audio_bos|><|AUDIO|><|audio_eos|>
  and the ordered list of decoded waveforms is passed to processor(audios=[...]).
  Text is placed as sibling {"type": "text"} blocks in the same "content" list.

Dependencies:
  pip install "transformers>=4.44" accelerate torch librosa
Hardware: ~16-19 GB VRAM (bf16/fp16). Audio best kept <=30 s per clip.
"""

import argparse
import librosa
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"


def build_conversation(context_text: str, target_text: str,
                       context_audio: str, target_audio: str):
    """Single user turn holding both audios + both transcriptions (ChatML)."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "You are analyzing a spoken conversation for sarcasm.\n\nCONTEXT:"},
                {"type": "audio", "audio_url": context_audio},
                {"type": "text", "text": f"Context transcription: {context_text}\n\nTARGET:"},
                {"type": "audio", "audio_url": target_audio},
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
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="auto"
    )

    conversation = build_conversation(
        context_text, target_text, args.context_audio, args.target_audio
    )

    # 1) Render ChatML text (does NOT insert audio features, only placeholders)
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )

    # 2) Decode every audio block, in order, at the model's native sample rate
    sr = processor.feature_extractor.sampling_rate  # 16000
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(librosa.load(ele["audio_url"], sr=sr)[0])

    # 3) Fuse text placeholders + ordered waveforms
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)

    # 4) Generate and strip the prompt tokens
    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_ids = generated_ids[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("=" * 80)
    print(response)
    print("=" * 80)


if __name__ == "__main__":
    main()
