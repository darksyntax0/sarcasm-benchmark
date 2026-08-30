#!/usr/bin/env python3
"""
Standalone multi-audio inference: Audio-Flamingo-3 (nvidia/audio-flamingo-3-hf)
==============================================================================
Task: sarcasm analysis of a TARGET utterance given a CONTEXT utterance.

Source this script is based on (verbatim API, verified 2026-08-21):
  - Official HF Transformers model doc + model card:
      https://huggingface.co/docs/transformers/model_doc/audioflamingo3
      https://huggingface.co/nvidia/audio-flamingo-3-hf
    Canonical call (single audio + multi-turn examples are the ONLY ones shown):
        inputs = processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True
        ).to(model.device, dtype=model.dtype)
        outputs = model.generate(**inputs, max_new_tokens=500)
        processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    The processor uses the literal placeholder token "<sound>" (audio_token='<sound>')
    and does "replace-in-place" fusion: each {"type":"audio"} block expands ONE
    <sound> placeholder into that clip's post-pool frames.

Multi-audio representation & IMPORTANT CAVEAT:
  Each {"type":"audio","path":...} block becomes one <sound> token. Mechanically,
  placing TWO audio blocks in a single user turn yields two <sound> tokens, each
  replaced by its own clip. HOWEVER, NVIDIA's official examples only demonstrate
  (a) one audio per turn and (b) multi-TURN chat (one audio per turn). Two audios
  in ONE turn is not in the official examples, so this script offers BOTH:
    - build_conversation_single_turn(): both audios in one user turn (primary)
    - build_conversation_multi_turn():  context in turn 1, target in turn 2
                                        (mirrors the officially-documented shape)
  Use --mode multiturn if the single-turn variant misbehaves on your build.

Dependencies:
  pip install "transformers>=4.57" accelerate torch soundfile librosa
Hardware: ~18-20 GB VRAM. Load in the model's own dtype (see note): the AF-Whisper
tower keeps some LayerNorms in fp32, so a naive bf16 load can raise a dtype error;
following the docs, we do device_map="auto" then cast tensors to model.dtype.
"""

import argparse
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

MODEL_ID = "nvidia/audio-flamingo-3-hf"


def build_conversation_single_turn(context_text, target_text,
                                   context_audio, target_audio):
    """Both audios (-> two <sound> tokens) in a single user turn."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "You are analyzing a spoken conversation for sarcasm.\n\n"
                         "CONTEXT (preceding utterance):"},
                {"type": "audio", "path": context_audio},
                {"type": "text", "text": f"Context transcription: {context_text}\n\n"
                                         "TARGET (utterance to analyze):"},
                {"type": "audio", "path": target_audio},
                {"type": "text",
                 "text": (f"Target transcription: {target_text}\n\n"
                          "Analyze the target utterance using the context. "
                          "Determine whether the target contains sarcasm and "
                          "explain your reasoning.")},
            ],
        }
    ]


def build_conversation_multi_turn(context_text, target_text,
                                  context_audio, target_audio):
    """One audio per turn: officially-documented multi-audio shape for AF3."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "You are analyzing a spoken conversation for sarcasm. "
                         "This first clip is the CONTEXT (preceding utterance). "
                         f"Its transcription is: {context_text}. "
                         "Acknowledge it and wait for the target."},
                {"type": "audio", "path": context_audio},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text",
                         "text": "Understood. I have the context. Please provide the target."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "This second clip is the TARGET utterance. Its transcription is: "
                         f"{target_text}. Using the context above, determine whether the "
                         "target contains sarcasm and explain your reasoning."},
                {"type": "audio", "path": target_audio},
            ],
        },
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-audio", default="context.wav")
    ap.add_argument("--context-text", default="context.txt")
    ap.add_argument("--target-audio", default="target.wav")
    ap.add_argument("--target-text", default="target.txt")
    ap.add_argument("--mode", choices=["single", "multiturn"], default="single")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    context_text = open(args.context_text, encoding="utf-8").read().strip()
    target_text = open(args.target_text, encoding="utf-8").read().strip()

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map="auto"
    )

    builder = (build_conversation_single_turn if args.mode == "single"
               else build_conversation_multi_turn)
    conversation = builder(context_text, target_text,
                           args.context_audio, args.target_audio)

    # apply_chat_template does the whole job: renders <sound> placeholders, reads
    # each `path`, extracts Whisper features, and returns the fused input dict.
    inputs = processor.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True, return_dict=True
    ).to(model.device, dtype=model.dtype)

    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    response = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0]

    print("=" * 80)
    print(response)
    print("=" * 80)


if __name__ == "__main__":
    main()
