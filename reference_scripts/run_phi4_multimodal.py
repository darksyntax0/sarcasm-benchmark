#!/usr/bin/env python3
"""
Standalone multi-audio inference: Phi-4-multimodal-instruct (microsoft/Phi-4-multimodal-instruct)
=================================================================================================
Task: sarcasm analysis of a TARGET utterance given a CONTEXT utterance.

Source this script is based on (verbatim API, verified 2026-08-21):
  - Official HF model card README (sample_inference_phi4mm.py path):
      https://huggingface.co/microsoft/Phi-4-multimodal-instruct
    Prompt token format (from the card):
        Single audio:      <|user|><|audio_1|>{prompt}<|end|><|assistant|>
        Multiple media:    <|user|><|image_1|><|image_2|>...<|audio_1|><|end|><|assistant|>
    i.e. each extra clip gets its own 1-indexed placeholder (<|audio_1|>, <|audio_2|>, ...).
    Processing call (card):
        inputs = processor(text=prompt, audios=[(audio, samplerate)], return_tensors='pt')
    -> `audios` is a LIST OF (waveform, sample_rate) TUPLES; N tuples align positionally
       to the N <|audio_k|> placeholders in the prompt string.
  - Multi-audio-in-one-call is exactly the mechanism used by the audio in-context-learning
    work referenced in the research doc (ASR-Adaptation, transcribe_with_shots): it appends
    every shot's (audio, 16000) plus the test (audio, 16000) to one list and passes
    `processor(text=prompt, audios=all_audios, ...)` in a single forward call.

Multi-audio representation:
  A raw prompt STRING with two placeholders <|audio_1|> (context) and <|audio_2|> (target),
  wrapped by <|user|> ... <|end|><|assistant|>. Text is interleaved directly in the string
  around the placeholders. The two waveforms are passed, in order, as audios=[(ctx,sr),(tgt,sr)].

Dependencies:
  pip install "transformers==4.48.2" accelerate torch soundfile peft backoff
  # Phi-4-mm ships custom code (trust_remote_code=True) with speech-LoRA adapters.
  # flash-attn is recommended by MSFT but optional; set attn below to "eager" if unavailable.
Hardware: ~11-13 GB VRAM. ASR/understanding best <=40 s per clip.
"""

import argparse
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"

USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END = "<|end|>"


def build_prompt(context_text: str, target_text: str) -> str:
    """Raw Phi-4 chat string with two audio placeholders (context, then target)."""
    body = (
        "You are analyzing a spoken conversation for sarcasm.\n\n"
        "CONTEXT (preceding utterance) audio: <|audio_1|>\n"
        f"Context transcription: {context_text}\n\n"
        "TARGET (utterance to analyze) audio: <|audio_2|>\n"
        f"Target transcription: {target_text}\n\n"
        "Analyze the target utterance using the context. Determine whether the "
        "target contains sarcasm and explain your reasoning."
    )
    return f"{USER}{body}{END}{ASSISTANT}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-audio", default="context.wav")
    ap.add_argument("--context-text", default="context.txt")
    ap.add_argument("--target-audio", default="target.wav")
    ap.add_argument("--target-text", default="target.txt")
    ap.add_argument("--max-new-tokens", type=int, default=1000)
    ap.add_argument("--attn", default="flash_attention_2",
                    help='set to "eager" or "sdpa" if flash-attn is not installed')
    args = ap.parse_args()

    context_text = open(args.context_text, encoding="utf-8").read().strip()
    target_text = open(args.target_text, encoding="utf-8").read().strip()

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        _attn_implementation=args.attn,
    )
    generation_config = GenerationConfig.from_pretrained(MODEL_ID)

    # Load both clips as (waveform, sample_rate). Order MUST match <|audio_1|>,<|audio_2|>.
    ctx_audio, ctx_sr = sf.read(args.context_audio)
    tgt_audio, tgt_sr = sf.read(args.target_audio)
    audios = [(ctx_audio, ctx_sr), (tgt_audio, tgt_sr)]

    prompt = build_prompt(context_text, target_text)

    inputs = processor(text=prompt, audios=audios, return_tensors="pt").to(model.device)

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        generation_config=generation_config,
        do_sample=False,  # deterministic; MSFT recommends temperature=0 for audio
    )
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("=" * 80)
    print(response)
    print("=" * 80)


if __name__ == "__main__":
    main()
