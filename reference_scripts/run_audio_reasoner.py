#!/usr/bin/env python3
"""
Standalone multi-audio inference: Audio-Reasoner (zhifeixie/Audio-Reasoner)
==========================================================================
Task: sarcasm analysis of a TARGET utterance given a CONTEXT utterance.

Source this script is based on (verbatim API, verified 2026-08-21):
  - Official repo inference.py / README ("Chat using ms-swift"):
      https://github.com/xzf-thu/Audio-Reasoner
      https://huggingface.co/zhifeixie/Audio-Reasoner
    Audio-Reasoner is a CoT-SFT checkpoint on top of Qwen2-Audio-Instruct and is run
    through the ms-swift (`swift.llm`) engine, NOT raw transformers:
        engine = PtEngine(checkpoint, max_batch_size=64, model_type='qwen2_audio')
        infer_stream(engine, InferRequest(messages=get_message(audiopath, prompt)))
    with the EXACT system prompt reproduced below and a per-message content list of
        {'type':'audio','audio': <path>} , {'type':'text','text': <prompt>}
    The structured output is <THINK>(<PLANNING>/<CAPTION>/<REASONING>/<SUMMARY>) + <RESPONSE>.

Multi-audio representation:
  Inherited from the Qwen2-Audio backbone (model_type='qwen2_audio'): the ms-swift
  qwen2_audio template renders one <audio> slot per audio entry in the content list.
  The official get_message uses ONE audio; here we extend it to TWO entries
  (context, then target) in the same user message -- mechanically identical to
  Qwen2-Audio multi-audio. NOTE: multi-audio is NOT independently validated for
  Audio-Reasoner in the paper/repo, so treat it as "inherited, not benchmarked".

Dependencies:
  pip install ms-swift librosa soundfile accelerate torch
  pip install transformers==4.48.0    # repo-pinned; version affects CoT quality
Hardware: ~19-21 GB VRAM (7B, fp16). Set the checkpoint path first (see --checkpoint).
"""

import argparse
from swift.llm import InferRequest, PtEngine, RequestConfig
from swift.plugin import InferStats

# Verbatim Audio-Reasoner system prompt (from official inference.py)
SYSTEM = (
    "You are an audio deep-thinking model. Upon receiving a question, please "
    "respond in two parts: <THINK> and <RESPONSE>. The <THINK> section should be "
    "further divided into four parts: <PLANNING>, <CAPTION>, <REASONING>, and "
    "<SUMMARY>."
)


def get_message(context_audio, target_audio, prompt):
    """Extends the official single-audio get_message() to two audio entries."""
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": context_audio},   # CONTEXT clip
                {"type": "audio", "audio": target_audio},     # TARGET clip
                {"type": "text", "text": prompt},
            ],
        },
    ]


def infer_stream(engine, infer_request):
    """Verbatim streaming helper from the official repo."""
    request_config = RequestConfig(max_tokens=2048, temperature=0, stream=True)
    metric = InferStats()
    gen = engine.infer([infer_request], request_config, metrics=[metric])
    output = ""
    print("response: ", end="")
    for resp_list in gen:
        if resp_list[0] is None:
            continue
        delta = resp_list[0].choices[0].delta.content
        print(delta, end="", flush=True)
        output += delta
    print()
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="zhifeixie/Audio-Reasoner",
                    help="HF id or local path to the Audio-Reasoner-7B checkpoint")
    ap.add_argument("--context-audio", default="context.wav")
    ap.add_argument("--context-text", default="context.txt")
    ap.add_argument("--target-audio", default="target.wav")
    ap.add_argument("--target-text", default="target.txt")
    args = ap.parse_args()

    context_text = open(args.context_text, encoding="utf-8").read().strip()
    target_text = open(args.target_text, encoding="utf-8").read().strip()

    # The semantic prompt; audio order is context-then-target (see get_message)
    prompt = (
        "You are analyzing a spoken conversation for sarcasm.\n\n"
        "CONTEXT: the first audio clip.\n"
        f"Context transcription: {context_text}\n\n"
        "TARGET: the second audio clip.\n"
        f"Target transcription: {target_text}\n\n"
        "Analyze the target utterance using the context. Determine whether the "
        "target contains sarcasm and explain your reasoning."
    )

    engine = PtEngine(args.checkpoint, max_batch_size=64, model_type="qwen2_audio")
    messages = get_message(args.context_audio, args.target_audio, prompt)

    print("=" * 80)
    infer_stream(engine, InferRequest(messages=messages))
    print("=" * 80)


if __name__ == "__main__":
    main()
