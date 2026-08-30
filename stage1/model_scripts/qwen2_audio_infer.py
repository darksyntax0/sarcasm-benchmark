"""Stage 1 inference script for Qwen/Qwen2-Audio-7B-Instruct. Run only with the qwen2-audio-env
interpreter (see configs/model_registry.yaml). Loads the model ONCE, then loops over every
manifest sample via run_model_loop.
"""
import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import librosa
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from bench_common import io_utils, prompt
from _model_infer_base import assert_audio_present, build_arg_parser, run_model_loop

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

print("[qwen2_audio] loading processor and model...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto"
)
model.eval()
print("[qwen2_audio] model loaded.", flush=True)


def make_infer_fn(max_new_tokens: int):
    def infer_fn(record: dict) -> str:
        text_prompt = prompt.build_stage1_prompt(record["target_text"])
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": record["target_wav_path"]},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        chat_text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audio_data, _ = librosa.load(
            record["target_wav_path"], sr=processor.feature_extractor.sampling_rate
        )

        # Real bug hit previously: this env's transformers renamed the processor kwarg from
        # `audios=` (plural) to `audio=` (singular) -- the old name is silently ignored (no
        # error, just a warning), which fed the model text-only input and produced a hallucinated
        # "successful" response. Must use singular `audio=`.
        inputs = processor(text=chat_text, audio=[audio_data], return_tensors="pt", padding=True).to(
            model.device
        )
        assert_audio_present(inputs, key_candidates=("input_features", "feature_attention_mask"))

        with torch.inference_mode():
            generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response

    return infer_fn


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    manifest_records = io_utils.read_jsonl(args.manifest)
    run_model_loop(
        model_name="qwen2_audio",
        manifest_records=manifest_records,
        infer_fn=make_infer_fn(args.max_new_tokens),
        output_path=args.output,
        timeout_seconds=args.timeout_seconds,
        resume=not args.no_resume,
    )
