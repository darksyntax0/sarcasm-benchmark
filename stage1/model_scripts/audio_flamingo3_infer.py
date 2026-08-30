"""Stage 1 inference script for nvidia/audio-flamingo-3-hf. Run only with the audio-flamingo
env interpreter (see configs/model_registry.yaml)."""
import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

from bench_common import io_utils, prompt
from _model_infer_base import assert_audio_present, build_arg_parser, run_model_loop

MODEL_ID = "nvidia/audio-flamingo-3-hf"

print("[audio_flamingo3] loading processor and model...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto"
)
model.eval()
print("[audio_flamingo3] model loaded.", flush=True)


def make_infer_fn(max_new_tokens: int):
    def infer_fn(record: dict) -> str:
        text_prompt = prompt.build_stage1_prompt(record["target_text"])
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": record["target_wav_path"]},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True
        ).to(model.device)
        assert_audio_present(inputs, key_candidates=("input_features", "input_features_mask"))
        # Processor emits fp32 audio features regardless of model dtype -- cast to match the
        # fp16 model weights or generate() raises a bias/input dtype mismatch.
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(model.dtype)

        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return decoded[0]

    return infer_fn


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    manifest_records = io_utils.read_jsonl(args.manifest)
    run_model_loop(
        model_name="audio_flamingo3",
        manifest_records=manifest_records,
        infer_fn=make_infer_fn(args.max_new_tokens),
        output_path=args.output,
        timeout_seconds=args.timeout_seconds,
        resume=not args.no_resume,
    )
