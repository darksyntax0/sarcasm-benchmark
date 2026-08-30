"""Stage 1 inference script for mistralai/Voxtral-Mini-3B-2507. Run only with the voxtral env
interpreter, which lives under a SEPARATE conda install root (miniconda3, not miniconda) --
see configs/model_registry.yaml."""
import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration

from bench_common import io_utils, prompt
from _model_infer_base import assert_audio_present, build_arg_parser, run_model_loop

MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"

print("[voxtral] loading processor and model...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = VoxtralForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="cuda"
)
model.eval()
print("[voxtral] model loaded.", flush=True)


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
            conversation, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        assert_audio_present(inputs, key_candidates=("input_features",))
        # Same fp32-processor/fp16-model mismatch seen with Audio-Flamingo-3.
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(model.dtype)

        with torch.inference_mode():
            generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        response = processor.batch_decode(
            generate_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )[0]
        return response

    return infer_fn


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    manifest_records = io_utils.read_jsonl(args.manifest)
    run_model_loop(
        model_name="voxtral",
        manifest_records=manifest_records,
        infer_fn=make_infer_fn(args.max_new_tokens),
        output_path=args.output,
        timeout_seconds=args.timeout_seconds,
        resume=not args.no_resume,
    )
