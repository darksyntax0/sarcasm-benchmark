"""Stage 1 inference script for microsoft/Phi-4-multimodal-instruct. Run only with the
phi4-env interpreter. Loads via AutoModelForCausalLM + trust_remote_code=True (the top-level
Phi4MMForCausalLM class is NOT exported by this env's transformers build, but the
trust_remote_code path pulls the model's own bundled modeling code and works regardless --
confirmed by a standalone processor+model smoke test)."""
import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import librosa
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from bench_common import io_utils, prompt
from _model_infer_base import assert_audio_present, build_arg_parser, run_model_loop

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"

print("[phi4] loading processor and model...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
# This env's transformers (5.16.0.dev0) still initializes the model skeleton on the meta device
# during from_pretrained even with low_cpu_mem_usage=False -- Phi-4's custom conformer-encoder
# init code (NemoConvSubsampling) calls int() on a tensor it expects to be real, which fails on
# a meta (shape-only) tensor with "Tensor.item() cannot be called on meta tensors". Forcing the
# torch default device to real CPU during construction sidesteps whatever internal meta-device
# context transformers is otherwise applying.
with torch.device("cpu"):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        _attn_implementation="eager",
        device_map=None,
        low_cpu_mem_usage=False,
    )
model = model.to("cuda")
model.eval()
print("[phi4] model loaded.", flush=True)


def make_infer_fn(max_new_tokens: int):
    def infer_fn(record: dict) -> str:
        text_prompt = prompt.build_stage1_prompt(record["target_text"])
        audio_data, sample_rate = librosa.load(record["target_wav_path"], sr=16000)
        full_prompt = f"<|user|><|audio_1|>{text_prompt}<|end|><|assistant|>"

        inputs = processor(
            text=full_prompt, audios=[(audio_data, sample_rate)], return_tensors="pt"
        ).to("cuda")
        assert_audio_present(inputs, key_candidates=("input_audio_embeds", "audio_embed_sizes"))

        with torch.inference_mode():
            generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        response = processor.batch_decode(
            generate_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        return response

    return infer_fn


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    manifest_records = io_utils.read_jsonl(args.manifest)
    run_model_loop(
        model_name="phi4_multimodal",
        manifest_records=manifest_records,
        infer_fn=make_infer_fn(args.max_new_tokens),
        output_path=args.output,
        timeout_seconds=args.timeout_seconds,
        resume=not args.no_resume,
    )
