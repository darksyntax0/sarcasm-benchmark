"""Shared loader/runner for standard causal-LM text models (Stage 3/4).

Most instruction LLMs load identically (AutoTokenizer + AutoModelForCausalLM + chat template);
this helper captures that path and records full reproducibility metadata. Per-model scripts pass a
config dict and override only what genuinely differs (dtype, trust_remote_code, thinking mode,
generation params). To keep the comparison fair, EVERY model uses the same message structure: the
full self-contained task prompt in a single user turn (no system role), identical wording, greedy
decoding.
"""
import os, sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from _text_base import (build_arg_parser, make_chat_generate, read_jsonl, run_text_loop)

_DTYPES = {"float16": "float16", "bfloat16": "bfloat16"}


def run_model(cfg):
    """cfg keys: model_key, model_id, conda_env, dtype ('float16'|'bfloat16'),
    trust_remote_code (bool), template_kwargs (dict), attn ('eager'|'sdpa'),
    extra_gen (dict, merged into generation kwargs)."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = build_arg_parser().parse_args()
    dtype = getattr(torch, cfg.get("dtype", "float16"))
    trc = cfg.get("trust_remote_code", False)

    tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=trc)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print(f"[{cfg['model_key']}] loading {cfg['model_id']} ({cfg.get('dtype','float16')})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], torch_dtype=dtype, device_map="auto",
        trust_remote_code=trc, attn_implementation=cfg.get("attn", "eager"),
    )
    model.eval()
    print(f"[{cfg['model_key']}] ready", flush=True)

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                      pad_token_id=tok.pad_token_id)
    gen_kwargs.update(cfg.get("extra_gen", {}))

    chat_generate = make_chat_generate(
        model, tok, gen_kwargs, system_prompt=None,
        template_kwargs=cfg.get("template_kwargs", {}),
    )

    static_meta = {
        "model_id": cfg["model_id"],
        "conda_env": cfg["conda_env"],
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "dtype": cfg.get("dtype", "float16"),
        "attn_implementation": cfg.get("attn", "eager"),
        "gen_max_new_tokens": args.max_new_tokens,
        "gen_do_sample": False,
        "gen_template_kwargs": str(cfg.get("template_kwargs", {})),
    }

    run_text_loop(args.stage, cfg["model_key"], read_jsonl(args.manifest), chat_generate,
                  args.output, args.timeout_seconds, static_meta, resume=not args.no_resume)
