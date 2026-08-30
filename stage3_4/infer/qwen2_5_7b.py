#!/usr/bin/env python3
"""Stage 3/4: Qwen2.5-7B-Instruct (env: qwen2-audio-env)."""
from _common_text import run_model
run_model(dict(model_key="qwen2_5_7b", model_id="Qwen/Qwen2.5-7B-Instruct",
               conda_env="qwen2-audio-env", dtype="bfloat16"))
