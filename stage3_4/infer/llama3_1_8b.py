#!/usr/bin/env python3
"""Stage 3/4: Llama-3.1-8B-Instruct (env: qwen2-audio-env)."""
from _common_text import run_model
run_model(dict(model_key="llama3_1_8b", model_id="meta-llama/Llama-3.1-8B-Instruct",
               conda_env="qwen2-audio-env", dtype="bfloat16"))
