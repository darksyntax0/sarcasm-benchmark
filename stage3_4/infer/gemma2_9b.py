#!/usr/bin/env python3
"""Stage 3/4: Gemma-2-9B-Instruct -> google/gemma-2-9b-it (env: qwen2-audio-env).
MODEL-SPECIFIC: Gemma-2 overflows/NaNs in float16; it must run in bfloat16 (emulated on V100 but
numerically correct). No system role (handled: full task is in the user turn)."""
from _common_text import run_model
run_model(dict(model_key="gemma2_9b", model_id="google/gemma-2-9b-it",
               conda_env="qwen2-audio-env", dtype="bfloat16"))
