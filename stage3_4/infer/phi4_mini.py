#!/usr/bin/env python3
"""Stage 3/4: Phi-4-mini-instruct (env: phi4_mini_text, transformers 4.49.0)."""
from _common_text import run_model
run_model(dict(model_key="phi4_mini", model_id="microsoft/Phi-4-mini-instruct",
               conda_env="phi4_mini_text", dtype="bfloat16", trust_remote_code=True))
