#!/usr/bin/env python3
"""Stage 3/4: Phi-3-mini-4k-instruct (env: phi4-env, transformers 4.48.0). OPTIONAL tier."""
from _common_text import run_model
run_model(dict(model_key="phi3_mini", model_id="microsoft/Phi-3-mini-4k-instruct",
               conda_env="phi4-env", dtype="bfloat16", trust_remote_code=True))
