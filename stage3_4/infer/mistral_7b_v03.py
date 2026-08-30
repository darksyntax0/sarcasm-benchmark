#!/usr/bin/env python3
"""Stage 3/4: Mistral-7B-Instruct-v0.3 (env: voxtral). No system role in Mistral's template;
we already put the full task in the user turn, so no special handling needed."""
from _common_text import run_model
run_model(dict(model_key="mistral_7b_v03", model_id="mistralai/Mistral-7B-Instruct-v0.3",
               conda_env="voxtral", dtype="bfloat16"))
