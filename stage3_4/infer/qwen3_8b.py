#!/usr/bin/env python3
"""Stage 3/4: Qwen3-8B (env: qwen2-audio-env).
Note: the requested 'Qwen3-8B-Instruct' resolves to Qwen/Qwen3-8B (Qwen3 has no separate
-Instruct repo; the base is instruction-tuned). enable_thinking=False for clean, comparable
JSON output (default thinking mode emits <think>...</think>)."""
from _common_text import run_model
run_model(dict(model_key="qwen3_8b", model_id="Qwen/Qwen3-8B",
               conda_env="qwen2-audio-env", dtype="bfloat16",
               template_kwargs={"enable_thinking": False}))
