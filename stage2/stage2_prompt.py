"""Stage 2 standardized task prompt + output schema (stdlib only).

Semantic task is identical across all 5 models (only each model's low-level chat syntax differs,
handled in each model's inference script). The output schema PRESERVES Stage 1's prediction
fields (predicted_category, confidence, explanation, reasoning_steps) and ADDS the Stage-2 /
ground-truth fields (is_sarcastic, sarcasm_source, sarcasm_target) so predictions line up 1:1
with sarcasm_analysis.json (category / sarcasm_source / sarcasm_target / explanation). No new
ground-truth schema is invented.
"""

# Six categories are the Stage-1 canonical set; OTHER covers anything else.
VALID_CATEGORIES = ("SARCASM", "IRONY", "SATIRE", "OVERSTATEMENT", "UNDERSTATEMENT", "RHETORICAL_QUESTION")
_CATEGORY_LIST = "\n".join(f"- {c}" for c in VALID_CATEGORIES)

# Stage-2 prediction keys (superset of Stage-1's, plus the fields GT carries).
REQUIRED_PRED_KEYS = ("is_sarcastic", "predicted_category", "sarcasm_source",
                      "sarcasm_target", "confidence", "explanation", "reasoning_steps")

FINAL_JSON_MARKER = "FINAL_JSON:"

_TASK = """You are analyzing a spoken conversation for sarcasm.

You are given a CONTEXT utterance (what was said just before) and a TARGET utterance (the line to analyze). {modality_note} Use the CONTEXT to interpret the TARGET.

CONTEXT transcription: {context_text}
TARGET transcription: {target_text}

Analyze the TARGET utterance using the context. Decide:
1. Whether the TARGET is sarcastic (true/false).
2. The sarcasm category: one of {category_list_inline}, or OTHER.
3. Who is being sarcastic (the sarcasm source / speaker).
4. Who or what is being targeted (the sarcasm target).
5. A short explanation of your reasoning.

Respond with a single valid JSON object with exactly these keys: is_sarcastic, predicted_category, sarcasm_source, sarcasm_target, confidence, explanation, reasoning_steps.

STRICT JSON RULES:
- Use DOUBLE QUOTES for all keys and string values. Never single quotes or Python dict syntax.
- Do not put an apostrophe/single-quote inside string values.
- Keep "explanation" and each "reasoning_steps" item to one short plain sentence, no embedded quotation marks.

FORMAT TEMPLATE (placeholders only -- replace every value with your own analysis of the actual audio and transcripts; do NOT copy these words):
{{"is_sarcastic": <true or false>, "predicted_category": "<one category name above, or OTHER>", "sarcasm_source": "<who is being sarcastic>", "sarcasm_target": "<who or what is targeted>", "confidence": <number 0.0-1.0>, "explanation": "<one sentence>", "reasoning_steps": ["<step one>", "<step two>"]}}
"""

_MARKER_SUFFIX = ('\nAfter your response, output exactly one line starting with "FINAL_JSON:" '
                  "followed by a single-line JSON object with exactly those keys.")


_MODALITY_BOTH = "For each you receive BOTH the audio and its transcription."
_MODALITY_TARGET_ONLY = ("You receive the TARGET as audio + transcription, and the CONTEXT as "
                         "transcription text only (no context audio is provided to this model).")


def build_task_text(context_text: str, target_text: str, use_final_json_marker: bool = False,
                    context_audio_available: bool = True) -> str:
    """The shared semantic task text (transcripts embedded). Each model script wraps this with
    its own audio-placeholder / chat-template mechanics. context_audio_available=False is used
    only by the Audio-Flamingo-3 reduced-modality variant (target audio + text context), so the
    prompt states the true modality rather than implying a context clip the model never heard."""
    t = _TASK.format(context_text=context_text, target_text=target_text,
                     category_list_inline=", ".join(VALID_CATEGORIES),
                     modality_note=_MODALITY_BOTH if context_audio_available else _MODALITY_TARGET_ONLY)
    if use_final_json_marker:
        t += _MARKER_SUFFIX
    return t
