"""Stage 3 (target-only) and Stage 4 (context+target) text prompts + output schema (stdlib only).

Semantically aligned with Stage 1/2: SAME 7-field JSON output schema so cross-stage evaluation is
possible. The ONLY difference from Stage 1/2 is modality (text instead of audio) and, between
Stage 3 vs 4, whether the CONTEXT utterance is present.

FAIR-COMPARISON GUARANTEES:
  * Stage 3 prompt contains ONLY the target transcription -- never context.
  * Stage 4 prompt contains context + target transcription -- never audio, never ground truth.
  * Task wording, categories, and output requirements are identical across stages.
"""
VALID_CATEGORIES = ("SARCASM", "IRONY", "SATIRE", "OVERSTATEMENT", "UNDERSTATEMENT", "RHETORICAL_QUESTION")
REQUIRED_PRED_KEYS = ("is_sarcastic", "predicted_category", "sarcasm_source",
                      "sarcasm_target", "confidence", "explanation", "reasoning_steps")
FINAL_JSON_MARKER = "FINAL_JSON:"

_SCHEMA_BLOCK = """Decide:
1. Whether the TARGET is sarcastic (true/false).
2. The sarcasm category: one of {cats}, or OTHER.
3. Who is being sarcastic (the sarcasm source / speaker).
4. Who or what is being targeted (the sarcasm target).
5. A short explanation of your reasoning.

Respond with a single valid JSON object with exactly these keys: is_sarcastic, predicted_category, sarcasm_source, sarcasm_target, confidence, explanation, reasoning_steps.

STRICT JSON RULES:
- Use DOUBLE QUOTES for all keys and string values. Never single quotes or Python dict syntax.
- Do not put an apostrophe/single-quote inside string values.
- Keep "explanation" and each "reasoning_steps" item to one short plain sentence, no embedded quotation marks.

FORMAT TEMPLATE (placeholders only -- replace every value with your own analysis; do NOT copy these words):
{{"is_sarcastic": <true or false>, "predicted_category": "<one category name above, or OTHER>", "sarcasm_source": "<who is being sarcastic>", "sarcasm_target": "<who or what is targeted>", "confidence": <number 0.0-1.0>, "explanation": "<one sentence>", "reasoning_steps": ["<step one>", "<step two>"]}}"""

_STAGE3 = """You are analyzing a spoken utterance (given as its transcription) for sarcasm.

TARGET UTTERANCE:
{target_text}

Analyze the TARGET utterance for sarcasm. {schema}"""

_STAGE4 = """You are analyzing a spoken conversation (given as transcriptions) for sarcasm.

CONTEXT UTTERANCE (what was said just before):
{context_text}

TARGET UTTERANCE (the line to analyze):
{target_text}

Analyze the TARGET utterance using the provided conversational CONTEXT. {schema}"""

_MARKER_SUFFIX = ('\nAfter your response, output exactly one line starting with "FINAL_JSON:" '
                  "followed by a single-line JSON object with exactly those keys.")


def _schema():
    return _SCHEMA_BLOCK.format(cats=", ".join(VALID_CATEGORIES))


def build_stage3_prompt(target_text: str, use_final_json_marker: bool = False) -> str:
    t = _STAGE3.format(target_text=target_text, schema=_schema())
    return t + (_MARKER_SUFFIX if use_final_json_marker else "")


def build_stage4_prompt(context_text: str, target_text: str, use_final_json_marker: bool = False) -> str:
    t = _STAGE4.format(context_text=context_text, target_text=target_text, schema=_schema())
    return t + (_MARKER_SUFFIX if use_final_json_marker else "")


def build_prompt(stage: int, context_text: str, target_text: str, use_final_json_marker: bool = False) -> str:
    if stage == 3:
        return build_stage3_prompt(target_text, use_final_json_marker)
    if stage == 4:
        return build_stage4_prompt(context_text, target_text, use_final_json_marker)
    raise ValueError(f"stage must be 3 or 4, got {stage}")
