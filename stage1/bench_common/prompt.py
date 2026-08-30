"""Stage 1 prompt construction, shared verbatim across all 5 model inference scripts so every
model is judged on the same instruction text (only the audio-loading mechanics differ per model)."""
from .constants import VALID_CATEGORIES

_CATEGORY_LIST = "\n".join(f"- {c}" for c in VALID_CATEGORIES)

_BASE_TEMPLATE = """You are a sarcasm detection expert analyzing speech.

TASK: Listen to the audio and classify the speaker's statement into exactly ONE category.

STATEMENT (transcript): {target_text}

CATEGORIES:
{category_list}
- OTHER: None of the above categories apply

Respond with a single valid JSON object with exactly these keys: predicted_category, confidence, explanation, reasoning_steps.

STRICT JSON RULES:
- Use DOUBLE QUOTES for all keys and string values. Never use single quotes or Python dict syntax.
- Do not use an apostrophe/single-quote character inside string values (e.g. write "the speaker is done" not "the speaker's done") since it can break parsing.
- Keep "explanation" and each item in "reasoning_steps" to ONE short plain sentence, no embedded quotation marks.

FORMAT TEMPLATE (this shows JSON syntax only -- every value below is a placeholder, not a real answer. Do not copy these placeholder words. Replace each one with your own analysis of the actual audio and transcript above):
{{"predicted_category": "<one of the category names above>", "confidence": <a number between 0.0 and 1.0>, "explanation": "<your one-sentence explanation, double quotes only, no apostrophes>", "reasoning_steps": ["<your first reasoning step>", "<your second reasoning step>"]}}
"""

_FINAL_JSON_SUFFIX = """
After your response, output exactly one line starting with "FINAL_JSON:" followed by a single-line JSON object with exactly the keys: predicted_category, confidence, explanation, reasoning_steps.
"""


def build_stage1_prompt(target_text: str, use_final_json_marker: bool = False) -> str:
    """Builds the Stage 1 instruction text: states the transcript, lists the 6 valid categories
    verbatim (uppercase, exact strings) plus OTHER, and asks for a trailing JSON object with
    predicted_category/confidence/explanation/reasoning_steps.

    If use_final_json_marker, appends the Audio-Reasoner-specific instruction to also emit a
    FINAL_JSON: marker line, since that model's fixed <THINK>/<RESPONSE> structure won't
    otherwise reliably surface bare JSON.
    """
    text = _BASE_TEMPLATE.format(target_text=target_text, category_list=_CATEGORY_LIST)
    if use_final_json_marker:
        text += _FINAL_JSON_SUFFIX
    return text
