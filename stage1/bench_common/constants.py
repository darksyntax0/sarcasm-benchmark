VALID_CATEGORIES = (
    "SARCASM",
    "IRONY",
    "SATIRE",
    "OVERSTATEMENT",
    "UNDERSTATEMENT",
    "RHETORICAL_QUESTION",
)

REQUIRED_GT_KEYS = ("category", "sarcasm_source", "sarcasm_target", "explanation", "sample_id")
REQUIRED_PRED_KEYS = ("predicted_category", "confidence", "explanation", "reasoning_steps")

FINAL_JSON_MARKER = "FINAL_JSON:"
