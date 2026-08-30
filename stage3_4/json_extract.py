# VENDORED (read-only copy) from benchmark_framework/bench_common/json_extract.py on 2026-08-21,
# so Stage 2 is self-contained and independently reproducible. Do NOT edit Stage 1 by editing this.
"""Robust JSON extraction from raw model text output.

Models are prompted to end their response with a JSON object, but not all of them comply
cleanly -- Audio-Reasoner in particular wraps everything in <THINK>/<RESPONSE> tags and needs
a marker-based fast path. This module never raises; callers get (dict, None) on success or
(None, error_message) on failure.
"""
import ast
import json


def _close_unbalanced_brackets(candidate: str):
    """Scans candidate (string/escape aware), rebuilding it while tracking a stack of open {/[
    characters outside of strings. Handles two observed real failure modes from Audio-Flamingo-3:
    (a) a list's ']' dropped right before an otherwise-correct '}' -- e.g.
    '..."item2"}' where '..."item2"]}' was meant -- fixed by inserting the missing closer(s) at
    the point of mismatch, not just at the end of the string; (b) a properly-closed list but the
    outer '}' missing entirely -- fixed by appending remaining closers once the scan ends. A
    closer with no matching opener at all (stack empty) is treated as unrepairable and bails out.
    Returns None if unterminated mid-string, unrepairable, or nothing needed fixing."""
    stack = []
    in_string = False
    escape = False
    out = []
    closers = {"{": "}", "[": "]"}
    openers = {"}": "{", "]": "["}
    for ch in candidate:
        if escape:
            escape = False
            out.append(ch)
            continue
        if ch == "\\":
            escape = True
            out.append(ch)
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string:
            out.append(ch)
            continue
        if ch in "{[":
            stack.append(ch)
            out.append(ch)
        elif ch in "}]":
            wanted_open = openers[ch]
            if not stack:
                return None
            while stack and stack[-1] != wanted_open:
                missing = stack.pop()
                out.append(closers[missing])
            if not stack:
                return None
            stack.pop()
            out.append(ch)
        else:
            out.append(ch)
    if in_string:
        return None
    for b in reversed(stack):
        out.append(closers[b])
    repaired = "".join(out)
    return repaired if repaired != candidate else None


def _attempt(candidate: str):
    # An empty dict is never a useful result in this codebase (every real prediction has at
    # least a predicted_category key, even if its value is null) -- rejecting {} here matters
    # specifically for bracket-repair: a truncated fragment like the bare "{" left over when a
    # marker line's real JSON content spans subsequent lines would otherwise "repair" into a
    # trivially-valid empty {} and short-circuit before the caller falls through to a scan that
    # would find the real, complete object elsewhere in the text.
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict) and parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    try:
        parsed = ast.literal_eval(candidate)
        if isinstance(parsed, dict) and parsed:
            return parsed
    except (ValueError, SyntaxError):
        pass
    return None


def _try_parse_dict(candidate: str):
    """Try strict JSON, then Python dict-literal syntax (ast.literal_eval) -- models sometimes
    emit single-quoted dict-repr-style output instead of JSON. As a last resort, retry both after
    stripping spurious backslash-escapes before quote characters (observed real failure mode:
    some models double-escape quotes as if already inside a JSON string, e.g. \\" appearing where
    a plain " was intended). This repair only runs after the unmodified text has already failed
    both parses, so it cannot corrupt output that would otherwise have parsed correctly -- real
    backslash+quote sequences are vanishingly rare in genuine natural-language explanations.
    literal_eval is safe (no code execution, only literal containers/strings/numbers/etc.) and
    simply raises on truly malformed text, which the caller treats as a normal parse failure."""
    parsed = _attempt(candidate)
    if parsed is not None:
        return parsed

    unescaped = None
    if "\\" in candidate:
        unescaped = candidate.replace('\\"', '"').replace("\\'", "'")
        parsed = _attempt(unescaped)
        if parsed is not None:
            return parsed

    # Repair: a key that lost its OPENING double-quote, e.g. `..."explanation": "x",reasoning_steps": [`
    # (observed real Audio-Flamingo-3 failure). Insert the missing quote before a bareword key that
    # sits right after `{` or `,` and is followed by `":`. Conservative: only touches barewords in
    # key position, never string contents (a comma inside a JSON string is preceded by a value, not
    # by `{`/`,` + word + `":`). Chain with bracket-repair since these often co-occur.
    import re as _re
    key_fixed = _re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)"\s*:', r'\1"\2":', candidate)
    variants = [candidate, unescaped, key_fixed]
    if unescaped is not None:
        variants.append(_re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)"\s*:', r'\1"\2":', unescaped))

    # Last resort: the model forgot to close one or more trailing brackets (observed real
    # failure mode -- see _close_unbalanced_brackets docstring). Try on each variant, and also on
    # the bracket-closed form of each, since a variant may need both key-quote and bracket repair.
    for base in variants:
        if base is None:
            continue
        parsed = _attempt(base)
        if parsed is not None:
            return parsed
        repaired = _close_unbalanced_brackets(base)
        if repaired is not None:
            parsed = _attempt(repaired)
            if parsed is not None:
                return parsed

    return None


def find_balanced_json_blocks(text: str) -> list:
    """Stack-based, string/escape-aware scan for top-level balanced {...} substrings.

    Escape-suppression is applied uniformly regardless of current in/out-of-string state: a
    backslash always makes the next character literal and never toggles string state. This
    matters for a real observed failure mode where a model spuriously backslash-escapes quotes
    that should have been plain string delimiters (e.g. `\\"text\\"` where `"text"` was meant) --
    without uniform escape handling, the first such escaped quote gets treated as a real string
    *open* (only exit-time escapes were being honored), permanently desyncing in_string tracking
    for the rest of the text and hiding the real closing brace inside a perceived unterminated
    string.
    """
    blocks = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    blocks.append(text[start : i + 1])
                    start = None
    return blocks


def extract_json(text: str, prefer_marker: str = None):
    """Try a marker fast path first (e.g. 'FINAL_JSON:' on its own line), then fall back to
    balanced-brace scanning of the whole text, trying the LAST block first since a trailing
    JSON object is the intended answer in every prompt template used here.

    Returns (dict_or_None, error_message_or_None). Never raises.
    """
    if not text:
        return None, "empty text"

    if prefer_marker:
        for line in text.splitlines():
            idx = line.find(prefer_marker)
            if idx != -1:
                candidate = line[idx + len(prefer_marker) :].strip()
                parsed = _try_parse_dict(candidate)
                if parsed is not None:
                    return parsed, None
                # Marker found but same-line parse failed -- the JSON may span multiple lines
                # after the marker. Fall through to whole-text brace scanning below rather than
                # giving up immediately.

    blocks = find_balanced_json_blocks(text)
    for candidate in reversed(blocks):
        parsed = _try_parse_dict(candidate)
        if parsed is not None:
            return parsed, None

    # No balanced block at all (or none parsed) -- the model may have dropped the FINAL closing
    # brace entirely, which find_balanced_json_blocks can never detect since it only recognizes
    # already-balanced spans. Fall back to "everything from the last '{' to end of text" and let
    # _try_parse_dict's bracket-repair attempt supply the missing closer(s).
    last_open = text.rfind("{")
    if last_open != -1:
        tail_candidate = text[last_open:].strip()
        parsed = _try_parse_dict(tail_candidate)
        if parsed is not None:
            return parsed, None

    if not blocks:
        return None, "no balanced JSON object found in output"
    return None, f"found {len(blocks)} brace block(s) but none parsed as a JSON object"
