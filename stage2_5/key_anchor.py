"""Key-anchored field extractor for dict-repr model output whose string values contain
unescaped apostrophes (observed failure mode of Qwen2-Audio: 535/536 of its unparseable rows).
Each value is bounded by the NEXT known key rather than by quote matching, which is what makes
it robust where json.loads, ast.literal_eval, and quote-based regex all fail.
"""
import re, json, ast
KEYS = ("is_sarcastic","predicted_category","sarcasm_source","sarcasm_target",
        "confidence","explanation","reasoning_steps")

def _strip_val(v):
    v = v.strip().rstrip(',').strip()
    # peel escaped-double, double, or single quotes
    for a,b in (('\\"','\\"'), ('"','"'), ("'","'")):
        if v.startswith(a) and v.endswith(b) and len(v) > len(a)+len(b)-1:
            return v[len(a):-len(b)].strip()
    if v.startswith('\\"'): return v[2:].strip()
    if v[:1] in "\"'": return v[1:].strip()
    return v

def _to_list(v):
    v = v.strip().rstrip(',').strip()
    if v.startswith('['): v = v[1:]
    if v.endswith(']'):   v = v[:-1]
    # split on quote-comma-quote boundaries; tolerant of nested apostrophes
    parts = re.split(r"""(?:'|\\?")\s*,\s*(?:'|\\?")""", v)
    out=[_strip_val(p) for p in parts]
    return [p for p in out if p]

def key_anchored_extract(text):
    """Extract the 7 prediction fields by anchoring on key names and bounding each value by
    the NEXT key, which is robust to unescaped apostrophes inside single-quoted dict-repr."""
    # positions of each key anchor:  'key':   "key":   key:
    anchors=[]
    for k in KEYS:
        m = re.search(rf"""(?:['"]|\\")?{k}(?:['"]|\\")?\s*:""", text)
        if m: anchors.append((m.start(), m.end(), k))
    if not anchors: return None
    anchors.sort()
    out={}
    for i,(s,e,k) in enumerate(anchors):
        end = anchors[i+1][0] if i+1 < len(anchors) else len(text)
        raw = text[e:end]
        raw = re.sub(r"[\s,]*$", "", raw)
        raw = re.sub(r"\}\s*$", "", raw).strip()
        if k == "reasoning_steps":
            out[k] = _to_list(raw) or None
        elif k == "is_sarcastic":
            t = raw.strip().strip("'\"\\ ").lower()
            out[k] = True if t.startswith("true") else (False if t.startswith("false") else None)
        elif k == "confidence":
            m = re.search(r"[0-9]*\.?[0-9]+", raw); out[k] = float(m.group()) if m else None
        else:
            val=_strip_val(raw)
            out[k] = val if val and val.lower() not in ("none","null") else None
    return out if out.get("predicted_category") else None
