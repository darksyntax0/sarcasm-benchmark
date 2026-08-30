import difflib
CATS=("SARCASM","IRONY","SATIRE","OVERSTATEMENT","UNDERSTATEMENT","RHETORICAL_QUESTION")
INVALID="__INVALID__"
MORPH={"SARCASTIC":"SARCASM","IRONIC":"IRONY","IRONIC_STATEMENT":"IRONY",
       "SATIRICAL":"SATIRE","RHETORICAL":"RHETORICAL_QUESTION"}
def normalize(v):
    """exact -> morphological variant -> unambiguous close typo -> INVALID.
    Multi-label hedges and out-of-inventory concepts are never mapped (scored incorrect)."""
    if not v or not isinstance(v,str): return INVALID
    x=v.strip().upper().replace(" ","_").replace("-","_").strip("_.")
    if x in CATS: return x
    if x in MORPH: return MORPH[x]
    if any(t in x for t in (",","/"," OR ","_OR_")): return INVALID   # hedged multi-label
    m=difflib.get_close_matches(x, CATS, n=2, cutoff=0.85)
    if len(m)==1: return m[0]
    if len(m)==2 and difflib.SequenceMatcher(None,x,m[0]).ratio() > difflib.SequenceMatcher(None,x,m[1]).ratio()+0.05:
        return m[0]
    return INVALID
