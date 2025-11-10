# normalize.py
import regex as re
import unicodedata

WS = re.compile(r"\s+")
def clean(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.strip().lower()
    s = WS.sub(" ", s)
    return s

def apply_synonyms(token: str, synonym_map: dict[str,str]) -> str:
    return synonym_map.get(token, token)

def strip_salt(token: str) -> str:
    # simplistic; extend with a salt table
    for salt in (" hydrochloride"," hcl"," sodium"," sulfate"," sulphate"," phosphate"," acetate"," tartrate"):
        if token.endswith(salt): return token.replace(salt,"")
    return token

def canon_ingredient_tokens(s: str, synonym_map: dict[str,str]) -> list[str]:
    # split on +, /, commas, " and ", "&", " with "
    parts = re.split(r"\s*(?:\+|/|,| and | & | with )\s*", s)
    toks = []
    for p in parts:
        t = strip_salt(p.strip())
        t = apply_synonyms(t, synonym_map)
        if t: toks.append(t)
    return sorted(set(toks))