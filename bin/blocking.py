# blocking.py
from __future__ import annotations  # allows forward-ref type hints without quotes (Py3.8+)

from collections import defaultdict
from typing import DefaultDict, Dict, List, Set

# absolute import if drugmatch is your package root
from models import CanonRow

def build_indexes(esoa_rows: list[CanonRow]):
    idx = {
        "ING_KEY": defaultdict(set),
        "ATC_L4": defaultdict(set),
        "ATC_L5": defaultdict(set),
        "FORM_ROUTE": defaultdict(set),
        "STRENGTH": defaultdict(set),
        "PREFIX": defaultdict(set),
    }
    for i,e in enumerate(esoa_rows):
        idx["ING_KEY"][e.canon_ingredients_key].add(i)
        if e.canon_atc_l4: idx["ATC_L4"][e.canon_atc_l4].add(i)
        if e.canon_atc_code: idx["ATC_L5"][e.canon_atc_code].add(i)
        if e.canon_form_family and e.canon_route:
            idx["FORM_ROUTE"][f"{e.canon_form_family}_{e.canon_route}"].add(i)
        if e.canon_strength and e.canon_strength.bucket:
            idx["STRENGTH"][e.canon_strength.bucket].add(i)
        pref = e.canon_ingredients_key.split("+")[0][:10] if e.canon_ingredients_key else ""
        if pref: idx["PREFIX"][pref].add(i)
    return idx

def candidates_for(p: CanonRow, idx) -> list[int]:
    C = set()
    if p.canon_ingredients_key: C |= idx["ING_KEY"].get(p.canon_ingredients_key, set())
    if p.canon_atc_code:        C |= idx["ATC_L5"].get(p.canon_atc_code, set())
    if p.canon_atc_l4:          C |= idx["ATC_L4"].get(p.canon_atc_l4, set())
    if p.canon_form_family and p.canon_route:
        C |= idx["FORM_ROUTE"].get(f"{p.canon_form_family}_{p.canon_route}", set())
    if p.canon_strength and p.canon_strength.bucket:
        C |= idx["STRENGTH"].get(p.canon_strength.bucket, set())
    if p.canon_ingredients_key:
        pref = p.canon_ingredients_key.split("+")[0][:10]
        C |= idx["PREFIX"].get(pref, set())
    return list(C)