# atc.py
from models import CanonRow
from typing import List, Optional, Set, Dict
import pandas as pd

class ATCIndex:
    def __init__(self, who_df: pd.DataFrame):
        self.by_l5_routes: Dict[str, Set[str]] = {}
        self.by_name: Dict[str, Set[str]] = {}
        for code, name, route in zip(
            who_df["atc_code"].astype(str).tolist(),
            who_df["atc_name"].astype(str).tolist(),
            who_df["adm_r"].astype(str).tolist(),
        ):
            code = code.strip()
            nm = (name or "").strip().lower()
            rt = (route or "").strip().lower()
            if code:
                if rt:
                    self.by_l5_routes.setdefault(code, set()).add(rt)
                if nm:
                    self.by_name.setdefault(nm, set()).add(code)

def lookup_atc_by_ingredients_and_route(
    ingredients: List[str],
    route: Optional[str],
    who_df: pd.DataFrame,
    idx: Optional[ATCIndex] = None,
) -> List[str]:
    if idx is None:
        idx = ATCIndex(who_df)
    tokens = [(t or "").strip().lower() for t in ingredients if t]
    joined = " ".join(tokens).strip()
    route_c = (route or "").strip().lower()

    cands: Set[str] = set()

    if joined and joined in idx.by_name:
        cands |= idx.by_name[joined]

    if not cands and tokens:
        for code, name in zip(who_df["atc_code"], who_df["atc_name"]):
            nm = (str(name) if name is not None else "").strip().lower()
            if nm and all(t in nm for t in tokens):
                cands.add(str(code))

    if route_c:
        cands = {
            c for c in cands
            if not idx.by_l5_routes.get(c) or route_c in idx.by_l5_routes[c]
        }
    return sorted(cands)

def choose_best(
    codes: List[str],
    who_df: pd.DataFrame,
    preferred_route: Optional[str] = None
) -> Optional[str]:
    if not codes:
        return None
    if preferred_route:
        pr = (preferred_route or "").strip().lower()
        with_pr = []
        for c in codes:
            routes = (
                who_df.loc[who_df["atc_code"] == c, "adm_r"]
                .dropna().astype(str).map(lambda s: s.strip().lower()).unique().tolist()
            )
            if pr in routes:
                with_pr.append(c)
        if with_pr:
            return sorted(with_pr)[0]
    return sorted(codes)[0]

def index_who(who_df):
    # build maps: ingredient -> {route -> [atc_codes]}
    # keep 4th/5th-level helpers
    ...

def infer_atc(crow: CanonRow, who_index) -> CanonRow:
    if crow.source != "ESOA": return crow
    if not crow.canon_ingredients: return crow
    cands = lookup_atc_by_ingredients_and_route(who_index, crow.canon_ingredients, crow.canon_route)
    if cands:
        best = choose_best(cands)  # prefer 5th-level if unique; else 4th
        crow.canon_atc_code = best
        crow.canon_atc_l4 = best[:4]
    return crow