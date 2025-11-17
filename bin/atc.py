#!/usr/bin/env python3
# atc.py

from __future__ import annotations

from typing import List, Optional, Set, Dict, TYPE_CHECKING
import pandas as pd

# Only import CanonRow for type checking to avoid circular imports at runtime
if TYPE_CHECKING:
    from models import CanonRow


class ATCIndex:
    """
    Holds the WHO ATC DataFrame plus two helper maps:
      - by_l5_routes: atc_code -> set of canonical routes (lowercased)
      - by_name: normalized atc_name -> set of atc_code
    """
    def __init__(self, who_df: pd.DataFrame):
        self.df: pd.DataFrame = who_df
        self.by_l5_routes: Dict[str, Set[str]] = {}
        self.by_name: Dict[str, Set[str]] = {}

        for code, name, route in zip(
            who_df["atc_code"].astype(str).tolist(),
            who_df["atc_name"].astype(str).tolist(),
            who_df["adm_r"].astype(str).tolist(),
        ):
            code = (code or "").strip()
            nm = (name or "").strip().lower()
            rt = (route or "").strip().lower()
            if not code:
                continue

            if rt:
                self.by_l5_routes.setdefault(code, set()).add(rt)
            if nm:
                self.by_name.setdefault(nm, set()).add(code)


def lookup_atc_by_ingredients_and_route(
    ingredients: List[str],
    route: Optional[str],
    idx: ATCIndex,
) -> List[str]:
    """
    Heuristic:
      1) exact name match (all ingredients joined) against WHO atc_name
      2) else subset match: all ingredient tokens appear in WHO atc_name
      3) filter by route if provided
    Uses the prebuilt ATCIndex (no need to pass who_df separately).
    """
    tokens = [(t or "").strip().lower() for t in ingredients if t]
    joined = " ".join(tokens).strip()
    route_c = (route or "").strip().lower()

    cands: Set[str] = set()

    # 1) exact normalized name
    if joined and joined in idx.by_name:
        cands |= idx.by_name[joined]

    # 2) subset match if nothing exact matched
    if not cands and tokens:
        for code, name in zip(idx.df["atc_code"], idx.df["atc_name"]):
            nm = (str(name) if name is not None else "").strip().lower()
            if nm and all(t in nm for t in tokens):
                cands.add(str(code))

    # 3) route filter: keep codes that either have no route constraint or include our route
    if route_c:
        cands = {
            c for c in cands
            if not idx.by_l5_routes.get(c) or route_c in idx.by_l5_routes[c]
        }

    return sorted(cands)


def choose_best(
    codes: List[str],
    idx: ATCIndex,
    preferred_route: Optional[str] = None,
) -> Optional[str]:
    """
    Tie-break:
      - prefer codes whose WHO routes include preferred_route (if given)
      - else return lexicographically smallest code.
    """
    if not codes:
        return None

    if preferred_route:
        pr = (preferred_route or "").strip().lower()
        with_pr: List[str] = []
        for c in codes:
            routes = (
                idx.df.loc[idx.df["atc_code"] == c, "adm_r"]
                .dropna()
                .astype(str)
                .map(lambda s: s.strip().lower())
                .unique()
                .tolist()
            )
            if pr in routes:
                with_pr.append(c)
        if with_pr:
            return sorted(with_pr)[0]

    return sorted(codes)[0]


def index_who(who_df: pd.DataFrame) -> ATCIndex:
    """
    Convenience wrapper: build an ATCIndex from a WHO ATC DataFrame.
    """
    return ATCIndex(who_df)


def infer_atc(crow: "CanonRow", who_index: ATCIndex) -> "CanonRow":
    """
    Enrich a CanonRow with ATC code and L4 using WHO index.
    Only acts on ESOA rows with ingredients.
    """
    if crow.source != "ESOA":
        return crow
    if not crow.canon_ingredients:
        return crow

    # NOTE: we now pass arguments in the correct order
    cands = lookup_atc_by_ingredients_and_route(
        crow.canon_ingredients,
        crow.canon_route,
        who_index,
    )

    if cands:
        best = choose_best(cands, who_index, preferred_route=crow.canon_route)
        if best:
            crow.canon_atc_code = best
            crow.canon_atc_l4 = best[:4]

    return crow
