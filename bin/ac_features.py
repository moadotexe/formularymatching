#!/usr/bin/env python3
# ac_features.py
import argparse, re
from dataclasses import dataclass
from collections import deque, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from normalize import (
    normalize_headers, ensure_unique_columns, pick_col, normalize_text
)

# ---- tiny AC engine ----------------------------------------------------------
@dataclass(frozen=True)
class Hit:
    start: int
    end: int
    pat: str
    payload: Dict[str, Any]

class Aho:
    def __init__(self, patterns_with_payloads: Iterable[Tuple[str, Dict[str,Any]]]):
        self.goto: List[Dict[str,int]] = [dict()]   # root = 0
        self.out:  List[List[int]]     = [[]]
        self.fail: List[int]           = [0]
        self.pats: List[str]           = []
        self.payloads: List[Dict[str,Any]] = []
        for p, payload in patterns_with_payloads:
            if not p: continue
            self._insert(p)
            self.payloads.append(payload)
        self._build()

    def _insert(self, pat: str) -> None:
        s = 0
        for ch in pat:
            if ch not in self.goto[s]:
                self.goto[s][ch] = len(self.goto)
                self.goto.append(dict()); self.out.append([]); self.fail.append(0)
            s = self.goto[s][ch]
        self.out[s].append(len(self.pats))
        self.pats.append(pat)

    def _build(self) -> None:
        q = deque()
        for _, s in self.goto[0].items():
            self.fail[s] = 0
            q.append(s)
        while q:
            r = q.popleft()
            for ch, s in self.goto[r].items():
                q.append(s)
                f = self.fail[r]
                while f and ch not in self.goto[f]:
                    f = self.fail[f]
                self.fail[s] = self.goto[f].get(ch, 0)
                self.out[s].extend(self.out[self.fail[s]])

    def finditer(self, text: str) -> Iterable[Hit]:
        s = 0
        for i, ch in enumerate(text):
            while s and ch not in self.goto[s]:
                s = self.fail[s]
            s = self.goto[s].get(ch, 0)
            if self.out[s]:
                for pid in self.out[s]:
                    pat = self.pats[pid]
                    yield Hit(i - len(pat) + 1, i + 1, pat, self.payloads[pid])

# ---- helpers -----------------------------------------------------------------
_WORD = set("abcdefghijklmnopqrstuvwxyz0123456789")

def _word_boundary_ok(s: str, l: int, r: int) -> bool:
    left_ok  = (l == 0) or (s[l-1] not in _WORD)
    right_ok = (r == len(s)) or (s[r:r+1] not in _WORD)
    return left_ok and right_ok

def _longest_non_overlapping(hits):
    # hits: list[(l,r,pat,payload)]
    hits = sorted(hits, key=lambda t: (t[0], -(t[1]-t[0])))
    out, cur_end = [], -1
    for h in hits:
        if h[0] >= cur_end:
            out.append(h)
            cur_end = h[1]
    return out

def _variants(s: str) -> List[str]:
    """Split multi-valued cells; normalize each piece."""
    if not isinstance(s, str): return []
    parts = re.split(r"[;/|,]+", s) if s else []
    parts = [normalize_text(p.strip()) for p in parts if p.strip()]
    return [p for p in parts if p]

# ---- dictionary builder -------------------------------------------------------
def build_patterns(pnf: pd.DataFrame, fda: pd.DataFrame, who: pd.DataFrame,
                   pnf_key_col: str,
                   pnf_name_cols: List[str]) -> List[Tuple[str, Dict[str,Any]]]:
    pnf = ensure_unique_columns(normalize_headers(pnf))
    fda = ensure_unique_columns(normalize_headers(fda))
    who = ensure_unique_columns(normalize_headers(who))

    pnf_key = pick_col(pnf, [pnf_key_col], must=True, label="PNF primary key")

    # PNF display/generic columns (optional)
    name_cols = [c for c in pnf_name_cols if c in pnf.columns]

    # FDA columns
    fda_brand = pick_col(fda, ["BRAND_NAME","BRAND","PRODUCT_NAME"], must=False)
    fda_gen   = pick_col(fda, ["GENERIC_NAME","GENERIC","ACTIVE_INGREDIENT"], must=False)

    # WHO ATC columns
    who_code  = pick_col(who, ["ATC_CODE","ATC","ATC CODE"], must=False)
    who_name  = pick_col(who, ["ATC_NAME","ATC NAME","DESCRIPTION"], must=False)

    patterns = {}

    def add(term: str, payload: Dict[str,Any]):
        n = normalize_text(term)
        if not n: return
        # keep first payload per term (or merge if you prefer)
        patterns.setdefault(n, payload)

    # PNF names → PNF key
    for _, r in pnf.iterrows():
        key = str(r[pnf_key])
        for c in name_cols:
            v = r.get(c, "")
            for tok in _variants(str(v)):
                add(tok, {"source":"PNF","pnf_key": key})

    # FDA brand/generic
    if fda_brand or fda_gen:
        for _, r in fda.iterrows():
            b = r.get(fda_brand, "")
            g = r.get(fda_gen, "")
            nb, ng = normalize_text(str(b)), normalize_text(str(g))
            if nb: add(nb, {"source":"FDA","brand": nb, "generic": ng})
            if ng: add(ng, {"source":"FDA","brand": nb, "generic": ng})

    # WHO ATC names + codes
    if who_name or who_code:
        for _, r in who.iterrows():
            nm = r.get(who_name, "")
            cd = r.get(who_code, "")
            nn = normalize_text(str(nm))
            if nn: add(nn, {"source":"WHO","atc_name": nn, "atc_code": str(cd) if pd.notna(cd) else ""})
            if pd.notna(cd):
                add(str(cd), {"source":"WHO","atc_code": str(cd)})

    return [(t, payload) for t, payload in patterns.items()]

# ---- AC feature generation ----------------------------------------------------
def add_ac_features(esoa: pd.DataFrame, patterns: List[Tuple[str,Dict[str,Any]]],
                    desc_candidates: List[str]) -> pd.DataFrame:
    esoa = ensure_unique_columns(normalize_headers(esoa))
    desc_col = next((c for c in desc_candidates if c in esoa.columns), None)
    if not desc_col:
        raise KeyError("ac_features: no description column found in eSOA.")

    ac = Aho(patterns)
    rows = []
    for idx, row in esoa.iterrows():
        raw = str(row[desc_col] or "")
        txt = normalize_text(raw)

        tmp = []
        for h in ac.finditer(txt):
            if _word_boundary_ok(txt, h.start, h.end):
                tmp.append((h.start, h.end, h.pat, h.payload))
        pruned = _longest_non_overlapping(tmp)

        terms, pnf_keys, atc_codes = [], [], []
        brand_hit = False; generic_hit = False; atc_hit = False
        for l, r, pat, payload in pruned:
            terms.append(txt[l:r])
            if "pnf_key" in payload:
                pnf_keys.append(payload["pnf_key"])
            if payload.get("atc_code"):
                atc_codes.append(payload["atc_code"])
            if payload.get("brand"):   brand_hit = True
            if payload.get("generic"): generic_hit = True
            if payload.get("atc_code"): atc_hit = True

        rows.append({
            "_row_id": idx,
            "_AC_HAS_MATCH": bool(pruned),
            "_AC_TERMS": "|".join(dict.fromkeys(terms)) if terms else "",
            "_AC_PNF_KEYS": "|".join(dict.fromkeys(pnf_keys)) if pnf_keys else "",
            "_AC_ATC_CODES": "|".join(dict.fromkeys(atc_codes)) if atc_codes else "",
            "_AC_BRAND_HIT": int(brand_hit),
            "_AC_GENERIC_HIT": int(generic_hit),
            "_AC_ATC_HIT": int(atc_hit),
        })

    feat = pd.DataFrame(rows).set_index("_row_id")
    out = esoa.copy()
    out.index.name = "_row_id"
    out = out.join(feat, how="left").reset_index(drop=True)
    out[["_AC_HAS_MATCH","_AC_BRAND_HIT","_AC_GENERIC_HIT","_AC_ATC_HIT"]] = \
        out[["_AC_HAS_MATCH","_AC_BRAND_HIT","_AC_GENERIC_HIT","_AC_ATC_HIT"]].fillna(False).astype(int)
    for c in ["_AC_TERMS","_AC_PNF_KEYS","_AC_ATC_CODES"]:
        if c in out.columns: out[c] = out[c].fillna("")
    return out

def main():
    ap = argparse.ArgumentParser(description="Add Aho–Corasick dictionary features to eSOA.")
    ap.add_argument("--esoa-in", required=True)
    ap.add_argument("--pnf-in", required=True)
    ap.add_argument("--fda-in", required=True)
    ap.add_argument("--who-in", required=True)
    ap.add_argument("--esoa-out", required=True)
    ap.add_argument("--pnf-key", default="ATC Code", help="PNF primary key column name (case/spacing tolerant).")
    ap.add_argument("--pnf-name-cols", nargs="*", default=[], help="PNF columns to expose as terms (e.g. DRUG_NAME GENERIC)")
    ap.add_argument("--desc-candidates", nargs="*", default=[
        "TECHNICAL SPECIFICATIONS","TECHNICAL_SPECIFICATIONS","DESCRIPTION","ITEM DESCRIPTION","SPECIFICATION","SPECIFICATIONS"
    ])
    args = ap.parse_args()

    esoa = pd.read_csv(args.esoa_in, sep=None, engine="python", encoding="utf-8")
    pnf  = pd.read_csv(args.pnf_in,  sep=None, engine="python", encoding="utf-8")
    fda  = pd.read_csv(args.fda_in,  sep=None, engine="python", encoding="utf-8")
    who  = pd.read_csv(args.who_in,  sep=None, engine="python", encoding="utf-8")

    patterns = build_patterns(pnf, fda, who, pnf_key_col=args.pnf_key, pnf_name_cols=args.pnf_name_cols)
    out = add_ac_features(esoa, patterns, desc_candidates=args.desc_candidates)

    Path(args.esoa_out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.esoa_out, index=False)
    print(f"✅ AC features added -> {args.esoa_out}  (rows={len(out):,})")

if __name__ == "__main__":
    main()