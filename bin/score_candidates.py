#!/usr/bin/env python3
import argparse
from enum import auto
from pathlib import Path
import pandas as pd
import math
import re
from typing import List, Tuple

# ----------------------------
# Helpers: unit parsing / route+form similarity
# ----------------------------

UNIT_RE = re.compile(r"(?P<qty>\d+(?:\.\d+)?)(?P<uom>MCG|MG|G|IU|ML|%|MMOL)", re.I)
RATIO_RE = re.compile(r"(?P<num>\d+(?:\.\d+)?)(?P<unum>MCG|MG|G|IU|ML|MMOL)\s*/\s*(?P<den>\d+(?:\.\d+)?)(?P<dnum>ML|G|L)", re.I)

# Very light conversion table to mg (mass) or IU (kept as IU), and mg/mL for ratios
TO_MG = {"MCG": 0.001, "MG": 1.0, "G": 1000.0}
DEN_TO_ML = {"ML": 1.0, "L": 1000.0, "G": 1.0}  # crude; treats g like mL for solids -> heuristic only

ROUTE_FAMILY = {
    "PO": "ENTERAL", "SL": "ENTERAL",
    "IV": "PARENTERAL", "IM": "PARENTERAL", "SC": "PARENTERAL",
    "TOP": "TOPICAL",
    "INH": "RESP",
    "IN": "RESP",
}

FORM_SYNONYM = {
    "TABLET": {"TABLET"},
    "CAPSULE": {"CAPSULE"},
    "SYRUP": {"SYRUP", "SOLUTION", "SUSPENSION"},  # liquid oral family (heuristic)
    "SOLUTION": {"SOLUTION"},
    "SUSPENSION": {"SUSPENSION"},
    "INHALER": {"INHALER", "SPRAY"},
    "CREAM": {"CREAM", "GEL", "OINTMENT"},
    # extend as needed
}

def _first(lst: List[str]) -> str:
    return lst[0] if isinstance(lst, list) and lst else ""

def parse_strength_value(s: str) -> Tuple[str, float]:
    """
    Return a numeric canonical value and a 'kind':
      - ('mg', value_in_mg) for mass
      - ('mg_per_ml', value_in_mg_per_ml) for ratios
      - ('iu', value_in_IU) for IU
      - ('other', 0.0) if unhandled
    Uses the FIRST recognizable token only (heuristic).
    """
    if not isinstance(s, str) or not s:
        return ("other", 0.0)

    # ratio like 5MG/5ML, 100MCG/ML, etc.
    m = RATIO_RE.search(s)
    if m:
        num = float(m.group("num"))
        unum = m.group("unum").upper()
        den = float(m.group("den"))
        dnum = m.group("dnum").upper()
        if unum in TO_MG and dnum in DEN_TO_ML and den > 0:
            mg = num * TO_MG[unum]
            ml = den * DEN_TO_ML[dnum]
            return ("mg_per_ml", mg / ml)
        return ("other", 0.0)

    # simple quantity like 500MG, 0.5G, 250MCG
    m = UNIT_RE.search(s)
    if m:
        qty = float(m.group("qty"))
        uom = m.group("uom").upper()
        if uom in TO_MG:
            return ("mg", qty * TO_MG[uom])
        if uom == "IU":
            return ("iu", qty)
        # ignore ML, %, MMOL for now -> could be extended
    return ("other", 0.0)

def strength_similarity(esoa_strengths: List[str], pnf_strengths: List[str]) -> float:
    """Compare first strength token from each side with simple conversions."""
    s_e = _first(esoa_strengths or [])
    s_p = _first(pnf_strengths or [])
    kind_e, val_e = parse_strength_value(s_e)
    kind_p, val_p = parse_strength_value(s_p)
    if kind_e == "other" or kind_p == "other":
        return 0.5 if s_e and s_p else 0.0
    if kind_e != kind_p:
        # allow mg vs mg_per_ml partial credit if val magnitudes are close for some products
        return 0.6 if {kind_e, kind_p} == {"mg", "mg_per_ml"} else 0.2
    # same kind -> compare within tolerance
    if val_e == 0 and val_p == 0:
        return 1.0
    if val_e == 0 or val_p == 0:
        return 0.0
    rel_err = abs(val_e - val_p) / max(val_e, val_p)
    if rel_err <= 0.05:  # within 5%
        return 1.0
    if rel_err <= 0.20:
        return 0.7
    return 0.2

def route_similarity(r_e: str | List[str], r_p: str | List[str]) -> float:
    """Compare routes and return similarity score between 0.0-1.0"""
    # Convert string inputs to lists if needed
    r_e_list = [r_e] if isinstance(r_e, str) else r_e
    r_p_list = [r_p] if isinstance(r_p, str) else r_p
    
    # Get first route from each list
    route_e = _first(r_e_list or [])
    route_p = _first(r_p_list or [])

    # Handle missing routes
    if not route_e or not route_p:
        return 0.0

    # Normalize routes
    route_e = route_e.upper()
    route_p = route_p.upper()

    # Exact match
    if route_e == route_p:
        return 1.0

    # Check route families
    fam_e = ROUTE_FAMILY.get(route_e)
    fam_p = ROUTE_FAMILY.get(route_p)
    if fam_e and fam_p and fam_e == fam_p:
        return 0.8

    return 0.2
    
def form_similarity(f_e: List[str], f_p: List[str]) -> float:
    fe, fp = _first(f_e or []), _first(f_p or [])
    if not fe or not fp:
        return 0.0
    fe, fp = fe.upper(), fp.upper()
    if fe == fp:
        return 1.0
    for canon, group in FORM_SYNONYM.items():
        if fe in group and fp in group:
            return 0.7
    return 0.0

def bool_to_boost(b: bool) -> float:
    return 1.0 if b else 0.0

# ----------------------------
# Scoring
# ----------------------------

def score_row(e_row, p_row) -> Tuple[float, dict]:
    s_sim = strength_similarity(e_row.get("_STRENGTHS", []), p_row.get("_STRENGTHS", []))
    f_sim = form_similarity(e_row.get("_FORMS", []), p_row.get("_FORMS", []))
    r_sim = route_similarity(e_row.get("_ROUTE_N", ""), p_row.get("_ROUTE_N", ""))

    brand_hit = bool_to_boost((e_row.get("_GENERIC_FROM_BRAND") or "") != "")
    atc_hit   = bool_to_boost(("ATC_CODE" in p_row.index) and str(p_row.get("ATC_CODE") or "") != "")

    # weights: emphasize dose, then route/form, with small boosts for corroboration
    score = 0.50 * s_sim + 0.20 * f_sim + 0.20 * r_sim + 0.10 * (0.6*brand_hit + 0.4*atc_hit)

    parts = {
        "strength_sim": round(s_sim, 3),
        "form_sim": round(f_sim, 3),
        "route_sim": round(r_sim, 3),
        "brand_hit": int(brand_hit),
        "atc_hit": int(atc_hit),
    }
    return (round(float(score), 4), parts)

def bucket(score: float, rank: int, ties: bool) -> str:
    """
    Auto-accept: score >= 0.90 AND top-1 unique.
    Needs-review: 0.60 <= score < 0.90 OR score>=0.90 but tie.
    Rejected: score < 0.60
    """
    if score >= 0.90 and rank == 1 and not ties:
        return "auto_accepted"
    if score >= 0.60:
        return "needs_review"
    return "rejected"

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Score eSOA–PNF candidates via _GENERIC_BASE blocking and SIM features.")
    ap.add_argument("--esoa-in", required=True)
    ap.add_argument("--pnf-in", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--topk", type=int, default=3, help="Keep top-K candidates per eSOA row (default 3)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    esoa = pd.read_csv(args.esoa_in)
    pnf  = pd.read_csv(args.pnf_in)

    # Canonicalize (some exporters lower-case arrays; re-evaluate columns)
    esoa.columns = [c.upper() for c in esoa.columns]
    pnf.columns  = [c.upper() for c in pnf.columns]

    # Ensure required columns
    for req in ["_GENERIC_BASE", "_STRENGTHS", "_FORMS", "_ROUTE_N", "_SIG3"]:
        if req not in esoa.columns:
            raise KeyError(f"eSOA missing {req}")
        if req not in pnf.columns:
            raise KeyError(f"PNF missing {req}")

    # Convert list-like columns if they were serialized as strings
    def ensure_listcol(df, col):
        if df[col].dtype == object:
            # try to parse simple list strings like "['500 MG']" or "500 MG"
            def parse_cell(x):
                if isinstance(x, list): return x
                s = str(x)
                if s.startswith("[") and s.endswith("]"):
                    # very naive split; robust option is ast.literal_eval
                    try:
                        import ast
                        v = ast.literal_eval(s)
                        if isinstance(v, list): return [str(z) for z in v]
                    except Exception:
                        pass
                return [s] if s else []
            df[col] = df[col].map(parse_cell)
        return df

    for c in ["_STRENGTHS","_FORMS"]:
        esoa = ensure_listcol(esoa, c)
        pnf  = ensure_listcol(pnf, c)

    # Blocking on _GENERIC_BASE
    pnf_blocks = {g: sub for g, sub in pnf.groupby("_GENERIC_BASE", dropna=False)}
    rows = []
    for _, er in esoa.iterrows():
        g = er.get("_GENERIC_BASE", "")
        if g not in pnf_blocks:
            continue
        cand = pnf_blocks[g]
        scored = []
        for _, pr in cand.iterrows():
            sc, parts = score_row(er, pr)
            scored.append((sc, parts, pr))
        if not scored:
            continue
        # rank
        scored.sort(key=lambda t: t[0], reverse=True)
        topk = scored[: max(1, args.topk)]
        # tie check for top-1
        has_tie = len(topk) > 1 and topk[0][0] == topk[1][0]

        rank = 0
        for sc, parts, pr in topk:
            rank += 1
            rows.append({
                "_GENERIC_BASE": g,
                "_SIG3_ESOA": er["_SIG3"],
                "_SIG3_PNF": pr["_SIG3"],
                "score": sc,
                "rank": rank,
                "ties_top1": int(has_tie),
                "strength_sim": parts["strength_sim"],
                "form_sim": parts["form_sim"],
                "route_sim": parts["route_sim"],
                "brand_hit": parts["brand_hit"],
                "atc_hit": parts["atc_hit"],
                # carry IDs/refs if present
                **{k: er[k] for k in er.index if k.endswith("_ID") or k in ["ITEM_NUMBER","ITEM_REF_CODE"] if k in er},
                **{f"{k}_PNF": pr[k] for k in pr.index if k in ["MOLECULE","ROUTE","TECHNICAL_SPECIFICATIONS","ATC_CODE"] and k in pr},
            })
    if not rows:
        (out_dir / "candidates_raw.csv").write_text("", encoding="utf-8")
        (out_dir / "scored.csv").write_text("", encoding="utf-8")
        (out_dir / "buckets.csv").write_text("", encoding="utf-8")
        print("No candidates generated. Check _GENERIC_BASE coverage.")
        return

    cand_df = pd.DataFrame(rows)
    cand_path = out_dir / "candidates_raw.csv"
    cand_df.to_csv(cand_path, index=False)

    # pick top-1 per eSOA _SIG3
    cand_df["_key"] = cand_df["_SIG3_ESOA"]
    top1 = cand_df.sort_values(["score", "rank"], ascending=[False, True]).groupby("_key", as_index=False).head(1).drop(columns=["_key"])
    # bucket
    top1["bucket"] = [
        bucket(sc, rk, bool(t)) for sc, rk, t in zip(top1["score"], top1["rank"], top1["ties_top1"])
    ]

    # write scored + buckets + diagnostics
    scored_path = out_dir / "scored.csv"
    buckets = top1["bucket"].value_counts(dropna=False).rename_axis("bucket").reset_index(name="count")
    buckets["percent"] = (buckets["count"] / len(top1) * 100).round(2)
    buckets_path = out_dir / "buckets.csv"

    # --- Summary with rates ---
    total = int(len(top1))
    auto  = int((top1["bucket"] == "auto_accepted").sum())
    review= int((top1["bucket"] == "needs_review").sum())
    rej   = int((top1["bucket"] == "rejected").sum())

    def pct(n): 
        return round((n / total * 100.0), 2) if total else 0.0

    summary_rows = [
        {"bucket":"auto_accepted", "count":auto,  "percent":pct(auto)},
        {"bucket":"needs_review",  "count":review,"percent":pct(review)},
        {"bucket":"rejected",      "count":rej,   "percent":pct(rej)},
        {"bucket":"TOTAL",         "count":total, "percent":100.00 if total else 0.0},
]


    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "summary_matches.csv"
    summary_md  = out_dir / "summary_matches.md"
    summary_df.to_csv(summary_csv, index=False)

    summary_md.write_text(
        "\n".join([
            "# Match Summary (Scored)",
            "",
            f"- Total eSOA rows with a candidate: **{total:,}**",
            f"- Auto-accepted: **{auto:,}** ({pct(auto)}%)",
            f"- Needs review: **{review:,}** ({pct(review)}%)",
            f"- Rejected: **{rej:,}** ({pct(rej)}%)",
            "",
            "## Breakdown table",
            summary_df.to_markdown(index=False) if hasattr(summary_df, "to_markdown") else summary_df.to_string(index=False),
        ]),
        encoding="utf-8"
        )

    print("\nSummary (scored):")
    print(summary_df.to_string(index=False))


    top1.to_csv(scored_path, index=False)
    buckets.to_csv(buckets_path, index=False)

    print(f"✅ Candidates (top{args.topk}) -> {cand_path}  rows={len(cand_df):,}")
    print(f"✅ Scored (top1)              -> {scored_path} rows={len(top1):,}")
    print(f"✅ Buckets                    -> {buckets_path}")
    print(buckets.to_string(index=False))

if __name__ == "__main__":
    main()