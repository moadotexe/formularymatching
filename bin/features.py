import argparse
from pathlib import Path
import pandas as pd

from normalize import (
    pick_col, normalize_route, rebuild_sig_with_route, extract_strengths, add_route_family_and_form_group           # NEW: numeric dosage features add_route_family_and_form_group,      # NEW: route/form groupings
)

# --- ADD: lightweight fallback base & robust _SIG3 builder -------------------
import re

SALTS = r"(?: HCL| HYDROCHLORIDE| SODIUM| POTASSIUM| CALCIUM| PHOSPHATE| SULFATE| MALEATE| TARTRATE| MESYLATE| NITRATE| ACETATE)\b"

def _fallback_generic_base_row(row: dict) -> str:
    """
    Best-effort base derivation when _GENERIC_BASE is missing.
    Try common fields; strip typical salt words; uppercase + normalize spaces.
    """
    for c in ("_GENERIC_BASE", "MOLECULE", "GENERIC_NAME", "GENERIC", "ACTIVE_INGREDIENT"):
        v = row.get(c)
        if isinstance(v, str) and v.strip():
            base = re.sub(SALTS, "", " " + v.strip().upper()).strip()
            base = re.sub(r"\s+", " ", base)
            return base
    # fallback from description if absolutely necessary
    desc = row.get("DESCRIPTION") or row.get("TECHNICAL SPECIFICATIONS") or row.get("TECHNICAL_SPECIFICATIONS")
    if isinstance(desc, str) and desc.strip():
        # keep first token-ish word as a weak base
        tok = re.split(r"[,/;()\-]", desc.upper(), maxsplit=1)[0]
        tok = re.sub(SALTS, "", " " + tok).strip()
        tok = re.sub(r"\s+", " ", tok)
        return tok
    return ""

def ensure_generic_base(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure _GENERIC_BASE exists and is non-empty, using fallback if needed."""
    df = df.copy()
    if "_GENERIC_BASE" not in df.columns:
        df["_GENERIC_BASE"] = ""

    mask_empty = df["_GENERIC_BASE"].astype(str).str.len().eq(0)
    if mask_empty.any():
        df.loc[mask_empty, "_GENERIC_BASE"] = df[mask_empty].apply(lambda r: _fallback_generic_base_row(r), axis=1)
    return df

def ensure_sig3_robust(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure _SIG3 never collapses to empty pieces:
    fill empty strengths/forms/routes with 'UNSPECIFIED' before building.
    """
    df = df.copy()
    # Ensure presence
    for col in ("_STRENGTHS", "_FORMS", "_ROUTE_N"):
        if col not in df.columns:
            df[col] = ""

    # Coerce empties
    s = df["_STRENGTHS"].astype(str).replace({"nan": "", "None": ""})
    f = df["_FORMS"].astype(str).replace({"nan": "", "None": ""})
    r = df["_ROUTE_N"].astype(str).replace({"nan": "", "None": ""})

    s = s.where(s.str.len() > 0, "UNSPECIFIED")
    f = f.where(f.str.len() > 0, "UNSPECIFIED")
    r = r.where(r.str.len() > 0, "UNSPECIFIED")

    df["_SIG3"] = s + "|" + f + "|" + r
    return df

# Optional: unified IO so you can switch to .parquet intermediates easily
def read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, sep=None, engine="python", encoding="utf-8")

def write_any(df: pd.DataFrame, path: str) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".parquet":
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False)

def ensure_route_n(df: pd.DataFrame) -> pd.DataFrame:
    if "_ROUTE_N" in df.columns:
        return df
    route_col = pick_col(df, ["ROUTE","ADMIN_ROUTE","ROA","ROUTE_OF_ADMINISTRATION","ADMINISTRATION_ROUTE"], must=False)
    if route_col:
        df = df.copy()
        df["_ROUTE_N"] = df[route_col].map(normalize_route)
    return df

def ensure_sig3(df: pd.DataFrame) -> pd.DataFrame:
    if "_SIG3" not in df.columns:
        df = rebuild_sig_with_route(df)
    return df

def add_fast_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Precompute fields the scorer/matcher rely on:
      - _STRENGTH_FIRST, _STRENGTH_KIND, _STRENGTH_VAL
      - _FORM_FIRST, _FORM_GROUP
      - _ROUTE_FAMILY
      (and make relevant columns categorical for faster joins/groupbys)
    """
    df = add_route_family_and_form_group(df)
    # Optional: make join keys categorical to speed merges downstream
    for col in ("_SIG3", "_ROUTE_N", "_ROUTE_FAMILY", "_FORM_FIRST", "_FORM_GROUP", "_STRENGTH_KIND"):
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df

def main():
    ap = argparse.ArgumentParser(description="Ensure _ROUTE_N and _SIG3; add numeric dosage & route/form group features.")
    ap.add_argument("--esoa-in", required=True)
    ap.add_argument("--pnf-in", required=True)
    ap.add_argument("--esoa-out", required=True)
    ap.add_argument("--pnf-out", required=True)
    args = ap.parse_args()

    # Read
    esoa = read_any(args.esoa_in)
    pnf  = read_any(args.pnf_in)

    # Normalize route + ensure composite signature
    esoa = ensure_route_n(esoa);  pnf = ensure_route_n(pnf)
    esoa = ensure_sig3(esoa);     pnf = ensure_sig3(pnf)
    esoa = ensure_generic_base(esoa); pnf = ensure_generic_base(pnf)
    esoa = ensure_sig3_robust(esoa);  pnf = ensure_sig3_robust(pnf)

    # NEW: fast, numeric features for dosage + route/form families
    esoa = add_fast_features(esoa)
    pnf  = add_fast_features(pnf)

    # Write
    write_any(esoa, args.esoa_out)
    write_any(pnf,  args.pnf_out)

    print(f"✅ Features -> {args.esoa_out} / {args.pnf_out} | "
          f"keys present: _SIG3={('_SIG3' in esoa.columns) and ('_SIG3' in pnf.columns)} | "
          f"numeric dosage: {('_STRENGTH_VAL' in esoa.columns) and ('_STRENGTH_VAL' in pnf.columns)}")

if __name__ == "__main__":
    main()