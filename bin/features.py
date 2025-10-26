#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

from normalize import (
    pick_col, normalize_route, rebuild_sig_with_route, extract_strengths, add_route_family_and_form_group           # NEW: numeric dosage features add_route_family_and_form_group,      # NEW: route/form groupings
)

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
    
    df = extract_strengths(df)
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