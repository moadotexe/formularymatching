#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from normalize import pick_col, normalize_route, rebuild_sig_with_route

def ensure_route_n(df: pd.DataFrame) -> pd.DataFrame:
    if "_ROUTE_N" in df.columns: return df
    route_col = pick_col(df, ["ROUTE","ADMIN_ROUTE","ROA","ROUTE_OF_ADMINISTRATION","ADMINISTRATION_ROUTE"], must=False)
    if route_col:
        df = df.copy()
        df["_ROUTE_N"] = df[route_col].map(normalize_route)
    return df

def ensure_sig3(df: pd.DataFrame) -> pd.DataFrame:
    if "_SIG3" not in df.columns:
        df = rebuild_sig_with_route(df)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esoa-in", required=True)
    ap.add_argument("--pnf-in", required=True)
    ap.add_argument("--esoa-out", required=True)
    ap.add_argument("--pnf-out", required=True)
    args = ap.parse_args()

    esoa = pd.read_csv(args.esoa_in);  pnf = pd.read_csv(args.pnf_in)
    esoa = ensure_route_n(esoa);       pnf = ensure_route_n(pnf)
    esoa = ensure_sig3(esoa);          pnf = ensure_sig3(pnf)

    Path(args.esoa_out).parent.mkdir(parents=True, exist_ok=True)
    esoa.to_csv(args.esoa_out, index=False)
    pnf.to_csv(args.pnf_out, index=False)
    print(f"✅ Features -> {args.esoa_out} / {args.pnf_out} | keys present: _SIG3={('_SIG3' in esoa.columns) and ('_SIG3' in pnf.columns)}")

if __name__ == "__main__":
    main()
