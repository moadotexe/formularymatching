#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from normalize import (
    normalize_headers, ensure_unique_columns, pick_col,
    load_brand_map, enrich_esoa_parentheses_with_fda_brand
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--brand-map", required=True)
    ap.add_argument("--encoding", default="utf-8")
    ap.add_argument("--desc-candidates", nargs="+", default=[
        "TECHNICAL SPECIFICATIONS","TECHNICAL_SPECIFICATIONS",
        "DESCRIPTION","ITEM DESCRIPTION","SPECIFICATION","SPECIFICATIONS"
    ])
    args = ap.parse_args()

    df = pd.read_csv(args.inp, sep=None, engine="python", encoding=args.encoding)
    df = ensure_unique_columns(normalize_headers(df))

    brand_map = load_brand_map(Path(args.brand_map))
    desc_col = pick_col(df, args.desc_candidates, must=False)
    if desc_col:
        df = enrich_esoa_parentheses_with_fda_brand(df, desc_col=desc_col, brand_map=brand_map)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"✅ ESOA prepared -> {args.out} | rows={len(df):,}")

if __name__ == "__main__":
    main()