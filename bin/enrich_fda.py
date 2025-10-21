#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from normalize import (
    load_brand_map, enrich_with_brand_and_atc, normalize_headers,
)

def enrich(df, brand_map):
    # brand_source_cols includes DESCRIPTION to catch inline brand mentions too
    return enrich_with_brand_and_atc(
        df,
        brand_map=brand_map,
        who_atc=None,  # WHO not used in the 3-file flow
        brand_source_cols=["BRAND","TRADE_NAME","TRADE","BRAND_NAME","DESCRIPTION"],
        generic_source_cols=["GENERIC","GENERIC_NAME","MOLECULE","DRUG_NAME","ACTIVE_INGREDIENT"]
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esoa-in", required=True)
    ap.add_argument("--pnf-in", required=True)
    ap.add_argument("--brand-map", required=True)
    ap.add_argument("--esoa-out", required=True)
    ap.add_argument("--pnf-out", required=True)
    args = ap.parse_args()

    brand_map = load_brand_map(Path(args.brand_map))

    esoa = pd.read_csv(args.esoa_in)
    pnf  = pd.read_csv(args.pnf_in)

    esoa = enrich(esoa, brand_map)
    pnf  = enrich(pnf, brand_map)

    Path(args.esoa_out).parent.mkdir(parents=True, exist_ok=True)
    esoa.to_csv(args.esoa_out, index=False)
    pnf.to_csv(args.pnf_out, index=False)
    print(f"✅ FDA-enriched -> {args.esoa_out} / {args.pnf_out}")

if __name__ == "__main__":
    main()