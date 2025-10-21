#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from normalize import normalize_headers, ensure_unique_columns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoding", default="utf-8")
    args = ap.parse_args()

    df = pd.read_csv(args.inp, sep=None, engine="python", encoding=args.encoding)
    df = ensure_unique_columns(normalize_headers(df))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"✅ PNF prepared -> {args.out} | rows={len(df):,}")

if __name__ == "__main__":
    main()