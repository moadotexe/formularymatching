#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Inner-join eSOA and PNF on _SIG3 (+ optional _GENERIC_BASE)")
    ap.add_argument("--esoa-in", required=True)
    ap.add_argument("--pnf-in", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--and-generic-base", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    e = pd.read_csv(args.esoa_in); p = pd.read_csv(args.pnf_in)

    # Canonicalize
    e.columns = [c.upper() for c in e.columns]
    p.columns = [c.upper() for c in p.columns]

    key = "_SIG3"
    if key not in e.columns or key not in p.columns:
        raise KeyError(f"Missing {key} in inputs. eSOA has={key in e.columns}, PNF has={key in p.columns}")

    on = [key]
    if args.and_generic_base:
        if "_GENERIC_BASE" not in e.columns or "_GENERIC_BASE" not in p.columns:
            raise KeyError("`--and-generic-base` set but _GENERIC_BASE missing.")
        on.append("_GENERIC_BASE")

    matches = e.merge(p, on=on, how="inner", suffixes=("_ESOA","_PNF"))
    matches.to_csv(out_dir / "matches.csv", index=False)

    # Anti-joins by primary key only
    eu = e.merge(p[[key]].drop_duplicates(), on=key, how="left", indicator=True)
    eu = eu[eu["_merge"]=="left_only"].drop(columns=["_merge"])
    pu = p.merge(e[[key]].drop_duplicates(), on=key, how="left", indicator=True)
    pu = pu[pu["_merge"]=="left_only"].drop(columns=["_merge"])
    eu.to_csv(out_dir / "unmatched_esoa.csv", index=False)
    pu.to_csv(out_dir / "unmatched_pnf.csv", index=False)

    print(f"✅ Matches -> {out_dir/'matches.csv'} | rows={len(matches):,}")
    print(f"   Unmatched eSOA -> {out_dir/'unmatched_esoa.csv'}  | PNF -> {out_dir/'unmatched_pnf.csv'}")

if __name__ == "__main__":
    main()