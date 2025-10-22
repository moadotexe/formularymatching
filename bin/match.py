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

        # --- Diagnostics buckets for eSOA by join key ---
    key = args.key  # ensure you've uppercased earlier
    esoa_counts = e[key].value_counts(dropna=False)
    pnf_counts  = p[key].value_counts(dropna=False)

    def bucket_for_sig(sig):
        # missing/empty key
        if pd.isna(sig) or str(sig) == "":
            return "rejected"
        e = int(esoa_counts.get(sig, 0))
        p = int(pnf_counts.get(sig, 0))
        if p == 0:
            return "rejected"
        if e == 1 and p == 1:
            return "auto_accepted"
        return "needs_review"

    diag = e[[key]].copy()
    diag["_BUCKET"] = diag[key].map(bucket_for_sig)

    # Tally + rates
    total = len(diag)
    auto  = int((diag["_BUCKET"] == "auto_accepted").sum())
    review= int((diag["_BUCKET"] == "needs_review").sum())
    rej   = int((diag["_BUCKET"] == "rejected").sum())

    def pct(n):
        return round((n / total * 100.0), 2) if total else 0.0

    
    summary_rows = [
        {"bucket":"auto_accepted", "count":auto,  "percent":pct(auto)},
        {"bucket":"needs_review",  "count":review,"percent":pct(review)},
        {"bucket":"rejected",      "count":rej,   "percent":pct(rej)},
        {"bucket":"TOTAL",         "count":total, "percent":100.00 if total else 0.0},
    ]
    summary_df = pd.DataFrame(summary_rows)

    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary_matches.csv"
    summary_md  = out_dir / "summary_matches.md"
    diag.to_csv(diag_dir / "buckets_by_esoa.csv", index=False)
    summary_df.to_csv(summary_csv, index=False)

    # Write Markdown
    try:
        tbl_md = summary_df.to_markdown(index=False)
    except Exception:
        tbl_md = summary_df.to_string(index=False)

    (Path(summary_md)).write_text(
        "\n".join([
            "# Match Summary (Strict Join)",
            "",
            f"- Total eSOA rows evaluated: **{total:,}**",
            f"- Auto-accepted: **{auto:,}** ({pct(auto)}%)",
            f"- Needs review: **{review:,}** ({pct(review)}%)",
            f"- Rejected: **{rej:,}** ({pct(rej)}%)",
            "",
            "## Breakdown table",
            tbl_md,
        ]),
        encoding="utf-8"
    )

    print("\nSummary (strict):")
    print(summary_df.to_string(index=False))


    print(f"✅ Matches -> {out_dir/'matches.csv'} | rows={len(matches):,}")
    print(f"   Unmatched eSOA -> {out_dir/'unmatched_esoa.csv'}  | PNF -> {out_dir/'unmatched_pnf.csv'}")

if __name__ == "__main__":
    main()