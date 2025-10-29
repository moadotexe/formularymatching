#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
import sys

def run(args_list):
    # args_list is a Python list, not a single string
    print("\n▶", " ".join(map(str, args_list)))
    subprocess.run(args_list, check=True)

def main():
    ESOA = "esoa_clean.csv"
    PNF  = "pnf.csv"
    FDA  = "fda_brand_map.csv"
    WHO  = "WHO ATC-DDD 2024-07-31.csv"  # path with spaces is fine in list form

    Path("work").mkdir(exist_ok=True)
    Path("out").mkdir(exist_ok=True)
    Path("deliver").mkdir(exist_ok=True)

    # 1) prepare
    run([sys.executable, "bin/prepare_esoa.py", "--in", ESOA, "--out", "work/esoa_prep.csv", "--brand-map", FDA])
    run([sys.executable, "bin/prepare_pnf.py",  "--in", PNF,  "--out", "work/pnf_prep.csv"])

    # 2) enrich
    run([sys.executable, "bin/enrich_fda.py",
         "--esoa-in", "work/esoa_prep.csv", "--pnf-in", "work/pnf_prep.csv",
         "--brand-map", FDA, "--esoa-out", "work/esoa_fda.csv", "--pnf-out", "work/pnf_fda.csv"])

    # 2.5) Aho–Corasick (args as list; list-valued flags = multiple tokens)
    run([sys.executable, "bin/ac_features.py",
         "--esoa-in", "work/esoa_fda.csv",
         "--pnf-in",  "work/pnf_fda.csv",
         "--fda-in",  FDA,
         "--who-in",  WHO,
         "--esoa-out", "work/esoa_fda_ac.csv",
         "--pnf-key", "MOLECULE",
         "--pnf-name-cols", "MOLECULE", "DRUG_NAME", "BRAND_NAME",
         "--desc-candidates", "TECHNICAL SPECIFICATIONS", "DESCRIPTION"])

    # 3) features (use AC-enriched eSOA)
    run([sys.executable, "bin/features.py",
         "--esoa-in", "work/esoa_fda_ac.csv", "--pnf-in", "work/pnf_fda.csv",
         "--esoa-out", "work/esoa_feat.csv", "--pnf-out", "work/pnf_feat.csv"])

    # 4) score
    run([sys.executable, "bin/score_candidates.py",
         "--esoa-in", "work/esoa_feat.csv", "--pnf-in", "work/pnf_feat.csv",
         "--out-dir", "out", "--topk", "3"])

    # 5) export
    run([sys.executable, "bin/export_final.py",
         "--esoa-in", "work/esoa_feat.csv", "--pnf-in", "work/pnf_feat.csv",
         "--matches", "out/scored.csv", "--out", "deliver/esoa_plus_pnf4.csv"])

    print("\n✅ Done! Final file: deliver/esoa_plus_pnf4.csv")

if __name__ == "__main__":
    main()