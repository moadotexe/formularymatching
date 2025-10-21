#!/usr/bin/env python3
import subprocess
from pathlib import Path

def sh(cmd: str):
    print(f"\n▶ {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    # Input files
    ESOA = "esoa_clean.csv"
    PNF  = "pnf.csv"
    FDA  = "fda_brand_map.csv"

    # Working dirs
    Path("work").mkdir(exist_ok=True)
    Path("out").mkdir(exist_ok=True)
    Path("deliver").mkdir(exist_ok=True)

    # 1️⃣ Prepare both datasets
    sh(f"python bin/prepare_esoa.py --in {ESOA} --out work/esoa_prep.csv --brand-map {FDA}")
    sh(f"python bin/prepare_pnf.py  --in {PNF}  --out work/pnf_prep.csv")

    # 2️⃣ Enrich with FDA
    sh(f"python bin/enrich_fda.py --esoa-in work/esoa_prep.csv --pnf-in work/pnf_prep.csv "
       f"--brand-map {FDA} --esoa-out work/esoa_fda.csv --pnf-out work/pnf_fda.csv")

    # 3️⃣ Feature engineering (_ROUTE_N + _SIG3)
    sh(f"python bin/features.py --esoa-in work/esoa_fda.csv --pnf-in work/pnf_fda.csv "
       f"--esoa-out work/esoa_feat.csv --pnf-out work/pnf_feat.csv")

    # 4️⃣ Scoring (dose/form/route similarity)
    sh(f"python bin/score_candidates.py --esoa-in work/esoa_feat.csv --pnf-in work/pnf_feat.csv "
       f"--out-dir out --topk 3")

    # 5️⃣ Export the final eSOA-centric file (using scored matches)
    sh(f"python bin/export_final.py --esoa-in work/esoa_feat.csv --pnf-in work/pnf_feat.csv --matches out/scored.csv --out deliver/esoa_plus_pnf4.csv")

    print("\n✅ Done! Final file: deliver/esoa_plus_pnf4.csv")

if __name__ == "__main__":
    main()