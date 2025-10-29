#!/usr/bin/env python3
"""
Robust runner for the formularymatching pipeline.

Usage: python run.py --esoa esoa_clean.csv --pnf pnf.csv --brand-map fda_brand_map.csv --who "WHO ATC-DDD 2024-07-31.csv"
"""

import argparse
import subprocess
import sys
import shlex
import logging
from pathlib import Path
from datetime import datetime

def run(cmd_list):
    """Run a command (list form). Log and raise on failure."""
    nxt = " ".join(shlex.quote(str(p)) for p in cmd_list)
    logging.info("▶ %s", nxt)
    try:
        subprocess.run(cmd_list, check=True)
    except subprocess.CalledProcessError as e:
        logging.error("Command failed (exit %s): %s", e.returncode, nxt)
        raise

def ensure_path_exists(p: Path, should_exist=True):
    if should_exist and not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")

def parse_args():
    ap = argparse.ArgumentParser(description="Run the formularymatching pipeline (robust runner).")
    ap.add_argument("--esoa", default="esoa_clean.csv", help="Input ESOA CSV")
    ap.add_argument("--pnf", default="pnf.csv", help="Input PNF CSV")
    ap.add_argument("--brand-map", default="fda_brand_map.csv", help="FDA brand map CSV")
    ap.add_argument("--who", default="WHO ATC-DDD 2024-07-31.csv", help="WHO ATC CSV")
    ap.add_argument("--python", default=sys.executable, help="Python executable to run scripts (defaults to current interpreter)")
    ap.add_argument("--workdir", default="work", help="Working directory")
    ap.add_argument("--outdir", default="out", help="Intermediate outputs directory")
    ap.add_argument("--deliver", default="deliver", help="Final deliverables directory")
    ap.add_argument("--topk", type=int, default=3, help="Top-K candidates to keep when scoring")
    ap.add_argument("--clean", action="store_true", help="Remove and recreate work/out/deliver directories before running")
    ap.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return ap.parse_args()

def main():
    args = parse_args()

    # logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")

    py = args.python
    ESOA = Path(args.esoa)
    PNF  = Path(args.pnf)
    FDA  = Path(args.brand_map)
    WHO  = Path(args.who)

    # Validate inputs early
    for p in (ESOA, PNF, FDA, WHO):
        logging.debug("Checking input %s", p)
        ensure_path_exists(p, should_exist=True)

    work = Path(args.workdir)
    out  = Path(args.outdir)
    deliver = Path(args.deliver)

    if args.clean:
        import shutil
        for d in (work, out, deliver):
            if d.exists():
                logging.info("Cleaning directory: %s", d)
                shutil.rmtree(d)

    # Ensure directories exist
    for d in (work, out, deliver):
        d.mkdir(parents=True, exist_ok=True)

    try:
        # 1) prepare
        run([py, "bin/prepare_esoa.py", "--in", str(ESOA), "--out", str(work / "esoa_prep.csv"), "--brand-map", str(FDA)])
        run([py, "bin/prepare_pnf.py",  "--in", str(PNF),  "--out", str(work / "pnf_prep.csv")])

        # 2) enrich
        run([py, "bin/enrich_fda.py",
             "--esoa-in", str(work / "esoa_prep.csv"),
             "--pnf-in",  str(work / "pnf_prep.csv"),
             "--brand-map", str(FDA),
             "--esoa-out", str(work / "esoa_fda.csv"),
             "--pnf-out", str(work / "pnf_fda.csv")])

        # 2.5) Aho–Corasick
        run([py, "bin/ac_features.py",
             "--esoa-in", str(work / "esoa_fda.csv"),
             "--pnf-in",  str(work / "pnf_fda.csv"),
             "--fda-in",  str(FDA),
             "--who-in",  str(WHO),
             "--esoa-out", str(work / "esoa_fda_ac.csv"),
             "--pnf-key", "MOLECULE",
             "--pnf-name-cols", "MOLECULE", "DRUG_NAME", "BRAND_NAME",
             "--desc-candidates", "TECHNICAL SPECIFICATIONS", "DESCRIPTION"])

        # 3) features
        run([py, "bin/features.py",
             "--esoa-in", str(work / "esoa_fda_ac.csv"),
             "--pnf-in", str(work / "pnf_fda.csv"),
             "--esoa-out", str(work / "esoa_feat.csv"),
             "--pnf-out", str(work / "pnf_feat.csv")])

        # 4) score
        run([py, "bin/score_candidates.py",
             "--esoa-in", str(work / "esoa_feat.csv"),
             "--pnf-in", str(work / "pnf_feat.csv"),
             "--out-dir", str(out),
             "--topk", str(args.topk)])

        # 5) export
        run([py, "bin/export_final.py",
             "--esoa-in", str(work / "esoa_feat.csv"),
             "--pnf-in", str(work / "pnf_feat.csv"),
             "--matches", str(out / "scored.csv"),
             "--out-dir", str(deliver),
             "--and-generic-base"])

    except Exception as exc:
        logging.exception("Pipeline failed: %s", exc)
        sys.exit(2)

    final_file = deliver / "esoa_plus_pnf4.csv"
    logging.info("✅ Done! Final file (note: export uses inner join) expected at: %s", final_file)
    print()

if __name__ == "__main__":
    main()