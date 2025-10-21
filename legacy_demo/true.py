from typing import List, Set, Optional
import re
import pandas as pd
from pathlib import Path

#This is under assumption everything is cleaned

# === CONFIG ===
DATA_DIR = Path("C:/Users/Juan Eduardo Banzon/Desktop/PIDS/PNF/eSOA_input")
ESOA_CSV = DATA_DIR / "esoa.csv"
PNF_XLSX = DATA_DIR / "pnf.csv"
WHO_ATC = DATA_DIR / "WHO ATC-DDD 2024-07-31.csv"
FDA_BRAND = DATA_DIR / "fda_brand_map.csv"
FDA_FOOD = DATA_DIR / "fda_food_products.csv"



# Compile regex patterns once
WHITESPACE_PAT = re.compile(r"\s+")
STRENGTH_PAT = re.compile(
    r"(?:(\d+(?:\.\d+)?)\s*(MCG|MG|G|IU|ML|%|MMOL)(?:\s*/\s*(\d+(?:\.\d+)?)\s*(ML|G|L))?)"
    r"|(?:(\d+(?:\.\d+)?)\s*(IU)\s*/\s*(ML))",
    re.IGNORECASE
)
COMBO_SEP_PAT = re.compile(r"|".join([r"\+", r"/", r"&", r"\bWITH\b", r"\bAND\b"]))

# Mapping dictionaries
UNIT_MAP = {
    "MGS": "MG", "MG.": "MG", "MGM": "MG", "MCG.": "MCG", "UG": "MCG",
    "GMS": "G", "GM": "G", "IU/ML":"IU/ML", "IU":"IU"
}

FORM_MAP = {
    "TAB":"TABLET", "TABLETS":"TABLET", "CAP":"CAPSULE", "CAPS":"CAPSULE",
    "CAPSULES":"CAPSULE", "SACHETS":"SACHET", "AMP":"AMPUL", "AMPULE":"AMPUL",
    "AMPOULE":"AMPUL", "VIALS":"VIAL", "SUSP":"SUSPENSION", "SYR":"SYRUP",
    "PFS":"PRE-FILLED SYRINGE", "INH":"INHALER", "SOLN":"SOLUTION",
    "DPS":"DROPS", "DRPS":"DROPS", "MR":"MODIFIED RELEASE", "SL":"SUBLINGUAL"
}

FORM_KEYS: Set[str] = {
    "TABLET", "CAPSULE", "SACHET", "AMPUL", "VIAL", "SYRUP", "SUSPENSION",
    "SOLUTION", "DROPS", "OINTMENT", "CREAM", "GEL", "PATCH", "INHALER",
    "SPRAY", "SUPPOSITORY", "LOZENGE", "ELIXIR", "EMULSION", "PRE-FILLED SYRINGE"
}

def pick_col(df, candidates, *, must=False, label=""):
    def norm(s):
        return re.sub(r'[^\w\s]', '', str(s)).strip().upper().replace(' ', '_')
    cmap = {norm(c): c for c in df.columns}
    for cand in candidates:
        key = norm(cand)
        if key in cmap:
            return cmap[key]
    if must:
        raise KeyError(f"[pick_col] Missing required column for {label}: tried {candidates}")
    return None

def normalize_text(text: str) -> str:
    """Normalize text with consistent spacing and case"""
    if pd.isna(text):
        return ""
    text = str(text).upper()
    text = re.sub(r"[^\w/%+().-]+", " ", text)
    text = WHITESPACE_PAT.sub(" ", text).strip()
    for old, new in UNIT_MAP.items():
        text = re.sub(fr"\b{old}\b", new, text)
    for old, new in FORM_MAP.items():
        text = re.sub(fr"\b{old}\b", new, text)
    return text

def extract_forms(text: str) -> List[str]:
    """Extract dosage forms from text"""
    norm_text = normalize_text(text)
    forms = [f for f in FORM_KEYS if re.search(rf"\b{re.escape(f)}\b", norm_text)]
    return forms or ["UNSPECIFIED"]

def extract_strengths(text: str) -> List[str]:
    """Extract strength specifications from text"""
    norm_text = normalize_text(text)
    strengths: Set[str] = set()
    
    for match in STRENGTH_PAT.finditer(norm_text):
        if match.group(1):
            base = f"{match.group(1)}{match.group(2)}"
            if match.group(3) and match.group(4):
                base = f"{base}/{match.group(3)}{match.group(4)}"
            strengths.add(base)
        elif match.group(5):
            strengths.add(f"{match.group(5)}{match.group(6)}/{match.group(7)}")
    
    return list(strengths) or ["UNSPECIFIED"]

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column headers by removing special characters and converting to uppercase"""
    df.columns = [
        re.sub(r'[^\w\s]', '', str(col)).strip().upper().replace(' ', '_')
        for col in df.columns
    ]
    return df

def ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure column names are unique by appending numbers to duplicates"""
    seen = {}
    new_cols = []
    
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    
    df.columns = new_cols
    return df

def normalize_salts_base(text: str) -> str:
    """Normalize salt forms to their base molecule names"""
    if pd.isna(text):
        return ""
    
    text = normalize_text(text)
    # Add common salt form patterns
    salt_patterns = [
        r"\s+HYDROCHLORIDE\b",
        r"\s+HCL\b",
        r"\s+SULFATE\b",
        r"\s+SODIUM\b",
        r"\s+POTASSIUM\b",
        r"\s+MALEATE\b",
        r"\s+CITRATE\b"
    ]
    
    for pattern in salt_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    return text.strip()

def process_dataframe(df: pd.DataFrame, desc_col: Optional[str], mol_col: Optional[str] = None) -> pd.DataFrame:
    """Process either eSOA or PNF dataframe with common operations"""
    if desc_col is None:
        raise ValueError("Description column name cannot be None")
        
    result = df.copy()
    result["_FORMS"] = result[desc_col].map(extract_forms)
    result["_STRENGTHS"] = result[desc_col].map(extract_strengths)
    result["_SIG"] = result.apply(
        lambda r: "|".join([
            ",".join(sorted(r["_STRENGTHS"])),
            ",".join(sorted(r["_FORMS"]))
        ]),
        axis=1
    )
    
    if mol_col:
        result[f"_{mol_col}_BASE"] = result[mol_col].map(normalize_salts_base)
    
    return result

def main():
    # Load and clean data
    esoa_raw = pd.read_csv(ESOA_CSV)
    pnf_raw = pd.read_csv(PNF_XLSX)

    print(pnf_raw.head())  # Debug: Print first few rows of PNF data
    
    # Clean headers & duplicates
    esoa_raw = ensure_unique_columns(normalize_headers(esoa_raw))
    pnf_raw = ensure_unique_columns(normalize_headers(pnf_raw))
    
    
    # Get column names with more candidate options
    esoa_desc_col = pick_col(esoa_raw, ["DESCRIPTION", "ITEM DESCRIPTION"], must=True, label="eSOA Description")
    pnf_spec_col = pick_col(pnf_raw, ["TECHNICAL SPECIFICATIONS"], must=True, label="PNF technical spec")
    pnf_mol_col = pick_col(
        pnf_raw, 
        ["PNF_MOLECULE", "MOLECULE", "GENERIC_NAME", "GENERIC NAME", "DRUG_NAME"],
        must=False,
        label="PNF molecule"
    )
    
    # Process dataframes
    esoa_df = process_dataframe(esoa_raw, esoa_desc_col)
    pnf_df = process_dataframe(pnf_raw, pnf_spec_col, pnf_mol_col)  # Use the found column name
    
    print(f"eSOA rows: {len(esoa_df)} | PNF rows: {len(pnf_df)}")
    if pnf_mol_col:
        print(f"Using molecule column: {pnf_mol_col}")
    else:
        print("Warning: No molecule column found in PNF data")
    
    return esoa_df, pnf_df

    print (esoa_df.head())



if __name__ == "__main__":
    main()