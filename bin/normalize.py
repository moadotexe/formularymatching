# normalize.py
from typing import List, Set, Optional, Tuple
import re
import pandas as pd
from pathlib import Path

# ---------- Normalization maps ----------
UNIT_MAP = {
    "MGS": "MG", "MG.": "MG", "MGM": "MG",
    "MCG.": "MCG", "UG": "MCG",
    "GMS": "G", "GM": "G",
    "IU/ML": "IU/ML", "IU": "IU"
}

FORM_MAP = {
    "TAB":"TABLET","TABLETS":"TABLET",
    "CAP":"CAPSULE","CAPS":"CAPSULE","CAPSULES":"CAPSULE",
    "SACHETS":"SACHET",
    "AMP":"AMPUL","AMPULE":"AMPUL","AMPOULE":"AMPUL",
    "VIALS":"VIAL",
    "SUSP":"SUSPENSION","SYR":"SYRUP",
    "PFS":"PRE-FILLED SYRINGE",
    "INH":"INHALER","SOLN":"SOLUTION",
    "DPS":"DROPS","DRPS":"DROPS",
    "MR":"MODIFIED RELEASE","SL":"SUBLINGUAL"
}

FORM_KEYS: Set[str] = {
    "TABLET","CAPSULE","SACHET","AMPUL","VIAL","SYRUP","SUSPENSION",
    "SOLUTION","DROPS","OINTMENT","CREAM","GEL","PATCH","INHALER",
    "SPRAY","SUPPOSITORY","LOZENGE","ELIXIR","EMULSION","PRE-FILLED SYRINGE"
}

ROUTE_MAP = {
    "ORAL": "PO", "PO": "PO", "P.O.": "PO",
    "INTRAVENOUS": "IV", "IV": "IV",
    "INTRAMUSCULAR": "IM", "IM": "IM",
    "TOPICAL": "TOP", "TRANSDERMAL": "TOP",
    "SUBLINGUAL": "SL",
    "NASAL": "IN", "INTRANASAL": "IN",
    "INHALATION": "INH", "INHALER": "INH",
    "SUBCUTANEOUS": "SC", "S.C.": "SC", "SQ": "SC"
}

# ---------- Regex ----------
WHITESPACE_PAT = re.compile(r"\s+")
STRENGTH_PAT = re.compile(
    r"(?:(\d+(?:\.\d+)?)\s*(MCG|MG|G|IU|ML|%|MMOL)(?:\s*/\s*(\d+(?:\.\d+)?)\s*(ML|G|L))?)"
    r"|(?:(\d+(?:\.\d+)?)\s*(IU)\s*/\s*(ML))",
    re.IGNORECASE
)

# ---------- IO ----------
def load_csv_any(path: Path, *, delimiter: Optional[str]=None, encoding: Optional[str]=None) -> pd.DataFrame:
    """Sniff delimiter if not provided; tolerant CSV loader."""
    return pd.read_csv(
        path,
        sep=delimiter if delimiter is not None else None,
        encoding=encoding or "utf-8",
        engine="python"
    )

# ---------- Columns / headers ----------
def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize headers: trim, uppercase, spaces->underscores, strip punctuation."""
    df = df.copy()
    df.columns = [
        re.sub(r"[^\w\s]", "", str(col)).strip().upper().replace(" ", "_")
        for col in df.columns
    ]
    return df

def ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Disambiguate duplicate column names by suffixing _1, _2, ..."""
    df = df.copy()
    seen, new_cols = {}, []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    return df

def pick_col(df: pd.DataFrame, candidates: List[str], *, must=False, label=""):
    """Pick the first existing column among candidate aliases (case/spacing tolerant)."""
    def norm(s): return re.sub(r"[^\w\s]", "", str(s)).strip().upper().replace(" ", "_")
    cmap = {norm(c): c for c in df.columns}
    for cand in candidates:
        key = norm(cand)
        if key in cmap:
            return cmap[key]
    if must:
        raise KeyError(f"[pick_col] Missing required column for {label}: tried {candidates}")
    return None

# ---------- Text normalization ----------
def normalize_text(text: str) -> str:
    if pd.isna(text): return ""
    text = str(text).upper()
    text = re.sub(r"[^\w/%+().-]+", " ", text)
    text = WHITESPACE_PAT.sub(" ", text).strip()
    for old, new in UNIT_MAP.items():
        text = re.sub(fr"\b{re.escape(old)}\b", new, text)
    for old, new in FORM_MAP.items():
        text = re.sub(fr"\b{re.escape(old)}\b", new, text)
    return text

def normalize_route(x: str) -> str:
    x = normalize_text(x)
    return ROUTE_MAP.get(x, x) if x else ""

def extract_forms(text: str) -> List[str]:
    norm_text = normalize_text(text)
    forms = [f for f in FORM_KEYS if re.search(rf"\b{re.escape(f)}\b", norm_text)]
    return forms or ["UNSPECIFIED"]

def extract_strengths(text: str) -> List[str]:
    norm_text = normalize_text(text)
    strengths: Set[str] = set()
    for m in STRENGTH_PAT.finditer(norm_text):
        if m.group(1):  # qty + unit (optional per qty/unit)
            base = f"{m.group(1)}{m.group(2).upper()}"
            if m.group(3) and m.group(4):
                base = f"{base}/{m.group(3)}{m.group(4).upper()}"
            strengths.add(base)
        elif m.group(5):  # IU/ML
            strengths[].add(f"{m.group(5)}{m.group(6).upper()}/{m.group(7).upper()}")
    return sorted(strengths) or ["UNSPECIFIED"]

def normalize_salts_base(text: str) -> str:
    """Strip common salt/ion terms for a generic 'base' molecule string."""
    if pd.isna(text): return ""
    text = normalize_text(text)
    for pat in [r"\s+HYDROCHLORIDE\b", r"\s+HCL\b", r"\s+SULFATE\b", r"\s+SODIUM\b",
                r"\s+POTASSIUM\b", r"\s+MALEATE\b", r"\s+CITRATE\b"]:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text.strip()

# ---------- Feature builders ----------
def as_key(strengths: List[str], forms: List[str]) -> Tuple[str, str, str]:
    s = ",".join(sorted(strengths or []))
    f = ",".join(sorted(forms or []))
    return s, f, f"{s}|{f}"

def process_dataframe(df: pd.DataFrame, desc_col: Optional[str]) -> pd.DataFrame:
    """Legacy two-part signature builder: _SIG = strengths|forms from a description column."""
    if desc_col is None:
        raise ValueError("Description column name cannot be None")
    res = df.copy()
    res["_FORMS"] = res[desc_col].map(extract_forms)
    res["_STRENGTHS"] = res[desc_col].map(extract_strengths)
    s, f, k = zip(*[as_key(sr, fr) for sr, fr in zip(res["_STRENGTHS"], res["_FORMS"])])
    res["_SIG_STRENGTHS"] = list(s)
    res["_SIG_FORMS"] = list(f)
    res["_SIG"] = list(k)
    return res

def build_sig(strengths: List[str], forms: List[str], route: str = "") -> str:
    s = ",".join(sorted(strengths or []))
    f = ",".join(sorted(forms or []))
    r = route or ""
    return f"{s}|{f}|{r}" if r else f"{s}|{f}"

def _ensure_strengths_forms(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure _STRENGTHS and _FORMS exist. If missing, derive from best available text column."""
    res = df.copy()
    need_strengths = "_STRENGTHS" not in res.columns
    need_forms = "_FORMS" not in res.columns
    if not (need_strengths or need_forms):
        return res

    # Pick a description/spec column to parse from
    desc_col = None
    for c in ["TECHNICAL_SPECIFICATIONS", "TECHNICAL SPECIFICATIONS", "DESCRIPTION", "DESC", "PRODUCT_NAME"]:
        if c in res.columns:
            desc_col = c
            break
    if desc_col is None:
        # Nothing to parse; create UNSPECIFIED
        if need_strengths: res["_STRENGTHS"] = [["UNSPECIFIED"]] * len(res)
        if need_forms:     res["_FORMS"]     = [["UNSPECIFIED"]] * len(res)
        return res

    # Derive from chosen column
    if need_strengths:
        res["_STRENGTHS"] = res[desc_col].map(extract_strengths)
    if need_forms:
        res["_FORMS"] = res[desc_col].map(extract_forms)
    return res

def _ensure_route_n(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure _ROUTE_N exists by normalizing any sensible route column."""
    res = df.copy()
    if "_ROUTE_N" in res.columns:
        return res

    route_col = None
    for cand in ["ROUTE", "ADMIN_ROUTE", "ROA", "ROUTE_OF_ADMINISTRATION", "ADMINISTRATION_ROUTE"]:
        if cand in res.columns:
            route_col = cand
            break
    if route_col:
        res["_ROUTE_N"] = res[route_col].map(normalize_route)
    else:
        res["_ROUTE_N"] = ""
    return res

def rebuild_sig_with_route(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create `_SIG3 = strengths|forms|route` deterministically.
    - Ensures `_STRENGTHS`, `_FORMS`, `_ROUTE_N`.
    - Preserves existing `_SIG`.
    """
    res = df.copy()
    res = _ensure_strengths_forms(res)
    res = _ensure_route_n(res)
    res["_SIG3"] = [
        build_sig(
            strs if isinstance(strs, list) else [],
            frms if isinstance(frms, list) else [],
            rt   if isinstance(rt, str)  else ""
        )
        for strs, frms, rt in zip(res.get("_STRENGTHS", [[]]*len(res)),
                                  res.get("_FORMS", [[]]*len(res)),
                                  res.get("_ROUTE_N", [""]*len(res)))
    ]
    return res

# ---------- Lookups ----------
def load_brand_map(path: Path) -> pd.DataFrame:
    """Load FDA brand→generic map and normalize fields."""
    df = load_csv_any(path)
    df = ensure_unique_columns(normalize_headers(df))
    brand_col = pick_col(df, ["BRAND_NAME", "BRAND", "PRODUCT_NAME"], must=True, label="FDA brand")
    gen_col   = pick_col(df, ["GENERIC_NAME", "GENERIC", "ACTIVE_INGREDIENT"], must=True, label="FDA generic")
    out = df[[brand_col, gen_col]].rename(columns={brand_col: "_FDA_BRAND", gen_col: "_FDA_GENERIC"})
    out["_FDA_BRAND_N"]   = out["_FDA_BRAND"].map(normalize_text)
    out["_FDA_GENERIC_N"] = out["_FDA_GENERIC"].map(normalize_text)
    out = out.dropna(subset=["_FDA_BRAND_N"]).drop_duplicates(subset=["_FDA_BRAND_N"])
    return out

def load_who_atc(path: Path) -> pd.DataFrame:
    """
    Load WHO ATC file. Some exports contain only ATC_CODE/ATC_NAME; some may include a generic column.
    If a generic column exists, we compute _ATC_GENERIC_BASE for high-confidence joins.
    """
    df = load_csv_any(path)
    df = ensure_unique_columns(normalize_headers(df))
    atc_code = pick_col(df, ["ATC_CODE", "ATC", "ATC CODE"], must=True, label="ATC code")
    atc_name = pick_col(df, ["ATC_NAME", "ATC NAME", "DESCRIPTION"], must=False)
    gen_col  = pick_col(df, ["GENERIC", "GENERIC_NAME", "INGREDIENT", "SUBSTANCE", "MOLECULE", "DRUG_NAME"], must=False)

    cols = [atc_code] + ([atc_name] if atc_name else []) + ([gen_col] if gen_col else [])
    out = df[cols].copy()
    out = out.rename(columns={
        atc_code: "ATC_CODE",
        (atc_name or "ATC_NAME"): "ATC_NAME",
        (gen_col or "GENERIC"): "_ATC_GENERIC"
    })

    if gen_col:
        out["_ATC_GENERIC_N"]    = out["_ATC_GENERIC"].map(normalize_text)
        out["_ATC_GENERIC_BASE"] = out["_ATC_GENERIC_N"].map(normalize_salts_base)
        out = out.dropna(subset=["_ATC_GENERIC_BASE"]).drop_duplicates(subset=["_ATC_GENERIC_BASE", "ATC_CODE"])
    else:
        out = out.drop(columns=["_ATC_GENERIC"], errors="ignore")
    return out

# ---------- FDA brand tokens from parentheses ----------
PAREN_PAT = re.compile(r"\(([^)]+)\)")

def extract_parenthetical_brands(text: str) -> List[str]:
    """Return a list of candidate brand tokens found in parentheses."""
    if not isinstance(text, str) or not text:
        return []
    raw = []
    for m in PAREN_PAT.finditer(text):
        chunk = m.group(1)
        parts = re.split(r"[;,/|]", chunk)  # split on common separators
        raw.extend(p.strip() for p in parts if p.strip())
    seen, out = set(), []
    for tok in raw:
        up = tok.upper()
        if up not in seen:
            seen.add(up)
            out.append(tok)
    return out

def enrich_esoa_parentheses_with_fda_brand(
    df: pd.DataFrame,
    *,
    desc_col: str,
    brand_map: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    If eSOA descriptions contain brands in parentheses, map them via FDA brand→generic.
    Writes/overrides:
      - _PAREN_BRANDS          (raw tokens)
      - _PAREN_BRANDS_N        (normalized tokens)
      - _GENERIC_FROM_BRAND    (from brand map; preserves existing if already set)
      - _GENERIC_BEST          (prefers explicit generic columns; else brand-mapped)
      - _GENERIC_BASE          (salt-stripped from _GENERIC_BEST)
    """
    res = df.copy()
    if desc_col not in res.columns or brand_map is None or brand_map.empty:
        return res

    res["_PAREN_BRANDS"] = res[desc_col].apply(extract_parenthetical_brands)
    res["_PAREN_BRANDS_N"] = res["_PAREN_BRANDS"].apply(lambda toks: [normalize_text(t) for t in toks])

    # brand→generic lookup
    brand_dict = dict(zip(brand_map["_FDA_BRAND_N"], brand_map["_FDA_GENERIC_N"]))

    def map_first_brand(tokens_n):
        for t in tokens_n or []:
            if t in brand_dict:
                return brand_dict[t]
        return ""

    mapped_generic = res["_PAREN_BRANDS_N"].apply(map_first_brand)

    if "_GENERIC_FROM_BRAND" not in res.columns:
        res["_GENERIC_FROM_BRAND"] = mapped_generic
    else:
        res["_GENERIC_FROM_BRAND"] = res["_GENERIC_FROM_BRAND"].where(
            res["_GENERIC_FROM_BRAND"].astype(str).str.len() > 0,
            mapped_generic
        )

    # Choose best generic (explicit generic columns take precedence)
    generic_source_cols = [c for c in ["GENERIC","GENERIC_NAME","MOLECULE","DRUG_NAME","ACTIVE_INGREDIENT"] if c in res.columns]
    def best_generic(row):
        for c in generic_source_cols:
            val = row.get(c, "")
            if isinstance(val, str) and val.strip():
                return normalize_text(val)
        gf = row.get("_GENERIC_FROM_BRAND", "")
        return gf if isinstance(gf, str) else ""

    res["_GENERIC_BEST"] = res.apply(best_generic, axis=1)
    res["_GENERIC_BASE"] = res["_GENERIC_BEST"].map(normalize_salts_base)
    return res

# ---------- Enrichment ----------
def enrich_with_brand_and_atc(
    df: pd.DataFrame,
    *,
    brand_map: Optional[pd.DataFrame],
    who_atc: Optional[pd.DataFrame],
    brand_source_cols: List[str],
    generic_source_cols: List[str]
) -> pd.DataFrame:
    """
    - Detect brand candidate and map to generic via FDA.
    - Compute _GENERIC_BEST and _GENERIC_BASE.
    - Normalize/attach _ROUTE_N.
    - Attach WHO ATC when there is a safe join key; otherwise **do not** erase existing ATC columns.
    """
    res = df.copy()

    def first_nonempty(row, cols):
        for c in cols:
            if c in row and pd.notna(row[c]) and str(row[c]).strip() != "":
                return str(row[c])
        return ""

    # brand candidate
    res["_BRAND_CANDIDATE"]   = res.apply(lambda r: first_nonempty(r, brand_source_cols), axis=1)
    res["_BRAND_CANDIDATE_N"] = res["_BRAND_CANDIDATE"].map(normalize_text)

    # FDA brand -> generic
    if brand_map is not None and not brand_map.empty:
        res = res.merge(
            brand_map[["_FDA_BRAND_N", "_FDA_GENERIC_N"]],
            left_on="_BRAND_CANDIDATE_N",
            right_on="_FDA_BRAND_N",
            how="left"
        )
        res["_GENERIC_FROM_BRAND"] = res["_FDA_GENERIC_N"].fillna("")

    # choose best generic (explicit generic cols first, else map)
    def best_generic(row):
        for c in generic_source_cols:
            if c in row and pd.notna(row[c]) and str(row[c]).strip() != "":
                return normalize_text(row[c])
        gf = row.get("_GENERIC_FROM_BRAND", "")
        return gf if gf else ""

    res["_GENERIC_BEST"] = res.apply(best_generic, axis=1)
    res["_GENERIC_BASE"] = res["_GENERIC_BEST"].map(normalize_salts_base)

    # Ensure normalized route
    res = _ensure_route_n(res)

    # WHO ATC:
    # Prefer join on _GENERIC_BASE <-> _ATC_GENERIC_BASE if WHO supplies it.
    # Otherwise, do nothing destructive—preserve existing ATC columns on res.
    if who_atc is not None and not who_atc.empty:
        if "_ATC_GENERIC_BASE" in who_atc.columns:
            who_mini = who_atc.drop_duplicates(subset=["_ATC_GENERIC_BASE"])[["_ATC_GENERIC_BASE"] + [c for c in ["ATC_CODE","ATC_NAME"] if c in who_atc.columns]]
            res = res.merge(
                who_mini,
                left_on="_GENERIC_BASE",
                right_on="_ATC_GENERIC_BASE",
                how="left"
            )
        else:
            # No safe join key provided by WHO; don't force an index-based merge that could misalign rows.
            # If res already has ATC columns, keep them; otherwise leave enrichment for later steps.
            pass

    return res