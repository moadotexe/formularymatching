# normalize.py
from __future__ import annotations
from typing import Iterable, Optional
import unicodedata, regex as re
import pandas as pd

__all__ = [
    "normalize_text",
    "normalize_headers",
    "ensure_unique_columns",
    "pick_col",
]

_WS = re.compile(r"\s+")
_NONWORD = re.compile(r"[^0-9a-zA-Z_]+")

def normalize_text(s: Optional[str]) -> str:
    """NFKC → hyphen family → space → lowercase → collapse spaces."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = (s.replace("\u2010","-").replace("\u2011","-")
           .replace("\u2012","-").replace("\u2013","-").replace("\u2014","-"))
    s = s.replace("-", " ").lower().strip()
    s = _WS.sub(" ", s)
    return s

def _norm_col_name(c: str) -> str:
    c = unicodedata.normalize("NFKC", str(c)).strip().lower()
    c = c.replace("/", " ").replace("-", " ").replace(".", " ")
    c = _WS.sub(" ", c).strip()
    c = c.replace(" ", "_")
    c = _NONWORD.sub("_", c)
    return c

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + snake_case all column names (no duplicates resolution here)."""
    df = df.copy()
    df.columns = [_norm_col_name(c) for c in df.columns]
    return df

def ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names unique by suffixing _2, _3, ... when needed."""
    df = df.copy()
    seen = {}
    cols = []
    for c in df.columns:
        base = c
        n = seen.get(base, 0) + 1
        seen[base] = n
        cols.append(base if n == 1 else f"{base}_{n}")
    df.columns = cols
    return df

def pick_col(df: pd.DataFrame, candidates: Iterable[str], *, must: bool = False, label: str = "") -> Optional[str]:
    """
    Return the first matching column name in df among candidates.
    Candidates may be raw names; we normalize them the same way as headers.
    If must=True and none is found, raise KeyError with a helpful message.
    """
    # Build a map from normalized->actual
    actual = { _norm_col_name(c): c for c in df.columns }
    for cand in candidates:
        key = _norm_col_name(cand)
        if key in actual:
            return actual[key]
    if must:
        pretty = ", ".join(candidates)
        raise KeyError(f"Required column not found for {label or 'column'}; tried: {pretty}. "
                       f"Available: {list(df.columns)}")
    return None