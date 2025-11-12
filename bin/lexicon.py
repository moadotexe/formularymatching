# lexicon.py
from __future__ import annotations

from typing import Any, Iterable, Optional, Type, List, Dict, Sequence, Tuple, Set, TYPE_CHECKING

from models import Vocabs


_pyac: Optional[Any] = None
_LocalAho: Optional[Type] = None
_USE_PYAC: bool = False
# ---------------------------
# Normalization & filtering
# ---------------------------
import unicodedata
import regex as re

_WS = re.compile(r"\s+")

def _normalize_text(s: str) -> str:
    """
    NFKC, lowercase, hyphen-like → space, collapse spaces.
    Use the same normalizer everywhere (patterns & text).
    """
    s = unicodedata.normalize("NFKC", s or "")
    # Normalize hyphen family to plain hyphen, then to space
    s = (s.replace("\u2010", "-").replace("\u2011", "-")
           .replace("\u2012", "-").replace("\u2013", "-").replace("\u2014", "-"))
    s = s.replace("-", " ")
    s = s.lower().strip()
    s = _WS.sub(" ", s)
    return s

def _is_good_str(v: Any) -> bool:
    return isinstance(v, str) and v.strip() != ""

def series_to_norm_str_set(values: Iterable[Any]) -> Set[str]:
    """
    Filters out None/NaN/non-strings, normalizes, and dedupes → Set[str].
    Safe for pandas Series .tolist() or any iterable.
    """
    out: Set[str] = set()
    for v in values:
        if _is_good_str(v):
            t = _normalize_text(v)  # type: ignore[arg-type]
            if t:
                out.add(t)
    return out

def map_brand_to_generic(brand_list: Iterable[Any], generic_list: Iterable[Any]) -> Dict[str, List[str]]:
    tmp: Dict[str, Set[str]] = {}
    for b, g in zip(brand_list, generic_list):
        if _is_good_str(b) and _is_good_str(g):
            b2 = _normalize_text(b)  # type: ignore[arg-type]
            g2 = _normalize_text(g)  # type: ignore[arg-type]
            tmp.setdefault(b2, set()).add(g2)
    return {k: sorted(v) for k, v in tmp.items()}


# ---------------------------
# AC builder (defensive)
# ---------------------------
def build_ac(words: Sequence[str] | Iterable[str]) -> Any:
    """
    Build an AC automaton from a sequence/iterable of strings.
    Ignores empties after normalization; deduplicates.
    Returns a pyahocorasick.Automaton if available, else your local Aho.
    """
    seen: Set[str] = set()
    # Normalize & dedupe
    norm_words: List[str] = []
    for w in words:
        t = _normalize_text(w)
        if not t or t in seen:
            continue
        seen.add(t)
        norm_words.append(t)

    if _USE_PYAC and _pyac is not None:
        A = _pyac.Automaton()
        for t in norm_words:
            A.add_word(t, t)
        A.make_automaton()
        return A
    elif _LocalAho is not None:
        items = [(t, t) for t in norm_words]
        A = _LocalAho(items)
        # optional: adapter for .iter if your class only has .finditer
        if not hasattr(A, "iter") and hasattr(A, "finditer"):
            def _iter(text: str):
                t_norm = _normalize_text(text)
                #normalize
                for h in A.finditer(t_norm):
                    #mimick ahocrasick's (end_index, payload)
                    yield (h.end - 1, h.pat)
                setattr(A, "iter", _iter)
            return A
        
        #Fail
        raise RuntimeError("No Aho-Corasick backend available"
                           "Install ahocorasick.py"
                           )


# ---------------------------
# Main builder
# ---------------------------
def build_vocabs(pnf_df, fda_brand_df, who_df, food_df) -> "Vocabs":
    """
    Assemble lexicons + automatons and return a Vocabs instance.
    This function is type-safe (no None in collections passed to sorted/build_ac).
    """
    # --- Ingredient lexicon from multiple sources ---
    ingred: Set[str] = set()
    if pnf_df is not None and "Molecule" in pnf_df.columns:
        ingred |= series_to_norm_str_set(pnf_df["Molecule"].tolist())
    if fda_brand_df is not None and "generic_name" in fda_brand_df.columns:
        ingred |= series_to_norm_str_set(fda_brand_df["generic_name"].tolist())
    if who_df is not None and "atc_name" in who_df.columns:
        ingred |= series_to_norm_str_set(who_df["atc_name"].tolist())

    # Optional synonym normalization (expand as needed)
    synonym_map: Dict[str, str] = {
        "acetaminophen": "paracetamol",
        "epinephrine": "adrenaline",
        "albuterol": "salbutamol",
        "lignocaine": "lidocaine",
    }
    ingred = {synonym_map.get(x, x) for x in ingred}

    # --- Brand lexicon ---
    brands: Set[str] = set()
    if fda_brand_df is not None and "brand_name" in fda_brand_df.columns:
        brands |= series_to_norm_str_set(fda_brand_df["brand_name"].tolist())

    # --- Food / supplement tokens ---
    food_tokens: Set[str] = set()
    if food_df is not None and "brand_name" in food_df.columns:
        food_tokens |= series_to_norm_str_set(food_df["brand_name"].tolist())
    if food_df is not None and "product_name" in food_df.columns:
        food_tokens |= series_to_norm_str_set(food_df["product_name"].tolist())

    # --- Build automatons (pure List[str] → AC) ---
    ingredient_list: List[str] = sorted(ingred)
    brand_list: List[str] = sorted(brands)
    food_list: List[str] = sorted(food_tokens)

    ingredient_ac = build_ac(ingredient_list)
    brand_ac      = build_ac(brand_list)
    food_ac       = build_ac(food_list)

    # --- Form/Route/Release/Unit maps (expand per your data profiling) ---
    form_map: Dict[str, Tuple[str, str]] = {
        "tab": ("tablet","solid_oral"),
        "tablet": ("tablet","solid_oral"),
        "cap": ("capsule","solid_oral"),
        "capsule": ("capsule","solid_oral"),
        "inj": ("injection solution","parenteral_solution"),
        "injection": ("injection solution","parenteral_solution"),
        "solution": ("solution","liquid_oral"),
        "suspension": ("suspension","liquid_oral"),
        "syrup": ("syrup","liquid_oral"),
        "cream": ("cream","topical_semisolid"),
        "ointment": ("ointment","topical_semisolid"),
        "gel": ("gel","topical_semisolid"),
        "eye drops": ("eye drops","ophthalmic_otic_nasal"),
        "ophthalmic": ("eye drops","ophthalmic_otic_nasal"),
        "nasal": ("nasal spray","ophthalmic_otic_nasal"),
        "spray": ("spray","respiratory"),
        "nebule": ("nebulizer solution","respiratory"),
        # add local variants like "fc tab", "caplet", "odt", etc.
    }

    route_map: Dict[str, str] = {
        "po": "oral", "oral": "oral",
        "iv": "intravenous", "intravenous": "intravenous",
        "im": "intramuscular", "intramuscular": "intramuscular",
        "sc": "subcutaneous", "subcut": "subcutaneous",
        "topical": "topical",
        "ophthalmic": "ophthalmic",
        "otic": "otic",
        "nasal": "nasal",
        "inhalation": "inhalation",
        "respiratory": "inhalation",
        "rectal": "rectal",
        "vaginal": "vaginal",
    }

    release_map: Dict[str, str] = {
        "er":"ER","xr":"ER","cr":"ER","sr":"ER","xl":"ER","mr":"ER","od":"ER","odt":"ER",
        "dr":"DR","ec":"DR",
    }

    unit_map: Dict[str, float] = {"g":1000.0,"mg":1.0,"mcg":0.001,"ml":1.0,"l":1000.0}

    # --- Brand → Generic map (avoids None) ---
    brand_to_generic: Dict[str, List[str]] = {}
    if fda_brand_df is not None and {"brand_name", "generic_name"} <= set(fda_brand_df.columns):
        brand_to_generic = map_brand_to_generic(
            fda_brand_df["brand_name"].tolist(),
            fda_brand_df["generic_name"].tolist()
        )

    # --- Return Vocabs (runtime import avoids circulars) ---
    from models import Vocabs as _Vocabs  # local import at runtime

    return _Vocabs(
        ingredient_lexicon=ingredient_list,
        ingredient_ac=ingredient_ac,
        brand_lexicon=brand_list,
        brand_ac=brand_ac,
        form_map=form_map,
        route_map=route_map,
        release_map=release_map,
        unit_map=unit_map,
        food_lexicon=food_list,
        food_ac=food_ac,
        brand_to_generic=brand_to_generic,
        synonym_map=synonym_map,
    )