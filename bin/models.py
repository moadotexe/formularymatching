# models.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

@dataclass
class StrengthStruct:
    type: str                              # "per_unit" | "per_volume" | "percent" | "iu" | "combo" | "per_container"
    per_unit_uom: Optional[str] = None     # e.g., "tablet", "capsule"
    value_mg: Optional[float] = None
    value_mg_per_ml: Optional[float] = None
    value_pct: Optional[float] = None
    basis: Optional[str] = None            # "w/w" | "w/v"
    value_iu: Optional[float] = None
    per: Optional[str] = None              # "dose" | "actuation" | "ml" | "unit"
    components: Optional[List[Dict[str, Any]]] = None
    container: Optional[str] = None
    bucket: Optional[str] = None
    assumed: bool = False
    basis_assumed: bool = False
    component_alignment: Optional[str] = None

@dataclass
class CanonRow:
    source: str                             # "PNF" | "ESOA"
    source_id: str
    source_text: str
    canon_brand: Optional[str]
    canon_ingredients: List[str]
    canon_ingredients_key: str
    canon_route: Optional[str]
    canon_form: Optional[str]
    canon_form_family: Optional[str]
    canon_release: Optional[str]
    canon_strength: Optional[StrengthStruct]
    canon_pack_size_qty: Optional[float]
    canon_pack_size_uom: Optional[str]
    canon_atc_code: Optional[str]
    canon_atc_l4: Optional[str]
    is_official: bool = True
    is_food_or_supp: bool = False

@dataclass
class Vocabs:
    ingredient_lexicon: List[str]
    ingredient_ac: Any                      # AC automaton
    brand_lexicon: List[str]
    brand_ac: Any
    form_map: Dict[str, Tuple[str, str]]    # raw -> (canon_form, family)
    route_map: Dict[str, str]               # raw -> canon_route
    release_map: Dict[str, str]             # token -> "IR"/"ER"/"DR"
    unit_map: Dict[str, float]              # conversions
    food_lexicon: List[str]
    food_ac: Any
    brand_to_generic: Dict[str, List[str]]
    synonym_map: Dict[str, str]             # e.g., "acetaminophen"->"paracetamol"
