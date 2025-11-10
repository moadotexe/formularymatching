# config.py
from pydantic import BaseSettings, Field
from typing import Dict

class Settings(BaseSettings):
    # ---- Paths
    pnf_path: str = "pnf.csv"
    esoa_path: str = "esoa_clean.csv"
    brand_path: str = "fda_brand_map.csv"
    food_path: str = "fda_food_products.csv"
    who_path: str = "WHO ATC-DDD 2024-07-31.csv"
    out_path: str = "matches.json"

    # ---- Prefilters
    require_is_official: bool = True
    exclude_food_products: bool = True

    # ---- Parsing
    strip_salts: bool = True
    default_basis_topical: str = "w/w"   # for % strengths
    default_basis_liquid: str = "w/v"

    # ---- Tolerances
    tol_solid: float = 0.05              # ±5%
    tol_liquid: float = 0.05             # ±5%

    # ---- Blocking
    enable_blocks: Dict[str, bool] = {
        "BK1_ING": True, "BK2_ATC": True, "BK3_FORM_ROUTE": True,
        "BK4_STRENGTH": True, "BK5_PREFIX": True, "BK6_BRAND_ALIAS": True
    }
    prefix_len: int = 10

    # ---- Scoring weights
    w_ingredient: float = 0.40
    w_strength: float  = 0.25
    w_form: float      = 0.12
    w_route: float     = 0.08
    w_atc: float       = 0.10
    w_text_gen: float  = 0.10
    w_text_brand: float= 0.06
    penalty_release_mismatch: float = 0.10

    # ---- Thresholds
    high_conf: float = 0.85
    review_lo: float = 0.75
    tie_margin: float = 0.10

    # ---- Performance / Logging
    chunk_size: int = 100000
    serialize_tries: bool = False
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_prefix = "DRUGMATCH_"

settings = Settings()