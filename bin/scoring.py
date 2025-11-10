# scoring.py
from rapidfuzz import fuzz

def ingredient_equal(p: CanonRow, e: CanonRow) -> float:
    return 1.0 if p.canon_ingredients_key == e.canon_ingredients_key and p.canon_ingredients_key else 0.0

def strength_similarity(p: CanonRow, e: CanonRow) -> float:
    # compare normalized StrengthStructs; apply tolerances by type
    ...

def form_score(p, e) -> float:
    if p.canon_form == e.canon_form and p.canon_form: return 1.0
    if p.canon_form_family and e.canon_form_family and p.canon_form_family == e.canon_form_family: return 0.6
    return 0.0

def route_score(p, e) -> float:
    return 1.0 if p.canon_route and p.canon_route == e.canon_route else 0.0

def atc_score(p, e) -> float:
    if p.canon_atc_code and e.canon_atc_code and p.canon_atc_code == e.canon_atc_code: return 1.0
    if p.canon_atc_l4 and e.canon_atc_l4 and p.canon_atc_l4 == e.canon_atc_l4: return 0.5
    return 0.0

def text_sim_generic(p, e) -> float:
    pg = " ".join(p.canon_ingredients) or ""
    eg = " ".join(e.canon_ingredients) or ""
    return fuzz.token_set_ratio(pg, eg)/100.0 if (pg and eg) else 0.0

def text_sim_brand(p, e) -> float:
    if not e.canon_brand: return 0.0
    return fuzz.QRatio((p.canon_ingredients or [""])[0], e.canon_brand)/100.0  # loose assist

def release_penalty(p, e) -> float:
    if not p.canon_release or not e.canon_release: return 0.0
    return 0.10 if p.canon_release != e.canon_release else 0.0

def final_score(p, e) -> float:
    I = ingredient_equal(p,e)
    D = strength_similarity(p,e)
    F = form_score(p,e)
    R = route_score(p,e)
    A = atc_score(p,e)
    G = text_sim_generic(p,e)
    B = text_sim_brand(p,e)
    score = 0.40*I + 0.25*D + 0.12*F + 0.08*R + 0.10*A + 0.10*G + 0.06*B
    score -= release_penalty(p,e)
    return max(0.0, min(1.0, score))

def classify(sorted_scores):  # list of (idx, score)
    HIGH, REVIEW = 0.85, 0.75
    if not sorted_scores: return "no_match"
    if sorted_scores[0][1] >= HIGH and (len(sorted_scores)==1 or sorted_scores[0][1]-sorted_scores[1][1] >= 0.10):
        return "auto_match"
    if sorted_scores[0][1] >= REVIEW:
        return "needs_review"
    return "no_match"