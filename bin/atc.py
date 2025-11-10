# atc.py
def index_who(who_df):
    # build maps: ingredient -> {route -> [atc_codes]}
    # keep 4th/5th-level helpers
    ...

def infer_atc(crow: CanonRow, who_index) -> CanonRow:
    if crow.source != "ESOA": return crow
    if not crow.canon_ingredients: return crow
    cands = lookup_atc_by_ingredients_and_route(who_index, crow.canon_ingredients, crow.canon_route)
    if cands:
        best = choose_best(cands)  # prefer 5th-level if unique; else 4th
        crow.canon_atc_code = best
        crow.canon_atc_l4 = best[:4]
    return crow