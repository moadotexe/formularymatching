# parse_rows.py
from models import CanonRow
from normalize import clean, canon_ingredient_tokens
from parse_strength import parse_strength

def canon_pnf_row(row, vocabs) -> CanonRow:
    mol = clean(row["Molecule"])
    tech = clean(row["Technical Specifications"])
    route_raw = clean(row.get("Route",""))
    atc = (row.get("ATC Code") or "").strip().upper()

    ingredients = canon_ingredient_tokens(mol, vocabs.synonym_map)
    ing_key = "+".join(ingredients)

    # detect form/route/release via token presence (simple contains → map)
    form, fam = detect_form(tech, vocabs.form_map)          # implement: pick most specific
    route = vocabs.route_map.get(route_raw, route_raw or route_from_form(form))
    release = detect_release(tech, vocabs.release_map)      # trivial token map

    strength = parse_strength(tech, fam or "", vocabs.unit_map)
    pack_qty, pack_uom = extract_pack(tech)                 # simple regexes per blueprint

    return CanonRow(
        source="PNF",
        source_id=str(row.name),
        source_text=row["Technical Specifications"],
        canon_brand=None,
        canon_ingredients=ingredients,
        canon_ingredients_key=ing_key,
        canon_route=route,
        canon_form=form,
        canon_form_family=fam,
        canon_release=release,
        canon_strength=strength,
        canon_pack_size_qty=pack_qty,
        canon_pack_size_uom=pack_uom,
        canon_atc_code=atc or None,
        canon_atc_l4=(atc[:4] if atc else None),
        is_official=True,
        is_food_or_supp=False
    )

def canon_esoa_row(row, vocabs) -> CanonRow|None:
    if int(row["IS_OFFICIAL"]) != 1: return None
    desc = clean(row["DESCRIPTION"])

    if matches_food(desc, vocabs.food_ac):  # implement AC match wrapper
        return None

    brand = ac_longest_leftmost(desc, vocabs.brand_ac)
    desc_for_ing = replace_brand_with_generic(desc, brand, vocabs.brand_to_generic) if brand else desc
    ingredients = extract_ingredients(desc_for_ing, vocabs) # AC hits → normalized
    ing_key = "+".join(ingredients)

    form, fam = detect_form(desc, vocabs.form_map)
    route = detect_route(desc, vocabs.route_map, form)
    release = detect_release(desc, vocabs.release_map)

    strength = parse_strength(desc, fam or "", vocabs.unit_map)
    pack_qty, pack_uom = extract_pack(desc)

    return CanonRow(
        source="ESOA",
        source_id=str(row["ITEM_NUMBER"]),
        source_text=row["DESCRIPTION"],
        canon_brand=brand,
        canon_ingredients=ingredients,
        canon_ingredients_key=ing_key,
        canon_route=route,
        canon_form=form,
        canon_form_family=fam,
        canon_release=release,
        canon_strength=strength,
        canon_pack_size_qty=pack_qty,
        canon_pack_size_uom=pack_uom,
        canon_atc_code=None, canon_atc_l4=None,
        is_official=True, is_food_or_supp=False
    )