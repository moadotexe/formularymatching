# lexicon.py
import ahocorasick

def build_ac(words: list[str]) -> ahocorasick.Automaton:
    A = ahocorasick.Automaton()
    for w in sorted(set(w.strip().lower() for w in words), key=len):
        if not w: continue
        A.add_word(w, w)
    A.make_automaton()
    return A

def build_vocabs(pnf_df, fda_brand_df, who_df, food_df) -> Vocabs:
    # ingredient set
    ingred = set()
    ingred |= set(pnf_df["Molecule"].dropna().str.lower())
    ingred |= set(fda_brand_df["generic_name"].dropna().str.lower())
    ingred |= set(who_df["atc_name"].dropna().str.lower())
    # expand synonyms (hardcode or load table)
    synonym_map = {
        "acetaminophen": "paracetamol",
        "epinephrine": "adrenaline",
        "albuterol": "salbutamol",
        "lignocaine": "lidocaine",
        # ...
    }
    ingred_norm = {synonym_map.get(x, x) for x in ingred}

    brand = set(fda_brand_df["brand_name"].dropna().str.lower())
    food_names = set(food_df["brand_name"].dropna().str.lower()) | set(food_df["product_name"].dropna().str.lower())

    brand_to_generic = (fda_brand_df.groupby("brand_name")["generic_name"]
                        .apply(lambda s: sorted({g.lower() for g in s if isinstance(g,str)})).to_dict())

    form_map = {  # minimal seeds; expand from data profiling
        "tab": ("tablet","solid_oral"), "tablet": ("tablet","solid_oral"),
        "cap": ("capsule","solid_oral"), "capsule": ("capsule","solid_oral"),
        "inj": ("injection solution","parenteral_solution"),
        "solution": ("solution","liquid_oral"), "suspension": ("suspension","liquid_oral"),
        "cream": ("cream","topical_semisolid"), "ointment": ("ointment","topical_semisolid"),
        "eye drops": ("eye drops","ophthalmic_otic_nasal"),
        # ...
    }
    route_map = {
        "po":"oral","oral":"oral","iv":"intravenous","intravenous":"intravenous",
        "im":"intramuscular","sc":"subcutaneous","topical":"topical","ophthalmic":"ophthalmic",
        "otic":"otic","nasal":"nasal","inhalation":"inhalation","rectal":"rectal","vaginal":"vaginal",
    }
    release_map = {"er":"ER","xr":"ER","cr":"ER","sr":"ER","xl":"ER","mr":"ER","od":"ER","odt":"ER",
                   "dr":"DR","ec":"DR"}
    unit_map = {"g":1000.0,"mg":1.0,"mcg":0.001,"ml":1.0,"l":1000.0}

    return Vocabs(
        ingredient_lexicon=sorted(ingred_norm),
        ingredient_ac=build_ac(list(ingred_norm)),
        brand_lexicon=sorted(brand),
        brand_ac=build_ac(list(brand)),
        form_map=form_map,
        route_map=route_map,
        release_map=release_map,
        unit_map=unit_map,
        food_lexicon=sorted(food_names),
        food_ac=build_ac(list(food_names)),
        brand_to_generic=brand_to_generic,
        synonym_map=synonym_map,
    )
