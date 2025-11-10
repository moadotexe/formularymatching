# parse_strength.py
import regex as re
from models import StrengthStruct

NUM = r"(?P<num>\d+(?:\.\d+)?)"
UNIT = r"(?P<unit>mg|g|mcg|iu)"
VOL  = r"(?P<vol>\d+(?:\.\d+)?)\s*(?P<volu>ml|l)"
PCT  = r"(?P<pct>\d+(?:\.\d+)?)\s*%"

def _to_mg(val: float, unit: str, unit_map) -> float:
    if unit.lower()=="iu": return val  # handled separately
    if unit.lower() not in unit_map: return val
    if unit.lower()=="g": return val*unit_map["g"]
    if unit.lower()=="mg": return val
    if unit.lower() in ("mcg",): return val*unit_map["mcg"]
    return val

def bucket_per_unit(mg: float, per: str) -> str:
    if mg < 100: r = round(mg, 0)         # 1 mg steps
    elif mg <= 500: r = round(mg/25)*25
    else: r = round(mg/50)*50
    return f"{int(r)}mg_per_{per}"

def bucket_per_ml(mgml: float) -> str:
    r = round(mgml*2)/2  # nearest 0.5 mg/mL
    r_str = f"{r}".rstrip("0").rstrip(".")
    return f"{r_str}mg_per_mL"

def parse_strength(text: str, form_family: str, unit_map: dict) -> StrengthStruct|None:
    t = text

    # 1) combos
    combo_pat = re.compile(rf"{NUM}\s*({UNIT})\s*/\s*{NUM}\s*({UNIT})(?:\s*/\s*{NUM}\s*({UNIT}))?(?:\s*(?:per|/)\s*(?P<per>ml|dose|actuation|tablet|capsule|unit))?")
    m = combo_pat.search(t)
    if m:
        comps = []
        for i in (1,3,5):  # num positions
            if m.group(i):
                v = float(m.captures("num")[(i-1)//2])
        # Simpler: iterate captures
        nums = [float(x) for x in m.captures("num")]
        units= m.captures("unit")
        for v,u in zip(nums,units):
            mg = _to_mg(v,u,unit_map)
            comps.append({"value_mg": mg, "unit":"mg"})
        per = (m.group("per") or ("tablet" if "solid" in form_family else "ml"))
        bucket = "+".join(str(int(round(c["value_mg"]))) for c in comps) + f"mg_per_{per}"
        return StrengthStruct(type="combo", components=comps, per=per, bucket=bucket)

    # 2) explicit concentration
    conc_pat = re.compile(rf"{NUM}\s*({UNIT})\s*/\s*{VOL}")
    m = conc_pat.search(t)
    if m:
        amt = _to_mg(float(m.group("num")), m.group("unit"), unit_map)
        vol_ml = float(m.group("vol")) * (1.0 if m.group("volu")=="ml" else 1000.0)
        mgml = amt/vol_ml if vol_ml else None
        return StrengthStruct(type="per_volume", value_mg_per_ml=mgml, bucket=bucket_per_ml(mgml))

    # 3) percent
    pct_pat = re.compile(PCT)
    m = pct_pat.search(t)
    if m:
        pct = float(m.group("pct"))
        basis = "w/w" if form_family in ("topical_semisolid",) else "w/v"
        # equivalent:
        eq = pct*10.0  # mg per mL or mg per g
        bucket = f"{round(pct,1)}pct_{'topical' if 'topical' in form_family else 'liquid'}"
        return StrengthStruct(type="percent", value_pct=pct, basis=basis, bucket=bucket)

    # 4) IU per unit/dose
    iu_pat = re.compile(rf"{NUM}\s*iu(?:\s*(?:per|/)\s*(?P<per>dose|actuation|tablet|capsule|unit))?")
    m = iu_pat.search(t)
    if m:
        per = m.group("per") or "unit"
        return StrengthStruct(type="iu", value_iu=float(m.group("num")), per=per, bucket=f"{int(float(m.group('num')))}iu_per_{per}")

    # 5) per-unit solids
    solid_pat = re.compile(rf"{NUM}\s*(mg|g|mcg)\s*(tablet|tab|capsule|cap|caplet|suppository|lozenge|odt)")
    m = solid_pat.search(t)
    if m:
        mg = _to_mg(float(m.group("num")), m.group(1), unit_map)
        per = {"tab":"tablet","cap":"capsule"}.get(m.group(2), m.group(2))
        return StrengthStruct(type="per_unit", value_mg=mg, per_unit_uom=per, bucket=bucket_per_unit(mg, per))

    # 6) parenteral powder
    pow_pat = re.compile(rf"{NUM}\s*(mg|g|mcg).*?(powder for (?:injection|solution))")
    m = pow_pat.search(t)
    if m:
        mg = _to_mg(float(m.group("num")), m.group(1), unit_map)
        return StrengthStruct(type="per_container", value_mg=mg, container="vial", bucket=f"{int(round(mg))}mg_per_vial")

    # 7) fallback numeric
    num_pat = re.compile(rf"{NUM}\s*(mg|g|mcg)")
    m = num_pat.search(t)
    if m:
        mg = _to_mg(float(m.group("num")), m.group(1), unit_map)
        # infer per from form family
        if "solid" in form_family:
            return StrengthStruct(type="per_unit", value_mg=mg, per_unit_uom="unit", bucket=bucket_per_unit(mg,"unit"), assumed=True)
        return StrengthStruct(type="per_unit", value_mg=mg, per_unit_uom="unit", bucket=bucket_per_unit(mg,"unit"), assumed=True)

    return None