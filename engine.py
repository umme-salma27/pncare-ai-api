"""PNCare-AI prediction engine (no user interface).

Same logic as the Streamlit app (PNC_APP/app.py): the same saved soft-voting pipeline, the same input mapping,
the same 0.27 threshold, the same WHO-referenced actions and the same exact Shapley explanation over 10 input groups.
"""
import json
import math
import os
import sys
import warnings
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pnc_core  # noqa: E402,F401  (the saved pipeline is built from these classes)

warnings.filterwarnings("ignore")
MODEL_PATH = os.path.join(HERE, "pnc_app_model.joblib")
REF_PATH = os.path.join(HERE, "pnc_app_reference.json")
MIN_PLACE_SHARE = 0.02

PLACE = {
    "Home": "home",
    "Upazila Health Complex": "upazila health complex",
    "District hospital": "district hospital",
    "Medical college hospital": "medical college hospital",
    "Specialized government hospital": "specialized govt hospital",
    "Maternal and Child Welfare Centre (MCWC)": "mcwc",
    "Union Health and Family Welfare Centre": "uh & family welfare center",
    "Community clinic": "community clinic",
    "Other public facility": "other public sector",
    "Private hospital": "private hospital",
    "Private clinic": "private clinic",
    "Private medical college hospital": "private medical college hospital",
    "Other private medical facility": "other private medical",
    "NGO static clinic": "ngo static clinic",
    "Delivery hut": "delivery hut",
    "Other place": "other",
}
EDUCATION = {"No education": "no education", "Primary": "primary", "Secondary or higher": "secondary"}
DIVISION = {d.title(): d for d in ["barishal", "chattogram", "dhaka", "khulna", "mymensingh", "rajshahi", "rangpur", "sylhet"]}
RESIDENCE = {"Rural": "rural", "Urban": "urban"}
MEDIA = {"Not at all": "not at all", "Less than weekly": "less than once a week", "At least weekly": "at least once a week"}
MEDIA_SCORE = {"not at all": 0, "less than once a week": 1, "at least once a week": 2}
YES_NO = {"Yes": True, "No": False}

INPUTS = [
    dict(key="place", csv="place_of_delivery", label="Place of delivery", kind="cat", options=PLACE),
    dict(key="anc", csv="anc_visits", label="ANC visits", kind="num", min=0, max=20, unit="visits"),
    dict(key="age", csv="mother_age_at_birth", label="Mother's age at birth", kind="num", min=13, max=49, unit="years"),
    dict(key="edu", csv="education", label="Education", kind="cat", options=EDUCATION),
    dict(key="division", csv="division", label="Division", kind="cat", options=DIVISION),
    dict(key="residence", csv="residence", label="Residence", kind="cat", options=RESIDENCE),
    dict(key="newspaper", csv="newspaper", label="Newspaper", kind="cat", options=MEDIA),
    dict(key="radio", csv="radio", label="Radio", kind="cat", options=MEDIA),
    dict(key="tv", csv="tv", label="Television", kind="cat", options=MEDIA),
    dict(key="poor", csv="poor_household", label="Poor household", kind="bool"),
    dict(key="parity", csv="birth_order_4_plus", label="4th or later birth", kind="bool"),
    dict(key="distance", csv="distance_big_problem", label="Distance a big problem", kind="bool"),
]
SPEC = {s["key"]: s for s in INPUTS}
for s in INPUTS:
    if s["kind"] == "cat":
        s["display"] = {v: k for k, v in s["options"].items()}
        s["parse"] = {**{k.lower(): v for k, v in s["options"].items()}, **{v: v for v in s["options"].values()}}
    elif s["kind"] == "bool":
        s["parse"] = {"yes": True, "no": False, "1": True, "0": False, "true": True, "false": False}

# Exact Shapley values are computed over these 10 players (the three media fields form one feature group)
PLAYERS = [("Place of delivery", ["place"]), ("ANC visits", ["anc"]), ("Mother's age at birth", ["age"]),
           ("Education", ["edu"]), ("Division", ["division"]), ("Residence", ["residence"]),
           ("Media exposure", ["newspaper", "radio", "tv"]), ("Poor household", ["poor"]),
           ("4th or later birth", ["parity"]), ("Distance a big problem", ["distance"])]

# The 22 RFE-selected model features, grouped by the input that produces them
FEATURE_NAMES = {
    "fe_place_group__home": "Delivery place: home",
    "fe_place_group__public": "Delivery place: public facility",
    "fe_place_group__private": "Delivery place: private facility",
    "m15_1__upazila_health_complex": "Upazila Health Complex",
    "m15_1__district_hospital": "District hospital",
    "m15_1__medical_college_hospital": "Medical college hospital",
    "m15_1__private_hospital": "Private hospital",
    "m15_1__private_clinic": "Private clinic",
    "fe_facility_x_2022": "Facility delivery x 2022",
    "fe_home_rural": "Home birth in a rural area",
    "m14_1": "ANC visits",
    "fe_no_anc": "No ANC visit",
    "fe_age_at_birth": "Mother's age at birth",
    "fe_teen_birth": "Teenage birth (<20 years)",
    "fe_secondary_plus": "Secondary or higher education",
    "v024__chattogram": "Division: Chattogram",
    "v024__barishal": "Division: Barishal",
    "v024__khulna": "Division: Khulna",
    "fe_media_index": "Media exposure index (0-6)",
    "fe_media_any": "Any media exposure",
    "fe_vulnerability": "Vulnerability score (0-7)",
    "survey_year": "Survey year 2022 (fixed)",
}
FEATURE_GROUPS = [
    ("Place of delivery", ["fe_place_group__home", "fe_place_group__public", "fe_place_group__private",
                           "m15_1__upazila_health_complex", "m15_1__district_hospital", "m15_1__medical_college_hospital",
                           "m15_1__private_hospital", "m15_1__private_clinic", "fe_facility_x_2022", "fe_home_rural"]),
    ("Antenatal care", ["m14_1", "fe_no_anc"]),
    ("Mother", ["fe_age_at_birth", "fe_teen_birth", "fe_secondary_plus"]),
    ("Location", ["v024__chattogram", "v024__barishal", "v024__khulna"]),
    ("Media exposure", ["fe_media_index", "fe_media_any"]),
    ("Vulnerability", ["fe_vulnerability"]),
    ("Survey", ["survey_year"]),
]
NUMERIC_FEATURES = {"m14_1", "fe_age_at_birth", "fe_media_index", "fe_vulnerability"}

TEST_METRICS = [("ROC-AUC", 0.9405), ("Accuracy", 0.8760), ("Precision", 0.8534), ("Recall", 0.8778),
                ("Specificity", 0.8745), ("F1-score", 0.8654)]
TEAL, RED = "#0f766e", "#dc2626"

WHO_URL = "https://www.who.int/publications/i/item/9789240045989"
REFERENCES = [
    "World Health Organization. <i>WHO recommendations on maternal and newborn care for a positive postnatal "
    f"experience.</i> Geneva: WHO; 2022. ISBN 978-92-4-004598-9. <a href='{WHO_URL}'>who.int</a>",
    "Croft TN, Marshall AMJ, Allen CK, et al. <i>Guide to DHS Statistics.</i> Rockville, Maryland, USA: ICF; 2018 "
    "(postnatal check during the first 2 days after birth).",
    "NIPORT, ICF. <i>Bangladesh Demographic and Health Survey 2017-18.</i> Dhaka and Rockville: NIPORT and ICF; 2020.",
    "NIPORT, ICF. <i>Bangladesh Demographic and Health Survey 2022: Final Report.</i> Dhaka and Rockville: "
    "NIPORT and ICF; 2024.",
]
ACTIONS_HIGH = [
    ("Ensure a postnatal check of the mother and newborn by skilled health personnel within 24 hours of birth: "
     "a facility stay of at least 24 hours after birth, or, after a home birth, the first contact as early as "
     "possible within 24 hours.", "Rec. 44"),
    ("Arrange a home visit in the first week after birth by skilled health personnel or a trained community "
     "health worker; if home visits are not feasible, book an outpatient postnatal contact.", "Rec. 48"),
    ("Assess the mother at each contact (bleeding, uterine tone, temperature, pulse and blood pressure in the "
     "first 24 hours; general well-being afterwards).", "Rec. 1"),
    ("Check the newborn for danger signs at each contact and refer if any is present: not feeding well, "
     "convulsions, fast breathing (>60/min), severe chest in-drawing, no spontaneous movement, fever (>37.5 °C), "
     "low temperature (<35.5 °C), jaundice in the first 24 hours or yellow palms and soles. Ask the family to seek "
     "care early if they see any of these signs.", "Rec. 25"),
    ("Schedule the remaining postnatal contacts: 48-72 hours, 7-14 days and week six after birth.", "Rec. 44"),
]
ACTIONS_LOW = [
    ("Follow the routine schedule of at least four postnatal contacts: within 24 hours, 48-72 hours, 7-14 days "
     "and week six after birth.", "Rec. 44"),
    ("Explain the newborn danger signs to the family and encourage them to seek care early if any appears.", "Rec. 25"),
]




def _load():
    import xgboost
    b = joblib.load(MODEL_PATH)
    for est in b["model"].named_steps["model"].estimators_:
        if type(est).__name__ == "XGBClassifier":
            booster = xgboost.Booster()
            booster.load_model(bytearray(b["xgb_json"]))
            est._Booster = booster
    return b


bundle = _load()
MODEL = bundle["model"]
TO_FEATURES = MODEL[:4]
THRESHOLD = float(bundle["threshold"])
YEAR = bundle["survey_year"]
APP_VERSION = str(bundle.get("app_version", "1.0"))


def to_raw(df):
    return pd.DataFrame({
        "m15_1": df["place"].values,
        "m14_1": df["anc"].astype(float).values,
        "v012": df["age"].astype(float).values,
        "b19_01": 0.0,
        "bord_01": np.where(df["parity"].astype(bool), 4.0, 2.0),
        "v106": df["edu"].values,
        "v190": np.where(df["poor"].astype(bool), "poorer", "middle"),
        "v467d": np.where(df["distance"].astype(bool), "big problem", "not a big problem"),
        "v024": df["division"].values,
        "v025": df["residence"].values,
        "v157": df["newspaper"].values,
        "v158": df["radio"].values,
        "v159": df["tv"].values,
        "survey_year": float(YEAR),
    })


def predict(df):
    return MODEL.predict_proba(to_raw(df))[:, 1]


def media_index(x):
    return sum(MEDIA_SCORE[x[k]] for k in ("newspaper", "radio", "tv"))


def vulnerability(x):
    return int(sum([x["edu"] == "no education", bool(x["poor"]), x["residence"] == "rural", x["age"] < 20,
                    bool(x["parity"]), bool(x["distance"]), media_index(x) == 0]))


_r = bundle["reference"]
REFERENCE = {"place": _r["m15_1"], "anc": float(_r["m14_1"]), "age": float(round(_r["v012"] - _r["b19_01"] / 12)),
             "edu": {"higher": "secondary"}.get(_r["v106"], _r["v106"]), "division": _r["v024"], "residence": _r["v025"],
             "newspaper": _r["v157"], "radio": _r["v158"], "tv": _r["v159"], "poor": _r["v190"] in ("poorest", "poorer"),
             "parity": float(_r["bord_01"]) >= 4, "distance": _r["v467d"] == "big problem"}
PLACE_MIX = [(REFERENCE["place"], 1.0)]
if os.path.exists(REF_PATH):
    with open(REF_PATH, encoding="utf-8") as f:
        _ref = json.load(f)
    for k, d in _ref["inputs"].items():
        REFERENCE[k] = d["median"] if "median" in d else d["mode"]
    _shares = {k: v for k, v in _ref["inputs"]["place"]["shares"].items() if v >= MIN_PLACE_SHARE and k in PLACE.values()}
    PLACE_MIX = [(k, v / sum(_shares.values())) for k, v in _shares.items()]

N_P = len(PLAYERS)
PLACE_IDX = 0
_MASKS = ((np.arange(2 ** N_P)[:, None] >> np.arange(N_P)) & 1).astype(bool)
_SIZES = _MASKS.sum(axis=1)
_WEIGHTS = np.array([math.factorial(k) * math.factorial(N_P - k - 1) / math.factorial(N_P) for k in range(N_P)])


@lru_cache(maxsize=2048)
def explain(input_items):
    """Exact Shapley values over the 10 input groups (identical to the app)."""
    x = dict(input_items)
    X = pd.DataFrame({k: np.where(_MASKS[:, i], x[k], REFERENCE[k]) for i, (_, keys) in enumerate(PLAYERS) for k in keys})
    own = _MASKS[:, PLACE_IDX]
    p = predict(pd.concat([X[own]] + [X[~own].assign(place=pl) for pl, _ in PLACE_MIX], ignore_index=True))
    v = np.empty(2 ** N_P)
    v[own] = p[:own.sum()]
    v[~own] = np.array([w for _, w in PLACE_MIX]) @ p[own.sum():].reshape(len(PLACE_MIX), -1)
    idx = np.arange(2 ** N_P)
    phi = np.empty(N_P)
    for i in range(N_P):
        s = idx[~_MASKS[:, i]]
        phi[i] = np.sum(_WEIGHTS[_SIZES[s]] * (v[s | (1 << i)] - v[s]))
    return float(v[-1]), float(v[0]), tuple(float(t) for t in phi)


def player_text(name, keys, x):
    if name == "Media exposure":
        return f"Media exposure: index {media_index(x)} of 6"
    k = keys[0]
    s = SPEC[k]
    if s["kind"] == "cat":
        return f"{name}: {s['display'][x[k]]}"
    if s["kind"] == "bool":
        return f"{name}: {'Yes' if x[k] else 'No'}"
    return f"{name}: {x[k]:g} {s['unit']}"


def selected_features(x):
    z = TO_FEATURES.transform(to_raw(pd.DataFrame([x]))).iloc[0]
    return {c: (int(YEAR) if c == "survey_year" else float(z[c])) for _, cols in FEATURE_GROUPS for c in cols}


def parse_inputs(raw):
    """Turn a dict keyed by the CSV column names into the internal input dict. Accepts the labels shown in the app
    (any case) or the model values. Raises ValueError with a clear message."""
    x, errors = {}, []
    for s in INPUTS:
        v = raw.get(s["csv"])
        if v is None:
            errors.append(f"{s['csv']}: missing")
            continue
        if s["kind"] == "num":
            try:
                v = float(v)
            except (TypeError, ValueError):
                errors.append(f"{s['csv']}: must be a number")
                continue
            if not s["min"] <= v <= s["max"]:
                errors.append(f"{s['csv']}: must be between {s['min']} and {s['max']}")
                continue
            x[s["key"]] = v
        else:
            key = str(v).strip().lower()
            if s["kind"] == "bool" and isinstance(v, bool):
                x[s["key"]] = v
            elif key in s["parse"]:
                x[s["key"]] = s["parse"][key]
            else:
                allowed = ", ".join(s["options"]) if s["kind"] == "cat" else "Yes, No"
                errors.append(f"{s['csv']}: '{v}' is not allowed (allowed: {allowed})")
    if errors:
        raise ValueError("; ".join(errors))
    return x


def assess(x, with_explanation=True):
    if with_explanation:
        risk, ref_risk, phi = explain(tuple(sorted(x.items())))
    else:
        risk, ref_risk, phi = float(predict(pd.DataFrame([x]))[0]), None, None
    high = risk >= THRESHOLD
    out = {"risk": round(risk, 4), "risk_percent": round(risk * 100, 1),
           "risk_level": "high" if high else "low", "threshold": THRESHOLD,
           "vulnerability_score": vulnerability(x), "media_exposure_index": media_index(x)}
    if with_explanation:
        reasons = sorted(({"input": player_text(n, k, x), "points": round(p * 100, 1)} for (n, k), p in zip(PLAYERS, phi)),
                         key=lambda r: -abs(r["points"]))
        out.update({
            "actions": [{"text": t, "who": r} for t, r in (ACTIONS_HIGH if high else ACTIONS_LOW)],
            "reasons": reasons,
            "reference_risk": round(ref_risk, 4),
            "selected_features": selected_features(x),
        })
    return out
