"""
UCL Risk Predictor: web server.

Serves the single-page front end in static/ and a small JSON API:

    GET  /api/options   dropdown lists, valid ranges and model facts
    POST /api/predict   run Model 1 or Model 2 for one reactor
    GET  /healthz       simple "is it running" check

The models are loaded once from artifacts/ (created by train_models.py).
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

import ucl_core as U

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"

app = Flask(__name__, static_folder=str(BASE / "static"), static_url_path="")

# ----------------------------------------------------------------------------
# Plain-language labels for the dropdowns
# ----------------------------------------------------------------------------
COUNTRY_NAMES = {
    "IRAN, ISLAMIC REPUBLIC OF": "Iran",
    "KOREA, REPUBLIC OF": "South Korea",
    "UK": "United Kingdom",
    "UAE": "United Arab Emirates",
    "USA": "United States",
}
TYPE_LABELS = {
    "PWR": "PWR (pressurised water)",
    "BWR": "BWR (boiling water)",
    "PHWR": "PHWR (pressurised heavy water)",
    "GCR": "GCR (gas-cooled)",
    "LWGR": "LWGR (graphite-moderated, e.g. RBMK)",
    "FBR": "FBR (fast breeder)",
}
NSSS_LABELS = {
    "Western_PWR": ("Western PWR makers", "Westinghouse, Combustion Engineering, Babcock & Wilcox"),
    "French_PWR": ("French and German PWR makers", "Framatome, Orano, Siemens KWU"),
    "GE_BWR": ("GE-lineage BWR makers", "General Electric, Hitachi, Toshiba, ASEA-Atom"),
    "Russian_VVER": ("Russian and Soviet-lineage makers", "Atomstroyexport, OKB Gidropress, Skoda"),
    "Chinese": ("Chinese makers", "CNNC, Dongfang, Shanghai Electric"),
    "Canadian_CANDU": ("Canadian and Indian heavy-water makers", "AECL, NPCIL"),
    "Korean_Other": ("Korean, Japanese and other makers", "KEPCO, Doosan, Mitsubishi, others"),
}
FAMILY_LABELS = {
    "VVER": "VVER (Soviet or Russian pressurised water)",
    "BWR": "BWR (any boiling-water design)",
    "CANDU": "CANDU or other heavy-water design",
    "AP1000": "AP1000",
    "EPR": "EPR",
    "GCR": "Gas-cooled (AGR, Magnox)",
    "FBR": "Fast breeder (BN series)",
    "Other": "Other or not listed",
}


def nice_country(raw):
    if raw in COUNTRY_NAMES:
        return COUNTRY_NAMES[raw]
    return raw.title()


# ----------------------------------------------------------------------------
# Load models once
# ----------------------------------------------------------------------------
def _single_thread(model):
    """Avoid spawning worker processes for every request."""
    parts = [model] + list(getattr(model, "estimators_", []))
    for part in parts:
        try:
            part.set_params(n_jobs=1)
        except Exception:
            if hasattr(part, "n_jobs"):
                part.n_jobs = 1


BUNDLES = {}
for key in ("model1", "model2"):
    bundle = joblib.load(ART / f"{key}.joblib")
    _single_thread(bundle["model"])
    BUNDLES[key] = bundle
META = json.loads((ART / "meta.json").read_text())

# ----------------------------------------------------------------------------
# Hard limits (reject) and soft limits (warn)
# ----------------------------------------------------------------------------
HARD = {
    "net_capacity_mw": (50, 2500),
    "age_years": (0, 80),
    "construction_years": (1, 60),
    "ucf_p1": (0, 100),
    "ucf_p2": (0, 100),
}
FIELD_UNITS = {
    "net_capacity_mw": " MW",
    "age_years": " years",
    "construction_years": " years",
    "ucf_p1": "%",
    "ucf_p2": "%",
}
FIELD_NAMES = {
    "net_capacity_mw": "Net electrical capacity",
    "age_years": "Reactor age",
    "construction_years": "Construction time",
    "ucf_p1": "UCF 2013 to 2017",
    "ucf_p2": "UCF 2010 to 2020",
}

EXPLAIN_GROUPS = {
    "model1": [
        ("country", "Country", ["country"]),
        ("type", "Reactor type", ["type"]),
        ("size", "Plant size", ["net_capacity_mw"]),
        ("age", "Reactor age", ["age_years"]),
        ("build", "Construction time", ["construction_years"]),
        ("lineage", "Supplier and design family", ["nsss_group", "model_family"]),
    ],
}
EXPLAIN_GROUPS["model2"] = EXPLAIN_GROUPS["model1"] + [
    ("history", "Past capability factor", ["ucf_p1", "ucf_p2"]),
]


class InputError(ValueError):
    pass


def parse_inputs(payload, model_key):
    """Check the form values. Returns (clean_inputs, warnings)."""
    meta = META["models"][model_key]
    info = BUNDLES[model_key]["info"]
    warnings = []
    clean = {}

    # Numbers
    numeric = ["net_capacity_mw", "age_years", "construction_years"]
    if model_key == "model2":
        numeric += ["ucf_p1", "ucf_p2"]
    for name in numeric:
        raw = payload.get(name)
        try:
            val = float(raw)
        except (TypeError, ValueError):
            raise InputError(f"{FIELD_NAMES[name]} must be a number.")
        if val != val or val in (float("inf"), float("-inf")):
            raise InputError(f"{FIELD_NAMES[name]} must be a number.")
        lo, hi = HARD[name]
        if not lo <= val <= hi:
            raise InputError(f"{FIELD_NAMES[name]} must be between {lo:g} and {hi:g}.")
        clean[name] = val
        rng = meta["ranges"].get(name)
        if rng and not rng[0] <= val <= rng[1]:
            warnings.append({
                "field": name,
                "text": (f"{FIELD_NAMES[name]} of {val:g}{FIELD_UNITS[name]} is outside the range the model "
                         f"learned from ({rng[0]:g} to {rng[1]:g}{FIELD_UNITS[name]}). "
                         "Treat this result with extra caution."),
            })

    # Choices
    t = str(payload.get("type", ""))
    if t not in info["type_classes"]:
        raise InputError("Choose a reactor type.")
    clean["type"] = t

    g = str(payload.get("nsss_group", ""))
    if g not in info["nsss_classes"]:
        raise InputError("Choose a supplier family.")
    clean["nsss_group"] = g

    f = str(payload.get("model_family", ""))
    if f not in FAMILY_LABELS:
        raise InputError("Choose a design family.")
    if f not in info["family_classes"]:
        warnings.append({
            "field": "model_family",
            "text": (f"{FAMILY_LABELS[f]} has no examples in the data this model learned from, "
                     "so it was treated as \"Other or not listed\"."),
        })
        f = "Other"
    clean["model_family"] = f

    c = str(payload.get("country", "")).upper()
    if c in ("", "OTHER"):
        clean["country"] = "OTHER"
        warnings.append({
            "field": "country",
            "text": ("No country information was used. The country is one of the strongest signals "
                     "in the model, so this result is less specific."),
        })
    elif c in info["country_mean"]:
        clean["country"] = c
    else:
        clean["country"] = c
        warnings.append({
            "field": "country",
            "text": (f"{nice_country(c)} has no reactors in the data this model learned from, "
                     "so an average country effect was used."),
        })

    return clean, warnings


def feature_row(clean, model_key):
    info = BUNDLES[model_key]["info"]
    return U.design_row(clean, info, temporal=(model_key == "model2"))


def predict_matrix(model_key, rows):
    bundle = BUNDLES[model_key]
    X = pd.DataFrame(rows)[bundle["features"]]
    Xs = pd.DataFrame(bundle["scaler"].transform(X), columns=bundle["features"])
    return bundle["model"].predict_proba(Xs)


def certainty(probs):
    ordered = sorted(probs, reverse=True)
    top, second = ordered[0], ordered[1]
    if top >= 0.75:
        level = "clear"
    elif top >= 0.50:
        level = "leaning"
    else:
        level = "close"
    return {"level": level, "top": top, "gap": top - second}


def run_prediction(payload):
    model_key = {"1": "model1", "2": "model2", 1: "model1", 2: "model2"}.get(payload.get("model"))
    if model_key is None:
        raise InputError("Choose Model 1 or Model 2.")
    clean, warnings = parse_inputs(payload, model_key)
    typical = dict(META["models"][model_key]["typical"])
    typical["country"] = "OTHER"

    # Row 0 is the real reactor. Each further row swaps one input group for the typical value.
    groups = EXPLAIN_GROUPS[model_key]
    inputs = [clean]
    for _gid, _label, keys in groups:
        swapped = dict(clean)
        for k in keys:
            swapped[k] = typical[k]
        inputs.append(swapped)
    probs = predict_matrix(model_key, [feature_row(i, model_key) for i in inputs])

    p = probs[0]
    k = int(p.argmax())
    drivers = []
    for idx, (gid, label, _keys) in enumerate(groups, start=1):
        effect = (p[k] - probs[idx][k]) * 100
        drivers.append({"key": gid, "label": label, "effect_pp": round(float(effect), 1)})
    drivers.sort(key=lambda d: abs(d["effect_pp"]), reverse=True)

    meta = META["models"][model_key]
    cm = meta["test"]["confusion_matrix"]
    med_total = sum(cm[1])
    return {
        "model": model_key,
        "predicted": U.CLASS_LIST[k],
        "probabilities": {c: round(float(p[i]), 4) for i, c in enumerate(U.CLASS_LIST)},
        "certainty": certainty([float(x) for x in p]),
        "drivers": drivers,
        "warnings": warnings,
        "model_facts": {
            "n_reactors": meta["n_reactors"],
            "n_test": meta["test"]["n_test"],
            "accuracy": meta["test"]["accuracy"],
            "medium_found": cm[1][1],
            "medium_total": med_total,
        },
    }


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/healthz")
def healthz():
    return jsonify(status="ok")


@app.get("/api/options")
def options():
    m1, m2 = META["models"]["model1"], META["models"]["model2"]
    countries = sorted(set(m1["countries"]) | set(m2["countries"]))
    return jsonify({
        "classes": U.CLASS_LIST,
        "countries": [{"value": c, "label": nice_country(c)}
                      for c in sorted(countries, key=nice_country)],
        "types": [{"value": t, "label": TYPE_LABELS.get(t, t)} for t in m1["types"]],
        "nsss_groups": [{"value": g, "label": NSSS_LABELS[g][0], "hint": NSSS_LABELS[g][1]}
                        for g in m1["nsss_groups"]],
        "families": [{"value": f, "label": FAMILY_LABELS[f]} for f in FAMILY_LABELS],
        "suggest_by_type": m1["suggest_by_type"],
        "models": {
            key: {
                "n_reactors": meta["n_reactors"],
                "n_features": meta["n_features"],
                "class_counts": meta["class_counts"],
                "accuracy": meta["test"]["accuracy"],
                "auc": meta["test"]["auc_ovr_weighted"],
                "n_test": meta["test"]["n_test"],
                "per_class": meta["test"]["per_class"],
                "confusion_matrix": meta["test"]["confusion_matrix"],
                "ranges": meta["ranges"],
                "typical": meta["typical"],
            } for key, meta in META["models"].items()
        },
    })


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send the reactor details as JSON."), 400
    try:
        return jsonify(run_prediction(payload))
    except InputError as exc:
        return jsonify(error=str(exc)), 400


@app.after_request
def headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    if request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
