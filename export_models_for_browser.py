"""
export_models_for_browser.py
Writes static/browser_models.json: the two trained models in a compact form
that a web page can run by itself (no server needed).

The page code that reads this file is static/local-engine.js. A test in
tools/ compares its answers with the Python server on random inputs.

Run from the project folder:   python export_models_for_browser.py
"""
import base64
import json
from pathlib import Path

import joblib
import numpy as np

import app as A

ART = Path(__file__).resolve().parent / "artifacts"
ROOT_STATIC = Path(__file__).resolve().parent / "static"


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def float32_floor(values):
    """Largest float32 that is not above each float64 threshold.

    scikit-learn compares a float32 feature value with a float64 threshold.
    Because the feature value is float32, "x <= t" gives the same answer as
    "x <= largest float32 not above t", so the stored threshold loses nothing.
    """
    t64 = np.asarray(values, dtype=np.float64)
    t32 = t64.astype(np.float32)
    too_high = t32.astype(np.float64) > t64
    t32[too_high] = np.nextafter(t32[too_high], np.float32(-np.inf))
    return t32


def export_forest(forest):
    """Random forest or extra trees: one record per tree, leaves hold class proportions."""
    offsets, feat, thr, left, right = [0], [], [], [], []
    leaf_table, leaf_index = [], {}
    for est in forest.estimators_:
        t = est.tree_
        n = t.node_count
        for i in range(n):
            if t.children_left[i] == -1:
                v = t.value[i][0]
                v = v / v.sum()
                key = tuple(np.round(v, 7))
                if key not in leaf_index:
                    leaf_index[key] = len(leaf_table)
                    leaf_table.append(key)
                feat.append(-1)
                thr.append(0.0)
                left.append(leaf_index[key])
                right.append(0)
            else:
                feat.append(int(t.feature[i]))
                thr.append(float(t.threshold[i]))
                left.append(int(t.children_left[i]))
                right.append(int(t.children_right[i]))
        offsets.append(offsets[-1] + n)
    assert len(leaf_table) < 32000
    return {
        "kind": "forest",
        "offsets": b64(np.array(offsets, dtype=np.int32)),
        "n_trees": len(forest.estimators_),
        "feat": b64(np.array(feat, dtype=np.int8)),
        "thr": b64(float32_floor(thr)),
        "left": b64(np.array(left, dtype=np.int16)),
        "right": b64(np.array(right, dtype=np.int16)),
        "leaves": b64(np.array(leaf_table, dtype=np.float32)),
    }


def export_xgb(est, features):
    """XGBoost multi-class: tree k belongs to class k % 3; prediction is softmax of the summed leaves."""
    df = est.get_booster().trees_to_dataframe()
    fidx = {f: i for i, f in enumerate(features)}
    offsets, feat, thr, left, right, cls = [0], [], [], [], [], []
    for tree_id, g in df.groupby("Tree", sort=True):
        g = g.sort_values("Node")
        ids = {row_id: pos for pos, row_id in enumerate(g["ID"])}
        for _, r in g.iterrows():
            if r["Feature"] == "Leaf":
                feat.append(-1)
                thr.append(float(r["Gain"]))
                left.append(0)
                right.append(0)
            else:
                feat.append(fidx[r["Feature"]])
                thr.append(float(r["Split"]))
                left.append(ids[r["Yes"]])
                right.append(ids[r["No"]])
        offsets.append(offsets[-1] + len(g))
        cls.append(int(tree_id) % 3)
    return {
        "kind": "xgb",
        "offsets": b64(np.array(offsets, dtype=np.int32)),
        "n_trees": len(cls),
        "cls": b64(np.array(cls, dtype=np.uint8)),
        "feat": b64(np.array(feat, dtype=np.int8)),
        "thr": b64(np.array(thr, dtype=np.float32)),
        "left": b64(np.array(left, dtype=np.int16)),
        "right": b64(np.array(right, dtype=np.int16)),
    }


def export_bundle(key):
    b = joblib.load(ART / f"{key}.joblib")
    model, feats = b["model"], b["features"]
    out = {
        "features": feats,
        "scaler": {"mean": [float(x) for x in b["scaler"].mean_],
                   "scale": [float(x) for x in b["scaler"].scale_]},
        "info": b["info"],
    }
    name = type(model).__name__
    if name == "StackingClassifier":
        names = [n for n, _ in model.estimators]
        parts = {}
        for n, est in zip(names, model.estimators_):
            parts[n] = export_forest(est) if hasattr(est, "estimators_") else export_xgb(est, feats)
        out["kind"] = "stack"
        out["parts"] = parts
        out["order"] = names
        out["meta_coef"] = model.final_estimator_.coef_.tolist()
        out["meta_intercept"] = model.final_estimator_.intercept_.tolist()
    elif name in ("RandomForestClassifier", "ExtraTreesClassifier"):
        out["kind"] = "forest"
        out["parts"] = {"rf": export_forest(model)}
        out["order"] = ["rf"]
    elif name == "LogisticRegression":
        out["kind"] = "logreg"
        out["parts"] = {}
        out["order"] = []
        out["coef"] = model.coef_.tolist()
        out["intercept"] = model.intercept_.tolist()
    elif name == "XGBClassifier":
        out["kind"] = "xgb"
        out["parts"] = {"xgb": export_xgb(model, feats)}
        out["order"] = ["xgb"]
    else:
        raise ValueError(f"cannot export a {name} to the browser")
    return out


def main():
    client = A.app.test_client()
    options = client.get("/api/options").get_json()
    payload = {
        "options": options,
        "meta": A.META,
        "limits": {k: list(v) for k, v in A.HARD.items()},
        "field_names": A.FIELD_NAMES,
        "field_units": A.FIELD_UNITS,
        "family_labels": A.FAMILY_LABELS,
        "country_names": {c["value"]: c["label"] for c in options["countries"]},
        "explain_groups": {k: [[g, l, ks] for g, l, ks in v] for k, v in A.EXPLAIN_GROUPS.items()},
        "model1": export_bundle("model1"),
        "model2": export_bundle("model2"),
    }
    text = json.dumps(payload, separators=(",", ":"))
    (ROOT_STATIC / "browser_models.json").write_text(text)
    print(f"Wrote static/browser_models.json ({len(text) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
