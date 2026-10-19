"""
train_models.py
Rebuilds the two models behind the UCL Risk Predictor website from the research
dataset and saves them in the artifacts/ folder.

Run it only when you want to retrain (for example after the dataset changes):

    python train_models.py UCL_Dataset_v5_longitudinal.xlsx

The website itself does NOT need the dataset or this script. It only loads the
saved files in artifacts/.

Model 1  Design features only (13 inputs). Stacking ensemble (RF + ExtraTrees + XGBoost,
         Logistic Regression on top). No hyperparameter search (as in the paper).
Model 2  Design features + 5 history features from UCF 2013-2017 and UCF 2010-2020
         (18 inputs). XGBoost tuned with a 40-iteration random search.
         UCF 2014-2024 is never used because it defines the target.

Both models are trained on the same 80 % stratified training split (SMOTE on the
training part only) that produced the test results reported in the paper, and the
test metrics saved in meta.json come from the matching 20 % held-out reactors.
"""
import copy
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import ucl_core as U

warnings.filterwarnings("ignore")
RS = 42
OUT = "artifacts"


def metrics_for(model, Xte, yte):
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)
    p, r, f, s = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2], zero_division=0)
    return {
        "n_test": int(len(yte)),
        "accuracy": round(accuracy_score(yte, pred) * 100, 2),
        "f1_weighted": round(f1_score(yte, pred, average="weighted") * 100, 2),
        "auc_ovr_weighted": round(roc_auc_score(yte, proba, multi_class="ovr",
                                                average="weighted") * 100, 2),
        "per_class": {c: {"precision": round(float(p[i]) * 100, 1),
                          "recall": round(float(r[i]) * 100, 1),
                          "f1": round(float(f[i]) * 100, 1),
                          "support": int(s[i])} for i, c in enumerate(U.CLASS_LIST)},
        "confusion_matrix": confusion_matrix(yte, pred, labels=[0, 1, 2]).tolist(),
    }


def typical_inputs(df, info):
    """The 'typical reactor' used to explain a prediction (median / most common value)."""
    t = {
        "type": df["Type"].mode().iloc[0],
        "country": None,  # unknown country -> overall mean encoding
        "net_capacity_mw": float(df["Net_Capacity_MW"].median()),
        "age_years": float(df["Reactor_Age"].median()),
        "construction_years": float(df["Construction_Duration"].median()),
        "nsss_group": df["NSSS_Group"].mode().iloc[0],
        "model_family": df["Model_Family"].mode().iloc[0],
    }
    if "UCF_P1" in df.columns:
        t["ucf_p1"] = float(df["UCF_P1"].median())
        t["ucf_p2"] = float(df["UCF_P2"].median())
    return t


def training_ranges(df):
    r = {
        "net_capacity_mw": [float(df["Net_Capacity_MW"].min()), float(df["Net_Capacity_MW"].max())],
        "age_years": [float(df["Reactor_Age"].min()), float(df["Reactor_Age"].max())],
        "construction_years": [float(df["Construction_Duration"].min()),
                               float(df["Construction_Duration"].max())],
    }
    if "UCF_P1" in df.columns:
        r["ucf_p1"] = [float(df["UCF_P1"].min()), float(df["UCF_P1"].max())]
        r["ucf_p2"] = [float(df["UCF_P2"].min()), float(df["UCF_P2"].max())]
    return r


def split_and_scale(df, feats):
    X = df[feats].copy()
    y = df["Target"]
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), columns=feats)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.20, stratify=y, random_state=RS)
    Xsm, ysm = SMOTE(random_state=RS, k_neighbors=3).fit_resample(Xtr, ytr)
    return scaler, Xsm, ysm, Xte, yte


def build_model1(xlsx):
    df = U.load_final(xlsx)
    df, info = U.engineer(df, phwr_is_candu=True)
    scaler, Xsm, ysm, Xte, yte = split_and_scale(df, U.FEAT_M1)

    xgb = XGBClassifier(random_state=RS, eval_metric="mlogloss", verbosity=0, n_jobs=-1)
    rf = RandomForestClassifier(class_weight="balanced", n_estimators=300,
                                random_state=RS, n_jobs=-1)
    etc = ExtraTreesClassifier(class_weight="balanced", n_estimators=300,
                               random_state=RS, n_jobs=-1)
    model = StackingClassifier(
        estimators=[("rf", copy.deepcopy(rf)), ("etc", copy.deepcopy(etc)),
                    ("xgb", copy.deepcopy(xgb))],
        final_estimator=LogisticRegression(max_iter=2000, random_state=RS),
        cv=5, passthrough=False, n_jobs=-1)
    model.fit(Xsm, ysm)
    return df, info, scaler, model, metrics_for(model, Xte, yte), len(ysm)


def build_model2(xlsx):
    df = U.load_final(xlsx)
    df = U.add_longitudinal(df, xlsx)
    df = df.dropna(subset=["UCF_2013_2017", "UCF_2010_2020"]).reset_index(drop=True)
    # Model 2 pipeline rule: only model names containing CANDU count as CANDU
    df, info = U.engineer(df, phwr_is_candu=False, temporal=True)
    scaler, Xsm, ysm, Xte, yte = split_and_scale(df, U.FEAT_M2)

    grid = {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1], "subsample": [0.8, 1.0]}
    kf5 = StratifiedKFold(5, shuffle=True, random_state=RS)
    search = RandomizedSearchCV(
        XGBClassifier(random_state=RS, eval_metric="mlogloss", verbosity=0, n_jobs=-1),
        grid, n_iter=40, cv=kf5, scoring="f1_weighted", random_state=RS, n_jobs=-1)
    search.fit(Xsm, ysm)
    model = search.best_estimator_
    m = metrics_for(model, Xte, yte)
    m["best_params"] = {k: (float(v) if isinstance(v, (np.floating, float)) else int(v))
                        for k, v in search.best_params_.items()}
    return df, info, scaler, model, m, len(ysm)


def main(xlsx):
    os.makedirs(OUT, exist_ok=True)
    meta = {"classes": U.CLASS_LIST, "models": {}}
    for key, builder in (("model1", build_model1), ("model2", build_model2)):
        print(f"Training {key} ...")
        df, info, scaler, model, mets, n_smote = builder(xlsx)
        feats = info["features"]
        print(f"  reactors={len(df)}  features={len(feats)}  "
              f"accuracy={mets['accuracy']}  F1={mets['f1_weighted']}  AUC={mets['auc_ovr_weighted']}")
        joblib.dump({"model": model, "scaler": scaler, "info": info, "features": feats},
                    f"{OUT}/{key}.joblib", compress=3)
        meta["models"][key] = {
            "n_reactors": int(len(df)),
            "n_features": len(feats),
            "class_counts": {c: int((df["UCL_Class"] == c).sum()) for c in U.CLASS_LIST},
            "test": mets,
            "typical": typical_inputs(df, info),
            "ranges": training_ranges(df),
            "types": info["type_classes"],
            "families": info["family_classes"],
            "nsss_groups": info["nsss_classes"],
            "countries": sorted(info["country_mean"].keys()),
            "suggest_by_type": {
                t: {"nsss_group": g["NSSS_Group"].mode().iloc[0],
                    "model_family": g["Model_Family"].mode().iloc[0]}
                for t, g in df.groupby("Type")},
        }
    with open(f"{OUT}/meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print("Saved artifacts/ (model1.joblib, model2.joblib, meta.json)")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "UCL_Dataset_v5_longitudinal.xlsx"
    main(path)
