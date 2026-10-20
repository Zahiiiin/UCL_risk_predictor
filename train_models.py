"""
train_models.py
Rebuilds the two models behind the UCL Risk Predictor website from the research
dataset and saves them in the artifacts/ folder.

Run it only when you want to retrain (for example after the dataset changes):

    python train_models.py data/UCL_Dataset_v6_PRIS.xlsx

The website itself does NOT need the dataset or this script. It only loads the
saved files in artifacts/.

Model 1  Design features only (13 inputs), all 387 reactors.
Model 2  Design features + 5 history features from the mean UCL in 2010-2013 and 2014-2017
         (18 inputs), the 324 reactors that have both history values.
         The label (mean UCL 2018-2024) is never an input.

WHICH CLASSIFIER, AND WHY (settings you can change below)
    The classifier of each model is set in CLASSIFIER. It was chosen from the repeated
    cross-validation benchmark (research/UCL_Benchmark_CV.py, results in
    research/expected_results/benchmark), not from a single test split: the rule is the highest
    macro F1. No hyperparameter search is used.

HOW THE NUMBERS SHOWN ON THE SITE ARE MADE
    The final model is trained on ALL reactors of its group (no reactors are held back), so
    it uses every example. The accuracy shown on the site cannot come from those same
    reactors, so it comes from 5 x repeated stratified 5-fold cross-validation (25 test
    folds). In every fold, scaling, the Country encoding and SMOTE are fitted on the training
    part only. This is the same procedure as the benchmark. The numbers agree with the benchmark to
    within about half a point (the row order and the PHWR family rule of Model 1 differ slightly).
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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import ucl_core as U

warnings.filterwarnings("ignore")
RS = 42
OUT = "artifacts"
REPEATS = 5
# Rule used to choose: the classifier with the highest macro F1 in the cross-validation benchmark
# (feature set A for Model 1: LR 61.4 vs RF 58.8, ETC 58.4, XGB 58.2, Stacking 56.1;
#  feature set D for Model 2: RF 65.5 vs ETC 64.9, LR 62.6, Stacking 62.2, XGB 59.3).
# Macro F1 weighs the rare Medium class as much as the others. Options: LR, RF, ETC, XGB, Stacking.
CLASSIFIER = {"model1": "LR", "model2": "RF"}


# ------------------------------------------------------------------------------
# Classifiers (same settings as the research pipelines, nothing tuned)
# ------------------------------------------------------------------------------
def make_classifier(name):
    xgb = XGBClassifier(random_state=RS, eval_metric="mlogloss", verbosity=0, n_jobs=-1)
    rf = RandomForestClassifier(class_weight="balanced", n_estimators=300,
                                random_state=RS, n_jobs=-1)
    etc = ExtraTreesClassifier(class_weight="balanced", n_estimators=300,
                               random_state=RS, n_jobs=-1)
    if name == "LR":
        return LogisticRegression(max_iter=2000, random_state=RS)
    if name == "RF":
        return rf
    if name == "ETC":
        return etc
    if name == "XGB":
        return xgb
    if name == "Stacking":
        return StackingClassifier(
            estimators=[("rf", copy.deepcopy(rf)), ("etc", copy.deepcopy(etc)),
                        ("xgb", copy.deepcopy(xgb))],
            final_estimator=LogisticRegression(max_iter=2000, random_state=RS),
            cv=5, passthrough=False, n_jobs=-1)
    raise ValueError(f"unknown classifier {name}")


# ------------------------------------------------------------------------------
# Cross-validated accuracy (what the site shows)
# ------------------------------------------------------------------------------
def country_encoding(train, test):
    """Mean class (0/1/2) per country. Training rows get an out-of-fold value computed
    inside the training part; test rows get the value from the whole training part."""
    glob = train["Target"].mean()
    full = train.groupby("Country")["Target"].mean()
    tr_enc = pd.Series(np.nan, index=train.index)
    inner = StratifiedKFold(5, shuffle=True, random_state=RS)
    for a, b in inner.split(train, train["Target"]):
        m = train.iloc[a].groupby("Country")["Target"].mean()
        tr_enc.iloc[b] = train.iloc[b]["Country"].map(m).values
    tr_enc = tr_enc.fillna(train["Country"].map(full)).fillna(glob)
    te_enc = test["Country"].map(full).fillna(glob)
    return tr_enc.values, te_enc.values


def round_rows_keep_sums(m):
    """Round a matrix of averages to whole numbers so that every row keeps its exact
    (whole) sum. Uses the largest-remainder rule."""
    out = np.floor(m).astype(int)
    for i in range(m.shape[0]):
        need = int(round(m[i].sum())) - out[i].sum()
        order = np.argsort(-(m[i] - np.floor(m[i])))
        for j in order[:need]:
            out[i, j] += 1
    return out


def cross_validate(df, feats, classifier):
    """5 x repeated stratified 5-fold CV. Returns the metrics saved in meta.json."""
    y = df["Target"].values
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=RS)
    acc, bacc, wf1, mf1, auc = [], [], [], [], []
    base_major, base_country = [], []
    cm_sum = np.zeros((3, 3))
    for tr, te in cv.split(df, y):
        dtr, dte = df.iloc[tr].copy(), df.iloc[te].copy()
        ytr, yte = y[tr], y[te]
        Xtr = dtr[[f for f in feats if f != "Country_OOF"]].copy()
        Xte = dte[[f for f in feats if f != "Country_OOF"]].copy()
        if "Country_OOF" in feats:
            Xtr["Country_OOF"], Xte["Country_OOF"] = country_encoding(dtr, dte)
        Xtr, Xte = Xtr[feats], Xte[feats]
        sc = StandardScaler().fit(Xtr)
        Xtr_s = pd.DataFrame(sc.transform(Xtr), columns=feats)
        Xte_s = pd.DataFrame(sc.transform(Xte), columns=feats)
        Xsm, ysm = SMOTE(random_state=RS, k_neighbors=3).fit_resample(Xtr_s, ytr)
        mdl = make_classifier(classifier)
        mdl.fit(Xsm, ysm)
        pred, proba = mdl.predict(Xte_s), mdl.predict_proba(Xte_s)
        acc.append(accuracy_score(yte, pred) * 100)
        bacc.append(balanced_accuracy_score(yte, pred) * 100)
        wf1.append(f1_score(yte, pred, average="weighted") * 100)
        mf1.append(f1_score(yte, pred, average="macro") * 100)
        auc.append(roc_auc_score(yte, proba, multi_class="ovr", average="macro") * 100)
        cm_sum += confusion_matrix(yte, pred, labels=[0, 1, 2])
        # baselines on the same folds
        maj = int(np.bincount(ytr).argmax())
        base_major.append(np.mean(yte == maj) * 100)
        cmaj = dtr.groupby("Country")["Target"].agg(lambda s: int(s.value_counts().idxmax()))
        bc = dte["Country"].map(cmaj).fillna(maj).astype(int).values
        base_country.append(np.mean(yte == bc) * 100)
    cm_mean = cm_sum / REPEATS
    prec = np.divide(np.diag(cm_mean), cm_mean.sum(axis=0), out=np.zeros(3), where=cm_mean.sum(axis=0) > 0)
    rec = np.divide(np.diag(cm_mean), cm_mean.sum(axis=1), out=np.zeros(3), where=cm_mean.sum(axis=1) > 0)
    f1c = np.divide(2 * prec * rec, prec + rec, out=np.zeros(3), where=(prec + rec) > 0)
    cm_int = round_rows_keep_sums(cm_mean)
    return {
        "method": f"{REPEATS} x repeated stratified 5-fold cross-validation ({5 * REPEATS} test folds)",
        "classifier": classifier,
        "n_test": int(len(df)),
        "accuracy": round(float(np.mean(acc)), 2),
        "accuracy_sd": round(float(np.std(acc, ddof=1)), 2),
        "balanced_accuracy": round(float(np.mean(bacc)), 2),
        "f1_weighted": round(float(np.mean(wf1)), 2),
        "f1_macro": round(float(np.mean(mf1)), 2),
        "auc_ovr_macro": round(float(np.mean(auc)), 2),
        "baselines": {"majority_class": round(float(np.mean(base_major)), 2),
                      "country_majority": round(float(np.mean(base_country)), 2)},
        "per_class": {c: {"precision": round(float(prec[i]) * 100, 1),
                          "recall": round(float(rec[i]) * 100, 1),
                          "f1": round(float(f1c[i]) * 100, 1),
                          "support": int(cm_int[i].sum())} for i, c in enumerate(U.CLASS_LIST)},
        "confusion_matrix": cm_int.tolist(),
    }


# ------------------------------------------------------------------------------
# Helpers for the site
# ------------------------------------------------------------------------------
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
    if "UCL_P1" in df.columns:
        t["ucl_p1"] = float(df["UCL_P1"].median())
        t["ucl_p2"] = float(df["UCL_P2"].median())
    return t


def training_ranges(df):
    r = {
        "net_capacity_mw": [float(df["Net_Capacity_MW"].min()), float(df["Net_Capacity_MW"].max())],
        "age_years": [float(df["Reactor_Age"].min()), float(df["Reactor_Age"].max())],
        "construction_years": [float(df["Construction_Duration"].min()),
                               float(df["Construction_Duration"].max())],
    }
    if "UCL_P1" in df.columns:
        r["ucl_p1"] = [float(df["UCL_P1"].min()), float(df["UCL_P1"].max())]
        r["ucl_p2"] = [float(df["UCL_P2"].min()), float(df["UCL_P2"].max())]
    return r


def prepare(xlsx, key):
    df = U.load_final(xlsx)
    if key == "model1":
        return U.engineer(df, phwr_is_candu=True)
    df = U.add_longitudinal(df, xlsx)
    df = df.dropna(subset=["UCL_2010_2013", "UCL_2014_2017"]).reset_index(drop=True)
    # Model 2 pipeline rule: only model names containing CANDU count as CANDU
    return U.engineer(df, phwr_is_candu=False, temporal=True)


def fit_final(df, feats, classifier):
    """Train on every reactor of the group (scaling and SMOTE fitted on all of them)."""
    X = df[feats].copy()
    y = df["Target"]
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), columns=feats)
    Xsm, ysm = SMOTE(random_state=RS, k_neighbors=3).fit_resample(Xs, y)
    model = make_classifier(classifier)
    model.fit(Xsm, ysm)
    return scaler, model, len(ysm)


def main(xlsx):
    os.makedirs(OUT, exist_ok=True)
    meta = {"classes": U.CLASS_LIST, "models": {}}
    for key in ("model1", "model2"):
        clf = CLASSIFIER[key]
        print(f"Training {key} ({clf}) ...")
        df, info = prepare(xlsx, key)
        feats = info["features"]
        mets = cross_validate(df, feats, clf)
        scaler, model, n_smote = fit_final(df, feats, clf)
        print(f"  reactors={len(df)}  features={len(feats)}  "
              f"CV accuracy={mets['accuracy']} (sd {mets['accuracy_sd']})  "
              f"macro F1={mets['f1_macro']}  AUC={mets['auc_ovr_macro']}  "
              f"baselines={mets['baselines']}")
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
    path = sys.argv[1] if len(sys.argv) > 1 else "data/UCL_Dataset_v6_PRIS.xlsx"
    main(path)
