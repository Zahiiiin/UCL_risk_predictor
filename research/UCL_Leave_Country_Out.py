"""
UCL_Leave_Country_Out.py
==============================================================================
Can the models predict the class of a reactor in a country they have never seen?

WHY THIS CHECK EXISTS
    In the benchmark, reactors of the same country appear in both the training and the test
    part, so the Country input (the mean class of the other reactors of the country) and
    the design inputs can lean on neighbours. Leave-one-country-out validation removes that:
    every reactor is predicted by a model trained on reactors of OTHER countries only.
    Country cannot be used (there is no training reactor from the country), so the design
    inputs without Country (12 inputs) are used. Each reactor is tested exactly once, so the
    scores below are pooled over all reactors. Nothing is tuned.

BASELINES
    Majority class of the training countries; persistence of past UCL (Model 2 reactors only).

HOW TO RUN (from the research folder)
    python UCL_Leave_Country_Out.py
    Writes results_LeaveCountryOut/LCO_summary.csv and LCO_predictions.csv (class predicted for every reactor)
==============================================================================
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

import UCL_Benchmark_CV as B

warnings.filterwarnings("ignore")
M = B.M
RS = B.RS


def run(df, feats, label, rows, preds, countries):
    y = df["Target"].values
    models = B.make_models()
    out = {m: np.full(len(df), -1) for m in list(models) + ["majority"]}
    for c in countries:
        te = np.where(df["Country"].values == c)[0]
        tr = np.where(df["Country"].values != c)[0]
        if len(te) == 0:
            continue
        ytr = y[tr]
        out["majority"][te] = int(np.bincount(ytr).argmax())
        Xtr, Xte = df.iloc[tr][feats], df.iloc[te][feats]
        sc = StandardScaler().fit(Xtr)
        Xtr_s = pd.DataFrame(sc.transform(Xtr), columns=feats)
        Xte_s = pd.DataFrame(sc.transform(Xte), columns=feats)
        Xsm, ysm = SMOTE(random_state=RS, k_neighbors=3).fit_resample(Xtr_s, ytr)
        for name, make in models.items():
            out[name][te] = make().fit(Xsm, ysm).predict(Xte_s)
    for name, pred in out.items():
        rows.append(score(label, "Baseline: majority class (training countries)" if name == "majority" else name, y, pred))
        preds[(label, name)] = pred


def score(label, name, y, pred):
    f1 = f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {"Set": label, "Model": name, "n": len(y), "accuracy": round(accuracy_score(y, pred) * 100, 2),
            "balanced_accuracy": round(balanced_accuracy_score(y, pred) * 100, 2),
            "macro_F1": round(f1_score(y, pred, average="macro", zero_division=0) * 100, 2),
            "F1_Low": round(f1[0] * 100, 2), "F1_Medium": round(f1[1] * 100, 2), "F1_High": round(f1[2] * 100, 2)}


def main():
    out = "results_LeaveCountryOut"
    os.makedirs(out, exist_ok=True)
    all_df = B.build_static_features(M.load_all(M.find_dataset(None)))
    sub = all_df[all_df["Model2_Subset"] == 1].reset_index(drop=True)
    rows, preds = [], {}
    countries = sorted(all_df["Country"].unique())
    run(all_df, B.DESIGN_NO_COUNTRY, "A. Design without country (12), all 387 reactors", rows, preds, countries)
    run(sub, B.DESIGN_NO_COUNTRY, "C. Design without country (12), 324 reactors", rows, preds, countries)
    run(sub, B.DESIGN_NO_COUNTRY + M.FEAT_TEMPORAL, "I. Model 2 without country (17), 324 reactors", rows, preds, countries)
    run(sub, M.FEAT_TEMPORAL, "E. History only (5), 324 reactors", rows, preds, countries)
    y = sub["Target"].values
    v = ((sub["UCL_P1"] + sub["UCL_P2"]) / 2).values
    rows.append(score("C. Design without country (12), 324 reactors", "Baseline: persistence (past UCL)", y,
                      np.where(v < 2, 0, np.where(v <= 6, 1, 2))))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "LCO_summary.csv"), index=False)
    # predicted and true class of every reactor (one row per set, classifier and reactor)
    parts = []
    for (label, name), pred in preds.items():
        d = all_df if label.startswith("A.") else sub
        parts.append(pd.DataFrame({"Set": label, "Model": name, "Reactor_Name": d["Reactor_Name"].values,
                                   "Country": d["Country"].values, "True": d["Target"].values, "Pred": pred}))
    pd.concat(parts).to_csv(os.path.join(out, "LCO_predictions.csv"), index=False)
    pd.set_option("display.width", 220)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
