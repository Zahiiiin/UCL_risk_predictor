"""
UCL_Threshold_Sensitivity.py
==============================================================================
Do the main findings depend on the class limits (2 % and 6 %)?

WHAT IT DOES
    The class limits are a choice of the authors. This script moves both limits together to five
    pairs (1 and 5, 1.5 and 5.5, 2 and 6, 2.5 and 6.5, 3 and 7 percent), re-labels all reactors from the
    numeric label UCL_Value, and repeats the benchmark for the two headline settings:
        design inputs only (set A of the benchmark, 387 reactors) and
        design plus past UCL (set D, 324 reactors),
    with Logistic Regression and Random Forest, the same 5 x 5-fold cross-validation and the same
    in-fold steps (scaling, country encoding, SMOTE). The baselines are the majority class, the
    majority class of the country, and (for set D) the persistence rule on the same limits.

    A reactor keeps its class under every pair only if its UCL_Value is far from all limits; the
    share of reactors whose class differs from the 2 and 6 percent label is written too.

HOW TO RUN (from the research folder)
    python UCL_Threshold_Sensitivity.py
    Writes results_Threshold/threshold_sensitivity.csv and threshold_class_counts.csv
==============================================================================
"""
import os
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

import UCL_Benchmark_CV as B

warnings.filterwarnings("ignore")
M = B.M
RS = B.RS
LIMITS = [(1.0, 5.0), (1.5, 5.5), (2.0, 6.0), (2.5, 6.5), (3.0, 7.0)]


def relabel(df, lo, hi):
    d = df.copy()
    v = d["UCL_Value"].round(2)
    d["Target"] = np.where(v < lo, 0, np.where(v <= hi, 1, 2))
    return d


def classify(v, lo, hi):
    return np.where(v < lo, 0, np.where(v <= hi, 1, 2))


def evaluate(df, feats, lo, hi, with_persistence, cv):
    y = df["Target"].values
    models = B.make_models()
    rows = []
    for fold, (tr, te) in enumerate(cv.split(df, y)):
        dtr, dte = df.iloc[tr].copy(), df.iloc[te].copy()
        ytr, yte = y[tr], y[te]
        maj = int(np.bincount(ytr).argmax())
        cmaj = dtr.groupby("Country")["Target"].agg(lambda s: int(s.value_counts().idxmax()))
        base = {"Baseline: majority class": np.full(len(te), maj),
                "Baseline: country majority": dte["Country"].map(cmaj).fillna(maj).astype(int).values}
        if with_persistence:
            base["Baseline: persistence (past UCL)"] = classify(((dte["UCL_P1"] + dte["UCL_P2"]) / 2).values, lo, hi)
        for name, pred in base.items():
            r = {"Model": name, "Fold": fold}; r.update(B.scores(yte, pred)); rows.append(r)
        Xtr = dtr[[f for f in feats if f != "Country_OOF"]].copy()
        Xte = dte[[f for f in feats if f != "Country_OOF"]].copy()
        if "Country_OOF" in feats:
            Xtr["Country_OOF"], Xte["Country_OOF"] = B.country_encoding(dtr, dte)
        Xtr, Xte = Xtr[feats], Xte[feats]
        sc = StandardScaler().fit(Xtr)
        Xtr_s = pd.DataFrame(sc.transform(Xtr), columns=feats)
        Xte_s = pd.DataFrame(sc.transform(Xte), columns=feats)
        Xsm, ysm = SMOTE(random_state=RS, k_neighbors=3).fit_resample(Xtr_s, ytr)
        for name in ("LR", "RF"):
            mdl = models[name](); mdl.fit(Xsm, ysm)
            pred = mdl.predict(Xte_s)
            r = {"Model": name, "Fold": fold}; r.update(B.scores(yte, pred)); rows.append(r)
    return pd.DataFrame(rows)


def main():
    out = "results_Threshold"
    os.makedirs(out, exist_ok=True)
    all_df = B.build_static_features(M.load_all(M.find_dataset(None)))
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RS)
    base_cls = classify(all_df["UCL_Value"].round(2).values, 2.0, 6.0)
    counts, res = [], []
    for lo, hi in LIMITS:
        d_all = relabel(all_df, lo, hi)
        sub = d_all[d_all["Model2_Subset"] == 1].reset_index(drop=True)
        cls = d_all["Target"].values
        counts.append({"Low_limit": lo, "High_limit": hi, "Low": int((cls == 0).sum()), "Medium": int((cls == 1).sum()),
                       "High": int((cls == 2).sum()), "majority_share_pct": round(float(np.bincount(cls).max() / len(cls) * 100), 1),
                       "share_changed_vs_2_6_pct": round(float((cls != base_cls).mean() * 100), 1)})
        for label, df, feats, pers in (("A. Design (13), 387", d_all, M.FEAT_M1, False), ("D. Model 2 (18), 324", sub, M.FEAT_M2, True)):
            print("limits", lo, hi, label, flush=True)
            r = evaluate(df, feats, lo, hi, pers, cv)
            g = r.groupby("Model", sort=False)[["accuracy", "balanced_accuracy", "macro_F1"]].agg(["mean", "std"])
            g.columns = ["%s_%s" % c for c in g.columns]
            g = g.round(2).reset_index()
            g.insert(0, "Set", label); g.insert(0, "High_limit", hi); g.insert(0, "Low_limit", lo)
            res.append(g)
    pd.DataFrame(counts).to_csv(os.path.join(out, "threshold_class_counts.csv"), index=False)
    pd.concat(res).to_csv(os.path.join(out, "threshold_sensitivity.csv"), index=False)
    print(pd.DataFrame(counts).to_string(index=False))
    print(pd.concat(res)[["Low_limit", "Set", "Model", "accuracy_mean", "balanced_accuracy_mean", "macro_F1_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
