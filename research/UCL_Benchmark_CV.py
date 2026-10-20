"""
UCL_Benchmark_CV.py
==============================================================================
Fair comparison of the UCL models with simple baselines, by repeated
cross-validation. Run it after UCL_Model1_Pipeline.py and UCL_Model2_Pipeline.py.

WHY THIS CHECK EXISTS
    One 80/20 split of 324 or 387 reactors is a small test set: one reactor moves
    the accuracy by about one point. This script scores every model on 25 test
    folds (5 x repeated stratified 5-fold cross-validation) and compares it with
    baselines that need no learning, so the paper can say how much the models
    really add.

WHAT IT DOES (nothing is tuned, so no score is selected on its own test fold)
    Baselines   majority class | majority class of the other reactors of the same
                country | persistence (class of the mean past UCL, Model 2 reactors only)
    Models      Logistic regression, Random Forest, Extra Trees, XGBoost, Stacking
                (same settings as the pipelines, no hyperparameter search)
    Feature sets
                A. Design (13 features)            on all 387 reactors
                B. Design (13 features)            on the 324 Model 2 reactors
                C. Design minus Country (12)       on the 324 reactors
                D. Model 2: design + history (18)  on the 324 reactors
                E. History only (5)                on the 324 reactors
    Inside every training fold: scaling is fitted on the training part, the Country
    encoding (mean class of the other reactors of that country, out-of-fold within the
    training part) is computed from the training part only, and SMOTE (k = 3) is applied
    to the training part only. The test fold never influences any of these.
    Scores: accuracy, balanced accuracy, macro F1, weighted F1, per-class F1, AUC
    (one-vs-rest, macro), and "within one class" (share of predictions that are not a
    Low/High mix-up). Model pairs are compared with the Nadeau-Bengio corrected
    resampled t-test on the fold accuracies.

HOW TO RUN (from the research folder)
    python UCL_Benchmark_CV.py
    python UCL_Benchmark_CV.py --quick      (2 repeats, fast smoke test)
==============================================================================
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

import UCL_Model2_Pipeline as M

warnings.filterwarnings("ignore")
RS = 42
CLASSES = ["Low", "Medium", "High"]
DESIGN_NO_COUNTRY = [f for f in M.FEAT_M1 if f != "Country_OOF"]


# ------------------------------------------------------------------------------
# FEATURES (everything except the country encoding, which is built inside each fold)
# ------------------------------------------------------------------------------
def build_static_features(df):
    d = df.copy()
    d["Target"] = d["UCL_Class"].map({"Low": 0, "Medium": 1, "High": 2})
    d["Type_enc"] = LabelEncoder().fit_transform(d["Type"])
    d["CapCat_enc"] = LabelEncoder().fit_transform(d["Capacity_Category"])
    d["Age_squared"] = d["Reactor_Age"] ** 2
    d["NSSS_Group"] = d["NSSS_Supplier"].map(M.NSSS_GROUP_MAP).fillna("Other")
    d["NSSS_Group_enc"] = LabelEncoder().fit_transform(d["NSSS_Group"])
    d["Thermal_Efficiency"] = 33.0   # constant, as in the pipelines
    d["Model_Family"] = d["Reactor_Model"].apply(M.model_family)
    d["Model_Family_enc"] = LabelEncoder().fit_transform(d["Model_Family"])
    d["Age_Group"] = d["Reactor_Age"].apply(M.age_group)
    d["Op_Category"] = d["Data_Category"].map({"A": 2, "B": 1, "C": 0})
    if "UCL_2010_2013" in d.columns:
        d["UCL_P1"] = d["UCL_2010_2013"]
        d["UCL_P2"] = d["UCL_2014_2017"]
        d["UCL_change_P1_P2"] = d["UCL_P2"] - d["UCL_P1"]
        d["UCL_mean_2period"] = (d["UCL_P1"] + d["UCL_P2"]) / 2
        d["UCL_trend_dir"] = d["UCL_change_P1_P2"].apply(M.trend_direction)
    return d


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


# ------------------------------------------------------------------------------
# MODELS (same settings as the pipelines, untuned)
# ------------------------------------------------------------------------------
def make_models():
    xgb = M.make_xgb
    return {
        "LR": lambda: LogisticRegression(max_iter=2000, random_state=RS),
        "RF": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                             random_state=RS, n_jobs=-1),
        "ETC": lambda: ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                            random_state=RS, n_jobs=-1),
        "XGB": xgb,
        "Stacking": M.make_stacking,
    }


def scores(y_true, pred, proba=None):
    f1 = f1_score(y_true, pred, labels=[0, 1, 2], average=None, zero_division=0)
    out = {
        "accuracy": accuracy_score(y_true, pred) * 100,
        "balanced_accuracy": balanced_accuracy_score(y_true, pred) * 100,
        "macro_F1": f1_score(y_true, pred, average="macro", zero_division=0) * 100,
        "weighted_F1": f1_score(y_true, pred, average="weighted", zero_division=0) * 100,
        "F1_Low": f1[0] * 100, "F1_Medium": f1[1] * 100, "F1_High": f1[2] * 100,
        "within_one_class": float(np.mean(np.abs(np.asarray(pred) - np.asarray(y_true)) <= 1)) * 100,
        "AUC_macro_OvR": np.nan,
    }
    if proba is not None:
        try:
            out["AUC_macro_OvR"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro") * 100
        except ValueError:
            pass
    return out


def run_set(df, feats, label, use_models, baselines, cv, tag, rows, preds_store):
    """Evaluate one feature set on one set of reactors."""
    y = df["Target"].values
    models = make_models()
    for fold, (tr, te) in enumerate(cv.split(df, y)):
        dtr, dte = df.iloc[tr].copy(), df.iloc[te].copy()
        ytr, yte = y[tr], y[te]
        n_tr, n_te = len(tr), len(te)
        maj = int(np.bincount(ytr).argmax())
        cmaj = dtr.groupby("Country")["Target"].agg(lambda s: int(s.value_counts().idxmax()))

        def add(name, pred, proba=None):
            r = {"Set": label, "Tag": tag, "Model": name, "Fold": fold, "n_train": n_tr, "n_test": n_te}
            r.update(scores(yte, pred, proba))
            rows.append(r)
            preds_store.setdefault((label, name), []).append((te, np.asarray(pred)))

        if "majority" in baselines:
            add("Baseline: majority class", np.full(n_te, maj))
        if "country" in baselines:
            add("Baseline: country majority", dte["Country"].map(cmaj).fillna(maj).astype(int).values)
        if "persistence" in baselines:
            v = ((dte["UCL_P1"] + dte["UCL_P2"]) / 2).values
            add("Baseline: persistence (past UCL)", np.where(v < 2, 0, np.where(v <= 6, 1, 2)))
        if not use_models:
            continue
        Xtr, Xte = dtr[[f for f in feats if f != "Country_OOF"]].copy(), dte[[f for f in feats if f != "Country_OOF"]].copy()
        if "Country_OOF" in feats:
            Xtr["Country_OOF"], Xte["Country_OOF"] = country_encoding(dtr, dte)
        Xtr, Xte = Xtr[feats], Xte[feats]
        sc = StandardScaler().fit(Xtr)
        Xtr_s = pd.DataFrame(sc.transform(Xtr), columns=feats)
        Xte_s = pd.DataFrame(sc.transform(Xte), columns=feats)
        Xsm, ysm = SMOTE(random_state=RS, k_neighbors=3).fit_resample(Xtr_s, ytr)
        for name, make in models.items():
            mdl = make()
            mdl.fit(Xsm, ysm)
            add(name, mdl.predict(Xte_s), mdl.predict_proba(Xte_s))


def nadeau_bengio(d, n_train, n_test):
    """Corrected resampled t-test for paired fold scores (Nadeau and Bengio, 2003)."""
    k = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return np.mean(d), np.nan, 1.0
    t = np.mean(d) / np.sqrt((1.0 / k + n_test / n_train) * var)
    return np.mean(d), t, float(2 * stats.t.sf(abs(t), k - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="results_Benchmark")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    repeats = 2 if args.quick else 5
    xlsx = M.find_dataset(args.data)
    all_df = build_static_features(M.load_all(xlsx))
    sub = all_df[all_df["Model2_Subset"] == 1].reset_index(drop=True)
    print("all reactors: %d | Model 2 reactors: %d | folds: %d" % (len(all_df), len(sub), 5 * repeats), flush=True)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=RS)

    rows, preds = [], {}
    plan = [
        (all_df, M.FEAT_M1, "A. Design (13), all 387 reactors", True, ["majority", "country"], "all"),
        (sub, M.FEAT_M1, "B. Design (13), 324 Model 2 reactors", True, ["majority", "country", "persistence"], "sub"),
        (sub, DESIGN_NO_COUNTRY, "C. Design without country (12), 324 reactors", True, [], "sub"),
        (sub, M.FEAT_M2, "D. Model 2: design + history (18), 324 reactors", True, [], "sub"),
        (sub, M.FEAT_TEMPORAL, "E. History only (5), 324 reactors", True, [], "sub"),
    ]
    for df, feats, label, use_models, bl, tag in plan:
        print("running:", label, flush=True)
        run_set(df, feats, label, use_models, bl, cv, tag, rows, preds)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "BM_fold_scores.csv"), index=False)

    # summary table
    metrics = ["accuracy", "balanced_accuracy", "macro_F1", "weighted_F1", "F1_Low", "F1_Medium", "F1_High",
               "within_one_class", "AUC_macro_OvR"]
    g = res.groupby(["Set", "Model"], sort=False)[metrics]
    summ = g.mean().round(2).join(g.std().round(2), rsuffix="_sd").reset_index()
    summ.to_csv(os.path.join(args.out, "BM_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print("\nMean over %d test folds (accuracy, macro F1, Medium F1, AUC, within-one-class)" % (5 * repeats))
    show = summ[["Set", "Model", "accuracy", "accuracy_sd", "macro_F1", "F1_Medium", "AUC_macro_OvR", "within_one_class"]]
    print(show.to_string(index=False))

    # paired comparisons (same folds)
    def fold_vec(label, model):
        r = res[(res["Set"] == label) & (res["Model"] == model)].sort_values("Fold")
        return r["accuracy"].values, int(r["n_train"].iloc[0]), int(r["n_test"].iloc[0])

    pairs = []
    A, B, C, D, E = [p[2] for p in plan]
    best_names = ["LR", "RF", "ETC", "XGB", "Stacking"]
    for m in best_names:
        pairs += [(B, m, B, "Baseline: country majority", "design model vs country baseline (324)"),
                  (D, m, B, m, "Model 2 vs Model 1, same classifier (324)"),
                  (D, m, B, "Baseline: persistence (past UCL)", "Model 2 vs persistence baseline (324)"),
                  (B, m, C, m, "design with vs without country (324)")]
        pairs += [(A, m, A, "Baseline: country majority", "design model vs country baseline (387)")]
    out = []
    for s1, m1, s2, m2, what in pairs:
        a, ntr, nte = fold_vec(s1, m1)
        b, _, _ = fold_vec(s2, m2)
        diff, t, p = nadeau_bengio(a - b, ntr, nte)
        out.append({"Comparison": what, "First": "%s | %s" % (s1.split(",")[0], m1), "Second": "%s | %s" % (s2.split(",")[0], m2),
                    "Mean_accuracy_difference_pts": round(diff, 2), "t": round(t, 3) if t == t else np.nan,
                    "p_corrected": round(p, 4)})
    pt = pd.DataFrame(out)
    pt.to_csv(os.path.join(args.out, "BM_paired_tests.csv"), index=False)
    print("\nPaired comparisons of fold accuracy (Nadeau-Bengio corrected t-test)")
    print(pt.to_string(index=False))

    # pooled confusion matrices (all test folds added up) for the main settings
    cms = {}
    for (label, name), lst in preds.items():
        if name in ("XGB", "RF", "LR") or name.startswith("Baseline"):
            df_ = all_df if label == A else sub
            cm = np.zeros((3, 3), int)
            for te, pr in lst:
                yt = df_["Target"].values[te]
                for t_, p_ in zip(yt, pr):
                    cm[t_, p_] += 1
            cms["%s | %s" % (label, name)] = cm.tolist()
    with open(os.path.join(args.out, "BM_confusion_matrices.json"), "w") as fh:
        json.dump({"classes": CLASSES, "rows_are_true_cols_are_predicted": cms}, fh, indent=1)
    import sklearn
    import xgboost
    with open(os.path.join(args.out, "BM_run_summary.json"), "w") as fh:
        json.dump({"repeats": repeats, "folds": 5 * repeats, "n_all": int(len(all_df)), "n_model2": int(len(sub)),
                   "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                                "xgboost": xgboost.__version__}}, fh, indent=1)
    print("\nSaved BM_* files in", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
