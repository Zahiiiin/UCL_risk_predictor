"""
UCL_Model2_History_Ablation.py
==============================================================================
How much does each UCF history period add to Model 2?

WHY THIS CHECK EXISTS
    The two history periods are UCF 2013-2017 and UCF 2010-2020. Both come from
    earlier IAEA editions, but their time windows overlap each other (2013-2017
    lies inside 2010-2020) and they also overlap the 2014-2024 window that defines
    the UCL label (2014-2017 and 2014-2020 respectively). So Model 2 uses recent
    performance data, not strictly earlier data. This script measures how much
    each period contributes and how strongly each one tracks the 2014-2024 UCF.

WHAT IT DOES
    1. Prints the correlation of each history period with UCF 2014-2024.
       UCF 2014-2024 is read here ONLY for this printout. It is never a model input.
    2. Trains the same tuned XGBoost as Model 2 on five feature sets and scores
       it on the same 74 test reactors as the main pipeline.
    3. Repeats the comparison with 5 x 5-fold cross-validation (SMOTE inside each
       fold, untuned XGBoost). A single 74-reactor test set moves by 1.35 points per
       reactor, so the repeated version is the steadier comparison.

HOW TO RUN (from the research folder)
    python UCL_Model2_History_Ablation.py
    python UCL_Model2_History_Ablation.py --quick     (fast smoke test)
==============================================================================
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import (RandomizedSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler

import UCL_Model2_Pipeline as M

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="results_Model2")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    hpo_iter = 4 if args.quick else M.HPO_ITER
    repeats = 1 if args.quick else 5

    xlsx = M.find_dataset(args.data)
    df, _, _ = M.load_data(xlsx)
    df = M.engineer(df)

    # 1. correlation with the label-defining window (diagnostic only)
    lg = pd.read_excel(xlsx, sheet_name="UCL_LONGITUDAL", header=0)
    lg["Reactor_Name"] = lg["Reactor_Name"].astype(str).str.strip()
    diag = df.merge(lg[["Reactor_Name", "UCF_2014_2024"]], on="Reactor_Name", how="left")
    print("Correlation with UCF 2014-2024 (the window that defines the label), n = %d" % len(diag))
    corr_rows = []
    for col, lab in [("UCF_P1", "UCF 2013-2017"), ("UCF_P2", "UCF 2010-2020")]:
        r = float(np.corrcoef(diag[col], diag["UCF_2014_2024"])[0, 1])
        corr_rows.append({"Period": lab, "Pearson_r_with_UCF_2014_2024": round(r, 3)})
        print("    %-14s r = %.3f" % (lab, r))
    r12 = float(np.corrcoef(diag["UCF_P1"], diag["UCF_P2"])[0, 1])
    print("    2013-2017 vs 2010-2020  r = %.3f" % r12)

    # 2. same split and scaling as the main pipeline
    sets = {
        "Design only (13)": M.FEAT_M1,
        "Design + UCF 2013-2017 (14)": M.FEAT_M1 + ["UCF_P1"],
        "Design + UCF 2010-2020 (14)": M.FEAT_M1 + ["UCF_P2"],
        "Design + both periods, no derived (15)": M.FEAT_M1 + ["UCF_P1", "UCF_P2"],
        "Full Model 2 (18)": M.FEAT_M2,
    }
    y = df["Target"]
    X = pd.DataFrame(StandardScaler().fit_transform(df[M.FEAT_M2]), columns=M.FEAT_M2)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=M.TEST_SIZE, stratify=y, random_state=M.RANDOM_STATE)
    kf = StratifiedKFold(M.CV_FOLDS, shuffle=True, random_state=M.RANDOM_STATE)
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats,
                                  random_state=M.RANDOM_STATE)

    rows = []
    print("\nTuned XGBoost on the main 74-reactor test set, and %d x 5-fold CV" % repeats)
    for name, feats in sets.items():
        Xs, ys = SMOTE(random_state=M.RANDOM_STATE, k_neighbors=3).fit_resample(X_tr[feats], y_tr)
        search = RandomizedSearchCV(
            M.make_xgb(), M.PARAM_GRIDS["XGB"], n_iter=hpo_iter, cv=kf,
            scoring="f1_weighted", random_state=M.RANDOM_STATE, n_jobs=-1)
        search.fit(Xs, ys)
        res, pred, _ = M.score_model(search.best_estimator_, X_te[feats], y_te)
        med_f1 = M.per_class_table(y_te, pred)["Medium"]["f1"]

        pipe = Pipeline([("smote", SMOTE(random_state=M.RANDOM_STATE, k_neighbors=3)),
                         ("xgb", M.make_xgb())])
        cv = cross_validate(pipe, X[feats], y, cv=rkf, n_jobs=-1,
                            scoring={"acc": "accuracy", "f1": "f1_weighted"})
        row = {"Feature_set": name, "Test_accuracy": res["Accuracy"],
               "Test_F1": res["F1"], "Test_AUC": res["AUC_ROC"],
               "Test_Medium_F1": med_f1,
               "CV_accuracy_mean": round(cv["test_acc"].mean() * 100, 2),
               "CV_accuracy_std": round(cv["test_acc"].std() * 100, 2),
               "CV_F1_mean": round(cv["test_f1"].mean() * 100, 2)}
        rows.append(row)
        print("  %-40s test acc %.2f  F1 %.2f  Medium F1 %.1f | CV acc %.2f +/- %.2f" %
              (name, row["Test_accuracy"], row["Test_F1"], med_f1,
               row["CV_accuracy_mean"], row["CV_accuracy_std"]))

    pd.DataFrame(rows).to_csv(os.path.join(args.out, "M2_table9_history_ablation.csv"),
                              index=False)
    pd.DataFrame(corr_rows).to_csv(
        os.path.join(args.out, "M2_table10_history_correlation.csv"), index=False)
    print("\nSaved M2_table9_history_ablation.csv and M2_table10_history_correlation.csv")


if __name__ == "__main__":
    main()
