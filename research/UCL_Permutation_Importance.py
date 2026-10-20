"""
UCL_Permutation_Importance.py
==============================================================================
Which inputs does the Random Forest really use? Permutation importance measured on
held-out reactors, in the same cross-validation as the benchmark.

WHY NOT JUST THE FOREST'S OWN IMPORTANCE?
    The impurity importance (MDI) that the Random Forest reports is computed on the
    training data and favours inputs with many distinct values, such as Country and
    Net Capacity (Strobl et al., 2007). Permutation importance asks a different and
    more direct question: how much does the accuracy on reactors the model has NOT seen
    fall when one input is shuffled?

HOW IT WORKS (nothing is tuned)
    For each of the 25 test folds of the benchmark (5 x repeated stratified 5-fold, the
    same folds), the pipeline of the benchmark is fitted on the training part (scaling,
    in-fold Country encoding, SMOTE with k = 3, Random Forest with 300 trees) and every
    input of the test part is shuffled 10 times, one at a time. The importance of an
    input is the mean fall in accuracy (percentage points) over the 10 shuffles and the
    25 folds. It is computed for Model 1 (13 design inputs, 324 reactors, set B of the
    benchmark) and for Model 2 (18 inputs, 324 reactors, set D).
    Inputs that are related (Age and Age squared, the history inputs) share their
    importance: shuffling one of them leaves the other as a substitute, so single values
    understate groups. The groups are therefore also shuffled together.

HOW TO RUN (from the research folder)
    python UCL_Permutation_Importance.py
    Writes results_Importance/PI_single.csv, PI_groups.csv and PI_run_summary.json
==============================================================================
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

import UCL_Benchmark_CV as B

warnings.filterwarnings("ignore")
M = B.M
RS = B.RS
N_SHUFFLE = 10

NICE = {
    "Country_OOF": "Country (UCL tendency)", "Type_enc": "Reactor type", "Net_Capacity_MW": "Net capacity (MW)",
    "CapCat_enc": "Capacity category", "Reactor_Age": "Reactor age (yr)", "Age_squared": "Age squared",
    "Operational_Years": "Operational years", "NSSS_Group_enc": "NSSS technology group",
    "Construction_Duration": "Construction duration (yr)", "Thermal_Efficiency": "Thermal efficiency (constant)",
    "Model_Family_enc": "Model family", "Age_Group": "Age group", "Op_Category": "Operational category",
    "UCL_P1": "UCL 2010-2013", "UCL_P2": "UCL 2014-2017", "UCL_change_P1_P2": "UCL change (P1 to P2)",
    "UCL_mean_2period": "UCL mean (2 periods)", "UCL_trend_dir": "UCL trend direction",
}
GROUPS = {
    "Country": ["Country_OOF"],
    "Age (age, age squared, age group)": ["Reactor_Age", "Age_squared", "Age_Group"],
    "Reactor type, family and supplier": ["Type_enc", "Model_Family_enc", "NSSS_Group_enc"],
    "Size (net capacity, capacity category)": ["Net_Capacity_MW", "CapCat_enc"],
    "Build (construction duration, operational years and category)": ["Construction_Duration", "Operational_Years", "Op_Category", "Thermal_Efficiency"],
    "Past UCL (all five history inputs)": ["UCL_P1", "UCL_P2", "UCL_change_P1_P2", "UCL_mean_2period", "UCL_trend_dir"],
}


def fold_matrices(df, feats, tr, te):
    dtr, dte = df.iloc[tr].copy(), df.iloc[te].copy()
    Xtr, Xte = dtr[[f for f in feats if f != "Country_OOF"]].copy(), dte[[f for f in feats if f != "Country_OOF"]].copy()
    if "Country_OOF" in feats:
        Xtr["Country_OOF"], Xte["Country_OOF"] = B.country_encoding(dtr, dte)
    Xtr, Xte = Xtr[feats], Xte[feats]
    sc = StandardScaler().fit(Xtr)
    Xtr_s = pd.DataFrame(sc.transform(Xtr), columns=feats)
    Xte_s = pd.DataFrame(sc.transform(Xte), columns=feats)
    Xsm, ysm = SMOTE(random_state=RS, k_neighbors=3).fit_resample(Xtr_s, dtr["Target"].values)
    return Xsm, ysm, Xte_s, dte["Target"].values


def drop_when_shuffled(model, X, y, cols, rng):
    base = (model.predict(X) == y).mean() * 100
    falls = []
    for _ in range(N_SHUFFLE):
        Xp = X.copy()
        perm = rng.permutation(len(X))
        Xp[cols] = X[cols].values[perm]
        falls.append(base - (model.predict(Xp) == y).mean() * 100)
    return float(np.mean(falls))


def run(sub, feats, label, cv):
    single, group = [], []
    for fold, (tr, te) in enumerate(cv.split(sub, sub["Target"].values)):
        Xsm, ysm, Xte, yte = fold_matrices(sub, feats, tr, te)
        mdl = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RS, n_jobs=-1).fit(Xsm, ysm)
        rng = np.random.RandomState(RS + fold)
        for f in feats:
            single.append({"Model": label, "Fold": fold, "Input": NICE[f], "Fall_in_accuracy_pts": drop_when_shuffled(mdl, Xte, yte, [f], rng)})
        for g, cols in GROUPS.items():
            cols = [c for c in cols if c in feats]
            if cols:
                group.append({"Model": label, "Fold": fold, "Group": g, "Fall_in_accuracy_pts": drop_when_shuffled(mdl, Xte, yte, cols, rng)})
        print(label, "fold", fold + 1, "of", cv.get_n_splits(), flush=True)
    return single, group


def main():
    out = "results_Importance"
    os.makedirs(out, exist_ok=True)
    all_df = B.build_static_features(M.load_all(M.find_dataset(None)))
    sub = all_df[all_df["Model2_Subset"] == 1].reset_index(drop=True)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RS)
    s1, g1 = run(sub, M.FEAT_M1, "Model 1 (design, 13)", cv)
    s2, g2 = run(sub, M.FEAT_M2, "Model 2 (design + history, 18)", cv)
    single, group = pd.DataFrame(s1 + s2), pd.DataFrame(g1 + g2)
    ss = single.groupby(["Model", "Input"], sort=False)["Fall_in_accuracy_pts"].agg(["mean", "std"]).round(2).reset_index()
    gg = group.groupby(["Model", "Group"], sort=False)["Fall_in_accuracy_pts"].agg(["mean", "std"]).round(2).reset_index()
    ss.to_csv(os.path.join(out, "PI_single.csv"), index=False)
    gg.to_csv(os.path.join(out, "PI_groups.csv"), index=False)
    with open(os.path.join(out, "PI_run_summary.json"), "w") as fh:
        json.dump({"folds": 25, "shuffles_per_input": N_SHUFFLE, "n_reactors": int(len(sub)),
                   "classifier": "RandomForest, 300 trees, balanced class weights, untuned"}, fh, indent=1)
    pd.set_option("display.width", 200)
    print(ss.sort_values(["Model", "mean"], ascending=[True, False]).to_string(index=False))
    print(gg.to_string(index=False))


if __name__ == "__main__":
    main()
