"""
UCL_Benchmark_Tuned.py
==============================================================================
PART I ROBUSTNESS CHECK: the same 25 outer test folds as UCL_Benchmark_CV.py, but with the settings of
every model searched INSIDE each outer training part (nested validation).

READ FIRST: UCL_Tuning_Spec.py and TUNING_PRESPECIFICATION.md. They fix the ranges, the 8 candidates,
the selection score (mean accuracy of 3 inner folds) and the decision rule. They were committed before this
script was run.

WHAT HAPPENS IN ONE OUTER FOLD
    1. The outer test fold is put aside. It is not touched until the very end.
    2. The outer training part is split into 3 inner folds (stratified).
       For every inner split the country encoding, the scaling and SMOTE (k = 3) are redone from the inner
       training part only, exactly as the outer pipeline does.
    3. LR, RF, ETC and XGB each try their 8 candidates (candidate 0 = the untuned setting).
       The candidate with the highest mean inner accuracy wins (ties go to the lower number).
    4. The winners are refitted on the whole outer training part (country encoding, scaling and SMOTE as in the
       untuned run) and predict the outer test fold.
    5. Stacking is built from the three tuned tree models of this fold with the fixed final logistic regression.

HOW TO RUN (from the project folder)
    python3 research/UCL_Benchmark_Tuned.py                      # all five sets, 25 outer folds (about 1 to 1.5 hours)
    python3 research/UCL_Benchmark_Tuned.py --quick --sets D     # smoke test: 1 repeat, set D only
    python3 research/UCL_Benchmark_Tuned.py --compare-only       # rebuild the comparison tables from stored fold scores
==============================================================================
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import UCL_Benchmark_CV as B
import UCL_Model2_Pipeline as M
import UCL_Tuning_Spec as S

warnings.filterwarnings("ignore")
RS = B.RS
assert RS == M.RANDOM_STATE == S.SEED == 42, "random seeds must agree with the untuned run"
HERE = os.path.dirname(os.path.abspath(__file__))
TUNED = ["LR", "RF", "ETC", "XGB"]
CAND = {m: S.part1_candidates(m) for m in TUNED}


# ------------------------------------------------------------------------------ building models
def build(model, params, n_trees=300):
    if model == "LR":
        return LogisticRegression(max_iter=2000, random_state=RS, **params)
    if model == "RF":
        return RandomForestClassifier(n_estimators=n_trees, class_weight="balanced", random_state=RS, n_jobs=-1, **params)
    if model == "ETC":
        return ExtraTreesClassifier(n_estimators=n_trees, class_weight="balanced", random_state=RS, n_jobs=-1, **params)
    if model == "XGB":
        return XGBClassifier(random_state=RS, eval_metric="mlogloss", verbosity=0, n_jobs=-1, **params)
    raise ValueError(model)


def build_stacking(best):
    """Same stack as M.make_stacking (200 trees, fixed final logistic regression, 5-fold), with the tuned base settings."""
    return StackingClassifier(
        estimators=[("rf", build("RF", best["RF"], n_trees=200)),
                    ("etc", build("ETC", best["ETC"], n_trees=200)),
                    ("xgb", build("XGB", best["XGB"]))],
        final_estimator=LogisticRegression(max_iter=2000, random_state=RS),
        cv=5, passthrough=False, n_jobs=-1)


# ------------------------------------------------------------------------------ data preparation (same steps as the untuned run)
def prepare(dtr, dte, feats):
    """Country encoding from the training part only, scaling fitted on the training part, SMOTE on the training part only."""
    Xtr = dtr[[f for f in feats if f != "Country_OOF"]].copy()
    Xte = dte[[f for f in feats if f != "Country_OOF"]].copy()
    if "Country_OOF" in feats:
        Xtr["Country_OOF"], Xte["Country_OOF"] = B.country_encoding(dtr, dte)
    Xtr, Xte = Xtr[feats], Xte[feats]
    sc = StandardScaler().fit(Xtr)
    Xtr_s = pd.DataFrame(sc.transform(Xtr), columns=feats)
    Xte_s = pd.DataFrame(sc.transform(Xte), columns=feats)
    Xsm, ysm = SMOTE(random_state=RS, k_neighbors=3).fit_resample(Xtr_s, dtr["Target"].values)
    return Xsm, ysm, Xte_s


def select_settings(dtr, feats):
    """Inner 3-fold search. Returns, per model, the winning candidate number and the table of inner accuracies."""
    dtr = dtr.reset_index(drop=True)
    ytr = dtr["Target"].values
    inner = StratifiedKFold(S.PART1_INNER_FOLDS, shuffle=True, random_state=RS)
    prepared = []
    for a, b in inner.split(dtr, ytr):
        Xsm, ysm, Xva = prepare(dtr.iloc[a], dtr.iloc[b], feats)
        prepared.append((Xsm, ysm, Xva, ytr[b]))
    table, win = {}, {}
    for m in TUNED:
        acc = np.zeros((len(CAND[m]), len(prepared)))
        for j, params in enumerate(CAND[m]):
            for i, (Xsm, ysm, Xva, yva) in enumerate(prepared):
                acc[j, i] = accuracy_score(yva, build(m, params).fit(Xsm, ysm).predict(Xva))
        mean = np.round(acc.mean(axis=1), 9)
        win[m] = int(np.argmax(mean))              # np.argmax returns the first maximum = lowest candidate number
        table[m] = mean
    return win, table


# ------------------------------------------------------------------------------ one feature set
def run_set(df, feats, label, cv, rows, chosen, max_folds=None):
    y = df["Target"].values
    for fold, (tr, te) in enumerate(cv.split(df, y)):
        if max_folds is not None and fold >= max_folds:
            break
        t0 = time.time()
        dtr, dte = df.iloc[tr].copy(), df.iloc[te].copy()
        ytr, yte = y[tr], y[te]
        n_tr, n_te = len(tr), len(te)
        maj = int(np.bincount(ytr).argmax())
        cmaj = dtr.groupby("Country")["Target"].agg(lambda s: int(s.value_counts().idxmax()))

        def add(name, pred, proba=None):
            r = {"Set": label, "Tag": "tuned", "Model": name, "Fold": fold, "n_train": n_tr, "n_test": n_te}
            r.update(B.scores(yte, pred, proba))
            rows.append(r)

        # baselines are recomputed so that the folds can be checked against the untuned file
        add("Baseline: majority class", np.full(n_te, maj))
        add("Baseline: country majority", dte["Country"].map(cmaj).fillna(maj).astype(int).values)
        if label.startswith("B"):          # the untuned run scores persistence only on set B
            v = ((dte["UCL_P1"] + dte["UCL_P2"]) / 2).values
            add("Baseline: persistence (past UCL)", np.where(v < 2, 0, np.where(v <= 6, 1, 2)))

        win, table = select_settings(dtr, feats)
        Xsm, ysm, Xte_s = prepare(dtr, dte, feats)
        best = {}
        for m in TUNED:
            params = CAND[m][win[m]]
            best[m] = params
            mdl = build(m, params).fit(Xsm, ysm)
            add(m + "-tuned", mdl.predict(Xte_s), mdl.predict_proba(Xte_s))
            chosen.append({"Set": label, "Fold": fold, "Model": m, "candidate": win[m],
                           "inner_acc_selected": float(table[m][win[m]]), "inner_acc_default": float(table[m][0]),
                           "params": json.dumps(params)})
        st = build_stacking(best).fit(Xsm, ysm)
        add("Stacking-tuned", st.predict(Xte_s), st.predict_proba(Xte_s))
        print("  %s | fold %2d | chosen %s | %.0f s" % (label[:2], fold, {m: win[m] for m in TUNED}, time.time() - t0), flush=True)


# ------------------------------------------------------------------------------ comparison with the untuned results
def compare(res, chosen, out_dir, untuned_path):
    un = pd.read_csv(untuned_path)
    checks = {}
    # 1. same folds? baseline fold scores must be identical to the untuned file
    bl = res[res.Model.str.startswith("Baseline")]
    mm = 0
    n_cmp = 0
    for _, r in bl.iterrows():
        u = un[(un.Set == r.Set) & (un.Model == r.Model) & (un.Fold == r.Fold)]
        if len(u) != 1:
            continue
        n_cmp += 1
        if (abs(u.iloc[0].accuracy - r.accuracy) > 1e-9) or int(u.iloc[0].n_test) != int(r.n_test) or int(u.iloc[0].n_train) != int(r.n_train):
            mm += 1
    checks["baseline_rows_compared_with_untuned"] = n_cmp
    checks["baseline_rows_that_differ"] = mm
    # 2. when candidate 0 (= the untuned setting) was chosen, the tuned fold score must equal the untuned fold score
    d0 = d0bad = 0
    for _, r in chosen[chosen.candidate == 0].iterrows():
        a = res[(res.Set == r.Set) & (res.Model == r.Model + "-tuned") & (res.Fold == r.Fold)]
        u = un[(un.Set == r.Set) & (un.Model == r.Model) & (un.Fold == r.Fold)]
        if len(a) == 1 and len(u) == 1:
            d0 += 1
            if abs(a.iloc[0].accuracy - u.iloc[0].accuracy) > 1e-9 or abs(a.iloc[0].macro_F1 - u.iloc[0].macro_F1) > 1e-9:
                d0bad += 1
    checks["default_chosen_cases_compared"] = d0
    checks["default_chosen_cases_that_differ_from_untuned"] = d0bad
    # same for stacking: all three tree models at default
    piv = chosen.pivot_table(index=["Set", "Fold"], columns="Model", values="candidate")
    s0 = s0bad = 0
    for (lab, fold), r in piv.iterrows():
        if r[["RF", "ETC", "XGB"]].sum() == 0:
            a = res[(res.Set == lab) & (res.Model == "Stacking-tuned") & (res.Fold == fold)]
            u = un[(un.Set == lab) & (un.Model == "Stacking") & (un.Fold == fold)]
            if len(a) == 1 and len(u) == 1:
                s0 += 1
                if abs(a.iloc[0].accuracy - u.iloc[0].accuracy) > 1e-9:
                    s0bad += 1
    checks["stacking_all_default_cases_compared"] = s0
    checks["stacking_all_default_cases_that_differ"] = s0bad
    json.dump(checks, open(os.path.join(out_dir, "BT_integrity_checks.json"), "w"), indent=1)
    print("integrity checks:", checks)

    metrics = ["accuracy", "balanced_accuracy", "macro_F1", "weighted_F1", "F1_Low", "F1_Medium", "F1_High",
               "within_one_class", "AUC_macro_OvR"]
    mod = res[~res.Model.str.startswith("Baseline")]
    g = mod.groupby(["Set", "Model"], sort=False)[metrics]
    summ = g.mean().round(2).join(g.std().round(2), rsuffix="_sd").reset_index()
    summ.to_csv(os.path.join(out_dir, "BT_summary.csv"), index=False)

    # tuned versus untuned, same classifier, same set, same folds
    rows = []
    for (lab, name), d in mod.groupby(["Set", "Model"], sort=False):
        base = name.replace("-tuned", "")
        for metric in ("accuracy", "balanced_accuracy", "macro_F1"):
            a = d.sort_values("Fold")[metric].values
            u = un[(un.Set == lab) & (un.Model == base)].sort_values("Fold")
            u = u[u.Fold.isin(d.Fold)][metric].values
            if len(u) != len(a):
                continue
            diff, t, p = B.nadeau_bengio(a - u, int(d.n_train.iloc[0]), int(d.n_test.iloc[0]))
            rows.append({"Set": lab.split(",")[0], "Model": base, "Metric": metric, "Untuned_mean": round(u.mean(), 2),
                         "Tuned_mean": round(a.mean(), 2), "Mean_difference_pts": round(diff, 2),
                         "t": round(t, 3) if t == t else np.nan, "p_corrected": round(p, 4), "n_folds": len(a)})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "BT_vs_untuned.csv"), index=False)

    # the paper's main comparisons, repeated with the tuned models
    L = {lab[0]: lab for lab in dict.fromkeys(res["Set"])}
    CB, PB = "Baseline: country majority", "Baseline: persistence (past UCL)"

    def vec(label, model, metric):
        r = res[(res.Set == label) & (res.Model == model)].sort_values("Fold")
        return r[metric].values, int(r.n_train.iloc[0]), int(r.n_test.iloc[0])

    pairs = []
    for m in ["LR", "RF", "ETC", "XGB", "Stacking"]:
        mt = m + "-tuned"
        if all(k in L for k in "AB"):
            pairs += [("B", mt, "B", CB, "design model vs country baseline (324)"),
                      ("A", mt, "A", CB, "design model vs country baseline (387)")]
        if all(k in L for k in "BD"):
            pairs += [("D", mt, "B", mt, "Model 2 vs Model 1, same classifier (324)"),
                      ("D", mt, "B", CB, "Model 2 vs country baseline (324)"),
                      ("D", mt, "B", PB, "Model 2 vs persistence baseline (324)")]
        if all(k in L for k in "BC"):
            pairs += [("B", mt, "C", mt, "design with vs without country (324)")]
    out = []
    for s1, m1, s2, m2, what in pairs:
        for metric in ("accuracy", "balanced_accuracy", "macro_F1"):
            a, ntr, nte = vec(L[s1], m1, metric)
            b, _, _ = vec(L[s2], m2, metric)
            if len(a) != len(b):
                continue
            diff, t, p = B.nadeau_bengio(a - b, ntr, nte)
            out.append({"Comparison": what, "First": "%s | %s" % (s1, m1), "Second": "%s | %s" % (s2, m2), "Metric": metric,
                        "Mean_difference_pts": round(diff, 2), "t": round(t, 3) if t == t else np.nan,
                        "p_corrected": round(p, 4)})
    pd.DataFrame(out).to_csv(os.path.join(out_dir, "BT_paired_tests.csv"), index=False)

    # which settings were chosen
    ch = chosen.groupby(["Set", "Model", "candidate"]).size().rename("times_chosen").reset_index()
    ch.to_csv(os.path.join(out_dir, "BT_choices.csv"), index=False)
    pd.set_option("display.width", 250)
    print("\nTuned versus untuned, accuracy:")
    print(pd.DataFrame(rows).query("Metric == 'accuracy'").drop(columns="Metric").to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "expected_results", "tuning", "part1"))
    ap.add_argument("--untuned", default=os.path.join(HERE, "expected_results", "benchmark", "BM_fold_scores.csv"))
    ap.add_argument("--quick", action="store_true", help="1 repeat (5 outer folds) instead of 5; code test only")
    ap.add_argument("--max-folds", type=int, default=None, help="stop after this many outer folds (code test only)")
    ap.add_argument("--sets", default=S.PART1_SETS)
    ap.add_argument("--compare-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.compare_only:
        res = pd.read_csv(os.path.join(a.out, "BT_fold_scores.csv"))
        chosen = pd.read_csv(os.path.join(a.out, "BT_chosen.csv"))
        compare(res, chosen, a.out, a.untuned)
        return
    repeats = 1 if a.quick else 5
    all_df = B.build_static_features(M.load_all(M.find_dataset(a.data)))
    sub = all_df[all_df["Model2_Subset"] == 1].reset_index(drop=True)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=RS)
    plan = [
        (all_df, M.FEAT_M1, "A. Design (13), all 387 reactors"),
        (sub, M.FEAT_M1, "B. Design (13), 324 Model 2 reactors"),
        (sub, B.DESIGN_NO_COUNTRY, "C. Design without country (12), 324 reactors"),
        (sub, M.FEAT_M2, "D. Model 2: design + history (18), 324 reactors"),
        (sub, M.FEAT_TEMPORAL, "E. History only (5), 324 reactors"),
    ]
    plan = [p for p in plan if p[2][0] in a.sets]
    rows, chosen = [], []
    t0 = time.time()
    for df, feats, label in plan:
        print("running:", label, flush=True)
        run_set(df, feats, label, cv, rows, chosen, a.max_folds)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "BT_fold_scores.csv"), index=False)     # saved after every set
        pd.DataFrame(chosen).to_csv(os.path.join(a.out, "BT_chosen.csv"), index=False)
        print("  elapsed %.0f min" % ((time.time() - t0) / 60), flush=True)
    res, ch = pd.DataFrame(rows), pd.DataFrame(chosen)
    compare(res, ch, a.out, a.untuned)
    import sklearn
    import xgboost
    json.dump({"repeats": repeats, "folds": 5 * repeats if a.max_folds is None else a.max_folds, "sets": a.sets,
               "seconds": round(time.time() - t0),
               "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                            "xgboost": xgboost.__version__}}, open(os.path.join(a.out, "BT_run_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
