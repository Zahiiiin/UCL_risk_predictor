"""
UCL_Forecast_Benchmark.py
=========================
OPTION B, STEP 2: test how well next-year UCL can be forecast, with an honest time-ordered evaluation.

THE IDEA IN PLAIN WORDS ("rolling origin")
------------------------------------------
Imagine it is the end of 2014.  A forecaster has all the data up to 2014 and must predict each reactor's
UCL for 2015.  A year later the real 2015 values arrive, and the forecaster repeats the exercise for 2016, and so on.
We replay this for every target year from 2010 to 2025:

    for each target year Y:
        train on rows whose target year is BEFORE Y          (what was known at the time)
        predict the rows whose target year is Y               (the "future")
        keep the predictions; move on to Y + 1

All predictions are then collected and scored together.  Nothing from the future ever enters the training set.
This replaces the shuffled cross-validation of the paper, which is wrong for time series.

WHAT IS COMPARED
----------------
BASELINES (no learning, just rules):
  persistence     : next year = the reactor's last usable year
  mean3, mean5    : next year = average of its last 3 or 5 usable years
  median5         : next year = median of its last 5 usable years
  country_mean3   : next year = the 3-year average of all reactors in the same country
  fleet_mean      : next year = this year's average over all reactors in the world
MACHINE LEARNING (all fitted on log(1 + UCL) because UCL is very skewed, then converted back):
  Ridge (a linear model), RF (random forest), XGB (gradient boosting)
FEATURE SETS (what the models may look at; each set adds to the one before):
  A  own UCL history only
  B  A + own planned and external unavailability
  C  B + context (world average, country average)
  D  C + reactor age and size

SCORES
------
MAE        mean absolute error in percentage points of UCL (smaller is better; the main score)
Class acc  how often the predicted Low/Medium/High class equals the true class of that year
AUC_high   how well the predicted value ranks next year's "UCL above 6 %" cases (0.5 = chance, 1 = perfect)
Uncertainty: a bootstrap that resamples whole reactors (1000 times) gives a 95 % interval for each difference,
because rows of the same reactor are not independent.

GROUPED TESTS (--grouped country | site): the models never see ANY training row from the held-out country
(or site).  This asks "would it work for a place the model has not met?".

Usage
-----
  python3 research/UCL_Forecast_Benchmark.py                      # main run, all feature sets
  python3 research/UCL_Forecast_Benchmark.py --grouped country --sets A C
  python3 research/UCL_Forecast_Benchmark.py --window 15           # train only on the latest 15 years
  python3 research/UCL_Forecast_Benchmark.py --subset 387          # only the 387 reactors of the paper
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(HERE, "results_Forecast", "forecast_panel.csv")
SEED = 42
LOW, HIGH = 2.0, 6.0

FEATURES = {
    "A": ["ucl_l0", "ucl_l1", "ucl_l2", "ucl_last", "ucl_gap", "ucl_mean3", "ucl_mean5", "ucl_mean10", "ucl_med5",
          "ucl_std5", "ucl_max5", "n_usable5", "n_usable10", "frac_high5", "delta_3_10"],
}
FEATURES["B"] = FEATURES["A"] + ["puf_l0", "puf_mean3", "puf_mean5", "xuf_mean5", "n_shut5"]
FEATURES["C"] = FEATURES["B"] + ["fleet_mean_t", "fleet_med_t", "fleet_mean3", "country_mean3", "country_units_t"]
FEATURES["D"] = FEATURES["C"] + ["age", "log_mw"]
SET_LABEL = {"A": "A. own UCL history", "B": "B. A + planned/external", "C": "C. B + world and country context",
             "D": "D. C + age and size"}


def to_class(v):
    v = np.asarray(v, dtype=float)
    return np.where(v < LOW, 0, np.where(v <= HIGH, 1, 2))


# ----------------------------------------------------------------------------------------------- models
def make_model(name):
    if name == "Ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
    if name == "RF":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=-1,
                                     random_state=SEED)
    if name == "XGB":
        return XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                            min_child_weight=5, random_state=SEED, n_jobs=2, tree_method="hist", verbosity=0)
    raise ValueError(name)


def baselines(T):
    """Rule-based forecasts for the test rows T (a DataFrame).  No fitting."""
    return {
        "persistence": T.ucl_last.values,
        "mean3": T.ucl_mean3.fillna(T.ucl_last).values,
        "mean5": T.ucl_mean5.values,
        "median5": T.ucl_med5.values,
        "country_mean3": T.country_mean3.fillna(T.ucl_mean5).values,
        "fleet_mean": T.fleet_mean_t.values,
    }


# ----------------------------------------------------------------------------------------------- rolling origin
def fold_of_group(groups, k=6):
    """Spread groups over k folds: biggest groups first, round-robin, so each fold has a similar number of rows."""
    order = groups.value_counts().index.tolist()
    return {g: i % k for i, g in enumerate(order)}


def run(P, sets, models, first_test, last_test, window, grouped, k=6):
    out = []
    gcol = {"country": "country", "site": "site"}.get(grouped)
    fold_map = fold_of_group(P[gcol], k) if gcol else None
    for Y in range(first_test, last_test + 1):
        T = P[P.target_year == Y].copy()
        if T.empty:
            continue
        lo = (Y - window) if window else -10 ** 6
        TR = P[(P.target_year < Y) & (P.target_year >= lo)]
        res = T[["unit", "country", "site", "target_year", "y", "y_class", "in_paper_387"]].copy()
        for name, v in baselines(T).items():
            res["base:" + name] = v
        for sk in sets:
            cols = FEATURES[sk]
            for m in models:
                pred = np.full(len(T), np.nan)
                if gcol is None:
                    mdl = make_model(m).fit(TR[cols], np.log1p(TR.y))
                    pred = np.expm1(mdl.predict(T[cols]))
                else:
                    tf = T[gcol].map(fold_map).values
                    trf = TR[gcol].map(fold_map).values
                    for f in range(k):
                        sel = tf == f
                        if not sel.any():
                            continue
                        keep = trf != f           # drop every training row of the held-out groups
                        mdl = make_model(m).fit(TR.loc[keep, cols], np.log1p(TR.y[keep]))
                        pred[sel] = np.expm1(mdl.predict(T.loc[sel, cols]))
                res[f"ml:{m}|{sk}"] = np.clip(pred, 0, 100)
        out.append(res)
        print(f"  target year {Y}: train rows {len(TR):5d}, test rows {len(T):4d}", flush=True)
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------------------------- scoring
def method_columns(D):
    return [c for c in D.columns if c.startswith("base:") or c.startswith("ml:")]


def score_table(D, label=""):
    rows = []
    y = D.y.values
    yc = D.y_class.values
    hi = (y > HIGH).astype(int)
    for c in method_columns(D):
        p = D[c].values
        ok = ~np.isnan(p)
        e = np.abs(p[ok] - y[ok])
        pc = to_class(p[ok])
        # balanced accuracy = average of the recall of the three classes (so the majority class does not dominate)
        rec = [float((pc[yc[ok] == k] == k).mean()) for k in (0, 1, 2) if (yc[ok] == k).any()]
        try:
            auc = float(roc_auc_score(hi[ok], p[ok]))
        except ValueError:
            auc = np.nan
        rows.append({"Method": c, "n": int(ok.sum()), "MAE": e.mean(), "MedAE": float(np.median(e)),
                     "RMSE": float(np.sqrt((e ** 2).mean())), "ClassAcc": float((pc == yc[ok]).mean() * 100),
                     "BalancedAcc": float(np.mean(rec) * 100), "AUC_high": auc})
    return pd.DataFrame(rows)


def bootstrap_vs(D, ref, methods, B=1000):
    """Resample reactors with replacement; report the difference in pooled MAE and class accuracy to the reference."""
    rng = np.random.default_rng(SEED)
    units = D.unit.values
    uniq, inv = np.unique(units, return_inverse=True)
    U = len(uniq)
    y = D.y.values
    yc = D.y_class.values
    n_u = np.bincount(inv, minlength=U).astype(float)
    def per_unit(c):
        p = D[c].values
        ae = np.bincount(inv, weights=np.abs(p - y), minlength=U)
        co = np.bincount(inv, weights=(to_class(p) == yc).astype(float), minlength=U)
        return ae, co
    ae_ref, co_ref = per_unit(ref)
    W = rng.multinomial(U, np.ones(U) / U, size=B).astype(float)      # B x U resampling weights
    den = W @ n_u
    mae_ref = (W @ ae_ref) / den
    acc_ref = (W @ co_ref) / den * 100
    rows = []
    for c in methods:
        ae, co = per_unit(c)
        d_mae = (W @ ae) / den - mae_ref
        d_acc = (W @ co) / den * 100 - acc_ref
        rows.append({"Method": c, "Reference": ref,
                     "dMAE": float(ae.sum() / n_u.sum() - ae_ref.sum() / n_u.sum()),
                     "dMAE_lo": float(np.percentile(d_mae, 2.5)), "dMAE_hi": float(np.percentile(d_mae, 97.5)),
                     "dClassAcc": float(co.sum() / n_u.sum() * 100 - co_ref.sum() / n_u.sum() * 100),
                     "dClassAcc_lo": float(np.percentile(d_acc, 2.5)), "dClassAcc_hi": float(np.percentile(d_acc, 97.5))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-test", type=int, default=2010)
    ap.add_argument("--last-test", type=int, default=2025)
    ap.add_argument("--window", type=int, default=0, help="0 = expanding window; N = only the latest N target years")
    ap.add_argument("--grouped", choices=["none", "country", "site"], default="none")
    ap.add_argument("--sets", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--models", nargs="+", default=["Ridge", "RF", "XGB"])
    ap.add_argument("--subset", choices=["all", "387"], default="all")
    ap.add_argument("--tag", default="")
    ap.add_argument("--rescore", action="store_true",
                    help="do not refit anything; re-read the stored predictions of this tag and recompute all scores")
    a = ap.parse_args()
    P = pd.read_csv(PANEL)
    if a.subset == "387":
        P = P[P.in_paper_387 == 1].reset_index(drop=True)
    tag = a.tag or ("main" if (a.grouped == "none" and not a.window and a.subset == "all") else
                    "_".join(x for x in [f"grouped-{a.grouped}" if a.grouped != "none" else "",
                                         f"window{a.window}" if a.window else "",
                                         f"subset{a.subset}" if a.subset != "all" else ""] if x))
    outdir = os.path.join(HERE, "expected_results", "forecast", tag)
    os.makedirs(outdir, exist_ok=True)
    print(f"run '{tag}': panel rows {len(P)}, sets {a.sets}, models {a.models}, grouped {a.grouped}, window {a.window}")
    t0 = time.time()
    if a.rescore:
        D = pd.read_csv(os.path.join(outdir, "FB_predictions.csv.gz"))
    else:
        D = run(P, a.sets, a.models, a.first_test, a.last_test, a.window or None,
                None if a.grouped == "none" else a.grouped)
        D.round(4).to_csv(os.path.join(outdir, "FB_predictions.csv.gz"), index=False)
    print(f"done in {time.time() - t0:.0f} s")

    def periods(D):
        return {"2010-2025": D, "2018-2025": D[D.target_year >= 2018], "2010-2017": D[D.target_year <= 2017]}
    summ, boots = [], []
    for pname, d in periods(D).items():
        if d.empty:
            continue
        s = score_table(d)
        s.insert(0, "Period", pname)
        ref = s.loc[s.Method == "base:persistence", "MAE"].iloc[0]
        s["MAE_skill_vs_persistence_pct"] = (1 - s.MAE / ref) * 100
        summ.append(s)
        ml = [c for c in method_columns(d) if c.startswith("ml:")]
        for refname in ["base:persistence", "base:median5", "base:mean5", "base:country_mean3"]:
            b = bootstrap_vs(d, refname, [c for c in method_columns(d) if c != refname])
            b.insert(0, "Period", pname)
            boots.append(b)
    S = pd.concat(summ, ignore_index=True)
    B = pd.concat(boots, ignore_index=True)
    S.round(4).to_csv(os.path.join(outdir, "FB_summary.csv"), index=False)
    B.round(4).to_csv(os.path.join(outdir, "FB_bootstrap.csv"), index=False)
    # per target year MAE (so that one odd year, e.g. 2020, can be seen)
    by = []
    for Y, d in D.groupby("target_year"):
        s = score_table(d)
        s.insert(0, "target_year", Y)
        by.append(s[["target_year", "Method", "n", "MAE", "ClassAcc"]])
    pd.concat(by).round(4).to_csv(os.path.join(outdir, "FB_by_year.csv"), index=False)
    if a.rescore:
        print("rescored from stored predictions; FB_run.json left as it was")
        return
    json.dump({"args": vars(a), "rows_scored": int(len(D)), "reactors_scored": int(D.unit.nunique()),
               "seconds": round(time.time() - t0)}, open(os.path.join(outdir, "FB_run.json"), "w"), indent=2)
    show = S[S.Period == "2010-2025"].sort_values("MAE")
    print(show[["Method", "n", "MAE", "MedAE", "ClassAcc", "BalancedAcc", "AUC_high", "MAE_skill_vs_persistence_pct"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
