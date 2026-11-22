"""
US_Daily_Forecast.py
====================
US sub-study (pre-specified in US_DAILY_PRESPEC.md): do descriptors of the past DAILY power history
(NRC Power Reactor Status Reports) improve the one-year-ahead UCL forecast beyond the annual PRIS inputs?
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from UCL_Forecast_Benchmark import FEATURES, PANEL, SEED, make_model, to_class

OUT = os.path.join(HERE, "expected_results", "us_daily")
MIN_DAYS = 300
DAILY = ["d_nshort3", "d_nshort1", "d_lshort3", "d_lshort1", "d_lpart3", "d_lpart1", "d_nred3", "d_dlong3", "d_longest1"]


def daily_features():
    UY = pd.read_csv(os.path.join(HERE, "results_US", "US_unit_year_daily_descriptors.csv"))
    UY = UY[UY.n_days >= MIN_DAYS]
    rows = []
    for u, g in UY.groupby("unit"):
        g = g.set_index("year")
        for t in range(2007, 2025):
            w = g.loc[[y for y in (t - 2, t - 1, t) if y in g.index]]
            if len(w) == 0:
                continue
            last = g.loc[t] if t in g.index else None
            rows.append({"unit": u, "origin_year": t,
                         "d_nshort3": w.n_short_shutdowns.mean(), "d_lshort3": w.loss_short_zero.mean(),
                         "d_lpart3": w.loss_partial.mean(), "d_nred3": w.n_reductions.mean(),
                         "d_dlong3": w.days_long_shutdowns.mean(),
                         "d_nshort1": np.nan if last is None else last.n_short_shutdowns,
                         "d_lshort1": np.nan if last is None else last.loss_short_zero,
                         "d_lpart1": np.nan if last is None else last.loss_partial,
                         "d_longest1": np.nan if last is None else last.longest_shutdown})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    P = pd.read_csv(PANEL)
    P = P[P.country.str.startswith("UNITED STATES") & (P.origin_year >= 2007)]
    DF = daily_features()
    P = P.merge(DF, on=["unit", "origin_year"], how="left")
    P = P[P[DAILY[0]].notna()].reset_index(drop=True)          # a daily history is needed
    FB = pd.read_csv(os.path.join(HERE, "expected_results", "forecast", "main_window15", "FB_predictions.csv.gz"))
    C, CD = FEATURES["C"], FEATURES["C"] + DAILY
    rows, imps = [], []
    for Y in range(2011, 2026):
        T, TR = P[P.target_year == Y], P[P.target_year < Y]
        res = T[["unit", "target_year", "y", "y_class"]].copy()
        res["base:median5"] = T.ucl_med5.values
        for m in ("RF", "Ridge"):
            for k, cols in (("C", C), ("C+daily", CD)):
                mdl = make_model(m).fit(TR[cols], np.log1p(TR.y))
                res[f"us:{m}|{k}"] = np.clip(np.expm1(mdl.predict(T[cols])), 0, 100)
                if m == "RF" and k == "C+daily":
                    pi = permutation_importance(mdl, T[cols], np.log1p(T.y), n_repeats=5, random_state=SEED,
                                                scoring="neg_mean_absolute_error")
                    imps.append({"target_year": Y, "daily_group": None, **{c: v for c, v in zip(cols, pi.importances_mean)}})
        for k, cols in (("C", C), ("C+daily", CD)):
            clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=-1,
                                         random_state=SEED).fit(TR[cols], (TR.y > 6).astype(int))
            res[f"p6:RF|{k}"] = clf.predict_proba(T[cols])[:, 1]
        rows.append(res)
        print(f"  {Y}: train {len(TR)} test {len(T)}", flush=True)
    D = pd.concat(rows, ignore_index=True)
    D = D.merge(FB[["unit", "target_year", "ml:RF|C"]].rename(columns={"ml:RF|C": "global:RF|C"}),
                on=["unit", "target_year"], how="left")
    assert D["global:RF|C"].notna().all()

    yc = D.y_class.values
    meth = ["base:median5", "global:RF|C", "us:RF|C", "us:RF|C+daily", "us:Ridge|C", "us:Ridge|C+daily"]
    S = pd.DataFrame([{"Method": m, "n": len(D), "MAE": np.mean(np.abs(D[m] - D.y)),
                       "ClassAcc": np.mean(to_class(D[m]) == yc) * 100} for m in meth])
    e6 = (D.y > 6).astype(float)
    for k in ("C", "C+daily"):
        S = pd.concat([S, pd.DataFrame([{"Method": f"p6:RF|{k}", "n": len(D), "Brier": np.mean((D[f"p6:RF|{k}"] - e6) ** 2)}])])

    rng = np.random.default_rng(SEED)
    uniq, inv = np.unique(D.unit.values, return_inverse=True); U = len(uniq)
    nu = np.bincount(inv, minlength=U).astype(float)
    W = [np.bincount(rng.integers(0, U, U), minlength=U) for _ in range(1000)]
    B = []
    pairs = [("us:RF|C+daily", "us:RF|C", "abs"), ("us:Ridge|C+daily", "us:Ridge|C", "abs"),
             ("us:RF|C", "base:median5", "abs"), ("us:RF|C+daily", "base:median5", "abs"),
             ("global:RF|C", "base:median5", "abs"), ("us:RF|C", "global:RF|C", "abs"),
             ("p6:RF|C+daily", "p6:RF|C", "brier")]
    for a, r, kind in pairs:
        if kind == "abs":
            d = np.abs(D[a] - D.y) - np.abs(D[r] - D.y)
        else:
            d = (D[a] - e6) ** 2 - (D[r] - e6) ** 2
        du = np.bincount(inv, weights=d, minlength=U)
        bs = [(w * du).sum() / (w * nu).sum() for w in W]
        ey = (lambda m: (np.abs(D[m] - D.y)).groupby(D.target_year).mean()) if kind == "abs" else None
        B.append({"Method": a, "Reference": r, "score": "MAE" if kind == "abs" else "Brier",
                  "diff": du.sum() / nu.sum(), "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5),
                  "years_lower": int((ey(a) < ey(r)).sum()) if ey else None, "years": int(D.target_year.nunique())})
    B = pd.DataFrame(B)
    I = pd.DataFrame(imps)
    imp = I[[c for c in I.columns if c not in ("target_year", "daily_group")]].mean().sort_values(ascending=False)

    os.makedirs(OUT, exist_ok=True)
    D.to_csv(os.path.join(OUT, "US_predictions.csv.gz"), index=False)
    S.to_csv(os.path.join(OUT, "US_summary.csv"), index=False)
    B.to_csv(os.path.join(OUT, "US_bootstrap.csv"), index=False)
    imp.rename("perm_importance_MAE_log").to_csv(os.path.join(OUT, "US_importance.csv"))
    json.dump({"rows": len(D), "reactors": int(D.unit.nunique()), "test_years": "2011-2025",
               "train_rows_first_year": int((P.target_year < 2011).sum()), "seconds": round(time.time() - t0)},
              open(os.path.join(OUT, "US_run.json"), "w"), indent=1)
    print(S.round(4).to_string()); print(B.round(4).to_string()); print(imp.head(12).round(5).to_string())


if __name__ == "__main__":
    main()
