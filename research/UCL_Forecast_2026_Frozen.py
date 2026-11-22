"""
UCL_Forecast_2026_Frozen.py
===========================
Prospective test. Freezes forecasts of the 2026 UCL for every reactor with a published 2025 value,
BEFORE PRIS publishes any 2026 value. The file is committed to the repository with its SHA-256 hash and
date, so the forecasts can be scored later against data that nobody had seen.

Methods frozen (all exactly as in the paper, 15-year training window = target years 2011 to 2025):
  point forecasts : median of last 5 usable years; random forest set C; ridge set C; mixed-effects model
  probabilities   : P(UCL 2026 > 6 %) from the calibrated median rule and the random forest classifier, set C
Scoring plan (fixed now): MAE and class accuracy for the point forecasts, Brier score for the probabilities,
95 % intervals from 1000 resamples of whole reactors (seed 42), on the reactors whose 2026 value is usable.
"""
import datetime, hashlib, inspect, json, os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import UCL_Forecast_Panel as FP
from UCL_Forecast_Benchmark import FEATURES, PANEL, SEED, make_model
import UCL_Forecast_MixedEffects as ME

OUT = os.path.join(HERE, "expected_results", "prospective_2026")


def origin_2025_rows():
    """Same features as build_panel, for origin year 2025 (target 2026 unknown)."""
    src = inspect.getsource(FP.build_panel)
    old = "keep = P.y.notna() & (P.n_usable5 >= MIN_USABLE_IN_5) & P.origin_year.between(years.min(), years.max() - 1)"
    assert old in src
    src = src.replace(old, "keep = (P.origin_year == years.max()) & (P.n_usable5 >= MIN_USABLE_IN_5) & P.ucl_l0.notna()")
    src = src.replace('P["y_class"] = ucl_class(P.y)', "")
    ns = dict(vars(FP))
    exec(src, ns)
    raw = FP.load_unit_year()
    return ns["build_panel"](raw)


def main():
    P = pd.read_csv(PANEL)
    TR = P[(P.target_year >= 2011) & (P.target_year <= 2025)]
    T = origin_2025_rows()
    # the unmodified builder must reproduce the stored panel, so the 2025 rows use identical feature code
    ref = FP.build_panel(FP.load_unit_year())
    C = FEATURES["C"]
    assert len(ref) == len(P) and np.allclose(ref[C].values.astype(float), P[C].values.astype(float), equal_nan=True)
    out = T[["unit", "country", "origin_year", "target_year"]].copy()
    out["base:median5"] = T.ucl_med5.values
    for m in ("RF", "Ridge"):
        mdl = make_model(m).fit(TR[C], np.log1p(TR.y))
        out[f"ml:{m}|C"] = np.clip(np.expm1(mdl.predict(T[C])), 0, 100)
    # mixed-effects model
    TRm = ME.prep(TR); Tm = ME.prep(T.assign(y=np.nan))
    r, how = ME.fit_year(TRm)
    fx, full = ME.predict(r, TRm, Tm)
    out["stat:MixedLM"] = np.clip(np.expm1(full), 0, 100)
    # probabilities of a UCL above 6 %
    e = (TR.y > 6).astype(int)
    cal = LogisticRegression(C=1e6, max_iter=2000).fit(np.log1p(TR[["ucl_med5"]].values), e)
    out["p6:calibrated_median"] = cal.predict_proba(np.log1p(T[["ucl_med5"]].values))[:, 1]
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=-1,
                                 random_state=SEED).fit(TR[C], e)
    out["p6:RF|C"] = clf.predict_proba(T[C])[:, 1]

    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, "UCL_2026_frozen_forecasts.csv")
    out.round(4).to_csv(f, index=False)
    h = hashlib.sha256(open(f, "rb").read()).hexdigest()
    meta = {"frozen_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "reactors": int(len(out)), "countries": int(out.country.nunique()), "training_rows": int(len(TR)),
            "mixed_model_fit": how, "sha256": h,
            "scoring_plan": "MAE and class accuracy (point), Brier score (P>6%), 95% intervals from 1000 "
                            "resamples of whole reactors, seed 42, on reactors with a usable 2026 value."}
    json.dump(meta, open(os.path.join(OUT, "UCL_2026_frozen_meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))
    print(out.describe().round(2).to_string())


if __name__ == "__main__":
    main()
