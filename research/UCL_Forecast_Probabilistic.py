"""
UCL_Forecast_Probabilistic.py
=============================
Added analysis A (pre-specified in PROBABILISTIC_AND_MIXED_PRESPEC.md): forecast the PROBABILITY of a bad
next year instead of the value.

Events: E6 = next-year UCL above 6 %, E20 = next-year UCL above 20 %.
Same panel, rows and rolling origin as UCL_Forecast_Benchmark.py (target years 2010 to 2025).

Usage:  python3 research/UCL_Forecast_Probabilistic.py --window 15
        python3 research/UCL_Forecast_Probabilistic.py --window 0      (all earlier years)
"""
import argparse, json, os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from UCL_Forecast_Benchmark import FEATURES, PANEL, SEED          # same inputs and panel as Part II
import UCL_Forecast_Panel as FP

EVENTS = {"E6": 6.0, "E20": 20.0}


def own_counts(limit):
    """For every unit and origin year t: number of usable years in t-4..t and how many exceed the limit."""
    raw = FP.load_unit_year()
    out = []
    for unit, g in raw.groupby("unit"):
        s = g.set_index("year")["UCL_u"].sort_index()
        s = s.reindex(range(int(s.index.min()), int(s.index.max()) + 1))
        n = s.notna().astype(float).rolling(5, min_periods=1).sum()
        k = (s > limit).astype(float).where(s.notna()).rolling(5, min_periods=1).sum()
        out.append(pd.DataFrame({"unit": unit, "origin_year": s.index, f"n5": n.values, f"k5_{int(limit)}": k.values}))
    return pd.concat(out, ignore_index=True)


def make_clf(name):
    if name == "LR":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=SEED)
    if name == "XGB":
        return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=5, random_state=SEED, n_jobs=2, tree_method="hist", verbosity=0,
                             objective="binary:logistic", eval_metric="logloss")
    raise ValueError(name)


def run(P, window, sets=("A", "C"), models=("LR", "RF", "XGB")):
    rows = []
    for Y in range(2010, 2026):
        T = P[P.target_year == Y]
        lo = (Y - window) if window else -10 ** 6
        TR = P[(P.target_year < Y) & (P.target_year >= lo)]
        res = T[["unit", "country", "target_year", "y"]].copy()
        for ev, lim in EVENTS.items():
            ytr = (TR.y > lim).astype(int).values
            p_clim = ytr.mean()
            res[f"{ev}:base:climatology"] = p_clim
            k = T[f"k5_{int(lim)}"].fillna(0).values; n = T["n5"].fillna(0).values
            res[f"{ev}:base:own_frequency"] = (k + p_clim) / (n + 1.0)
            cal = LogisticRegression(C=1e6, max_iter=2000).fit(np.log1p(TR[["ucl_med5"]].values), ytr)
            res[f"{ev}:base:calibrated_median"] = cal.predict_proba(np.log1p(T[["ucl_med5"]].values))[:, 1]
            for sk in sets:
                for m in models:
                    mdl = make_clf(m).fit(TR[FEATURES[sk]], ytr)
                    res[f"{ev}:ml:{m}|{sk}"] = mdl.predict_proba(T[FEATURES[sk]])[:, 1]
        rows.append(res)
        print(f"  target year {Y}: train {len(TR)}, test {len(T)}", flush=True)
    return pd.concat(rows, ignore_index=True)


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        s = idx == b
        if s.any():
            e += s.mean() * abs(y[s].mean() - p[s].mean())
    return e


def scores(D):
    out = []
    for ev, lim in EVENTS.items():
        y = (D.y > lim).astype(int).values
        clim = D[f"{ev}:base:climatology"].values
        bs_clim = np.mean((clim - y) ** 2)
        for c in [c for c in D.columns if c.startswith(ev + ":")]:
            p = np.clip(D[c].values, 1e-6, 1 - 1e-6)
            bs = np.mean((p - y) ** 2)
            top = p >= np.quantile(p, 0.9)
            out.append({"Event": ev, "Method": c.split(":", 1)[1], "n": len(y), "event_rate": y.mean() * 100,
                        "Brier": bs, "BSS_vs_climatology": 1 - bs / bs_clim,
                        "AUC": roc_auc_score(y, p) if c != f"{ev}:base:climatology" else 0.5,
                        "LogLoss": log_loss(y, p), "ECE": ece(y, p),
                        "rate_in_top10pct": y[top].mean() * 100})
    return pd.DataFrame(out)


def bootstrap(D, refs=("base:calibrated_median", "base:own_frequency"), B=1000):
    rng = np.random.default_rng(SEED)
    uniq, inv = np.unique(D.unit.values, return_inverse=True)
    U = len(uniq)
    out = []
    for ev, lim in EVENTS.items():
        y = (D.y > lim).astype(float).values
        cols = [c.split(":", 1)[1] for c in D.columns if c.startswith(ev + ":ml:")]
        for ref in refs:
            r = D[f"{ev}:{ref}"].values
            for m in cols:
                p = D[f"{ev}:{m}"].values
                d_unit = np.bincount(inv, weights=(p - y) ** 2 - (r - y) ** 2, minlength=U)
                n_unit = np.bincount(inv, minlength=U).astype(float)
                bs = []
                for _ in range(B):
                    w = np.bincount(rng.integers(0, U, U), minlength=U)
                    bs.append((w * d_unit).sum() / (w * n_unit).sum())
                out.append({"Event": ev, "Method": m, "Reference": ref, "dBrier": d_unit.sum() / n_unit.sum(),
                            "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5)})
    return pd.DataFrame(out)


def reliability(D, bins=10):
    out = []
    edges = np.linspace(0, 1, bins + 1)
    for ev, lim in EVENTS.items():
        y = (D.y > lim).astype(int).values
        for c in [c for c in D.columns if c.startswith(ev + ":")]:
            p = D[c].values
            idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
            for b in range(bins):
                s = idx == b
                if s.any():
                    out.append({"Event": ev, "Method": c.split(":", 1)[1], "bin": b, "n": int(s.sum()),
                                "mean_p": p[s].mean(), "obs_rate": y[s].mean()})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=15)
    a = ap.parse_args()
    t0 = time.time()
    P = pd.read_csv(PANEL)
    for lim in EVENTS.values():
        oc = own_counts(lim)
        P = P.merge(oc[["unit", "origin_year", f"k5_{int(lim)}"] + (["n5"] if "n5" not in P else [])],
                    on=["unit", "origin_year"], how="left")
    # consistency check with the stored panel: own count of years above 6 % equals frac_high5 * n_usable5
    chk = np.nanmax(np.abs(P["k5_6"] - P["frac_high5"] * P["n_usable5"]))
    assert chk < 1e-6 and np.nanmax(np.abs(P["n5"] - P["n_usable5"])) < 1e-9, chk
    D = run(P, a.window)
    assert len(D) == 6196
    tag = f"window{a.window}" if a.window else "expanding"
    outdir = os.path.join(HERE, "expected_results", "probabilistic", tag)
    os.makedirs(outdir, exist_ok=True)
    D.to_csv(os.path.join(outdir, "PR_predictions.csv.gz"), index=False)
    S = scores(D); S.to_csv(os.path.join(outdir, "PR_summary.csv"), index=False)
    bootstrap(D).to_csv(os.path.join(outdir, "PR_bootstrap.csv"), index=False)
    reliability(D).to_csv(os.path.join(outdir, "PR_reliability.csv"), index=False)
    json.dump({"window": a.window, "rows": len(D), "reactors": int(D.unit.nunique()), "seconds": round(time.time() - t0)},
              open(os.path.join(outdir, "PR_run.json"), "w"), indent=1)
    print(S.round(4).to_string())


if __name__ == "__main__":
    main()
