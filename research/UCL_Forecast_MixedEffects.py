"""
UCL_Forecast_MixedEffects.py
============================
Added analysis B (pre-specified in PROBABILISTIC_AND_MIXED_PRESPEC.md): a standard statistical reliability
model as a benchmark for Part II.

Linear mixed model for log(1 + next-year UCL):
  fixed:  log1p(med5), log1p(last), log1p(mean10), frac_high5, n_usable5, log1p(fleet_mean_t)
  random: intercept per country, intercept per reactor within country (variance component)
Fitted by REML (statsmodels MixedLM) on the training rows of each target year (latest 15 years).
Forecast = fixed part + estimated country and reactor effects (zero for a country or reactor not seen).

The forecasts are compared with the stored Part II forecasts of the same rows
(expected_results/forecast/main_window15/FB_predictions.csv.gz).
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from UCL_Forecast_Benchmark import PANEL, SEED, to_class

WINDOW = 15
FIX = "ly ~ l_med5 + l_last + l_mean10 + frac_high5 + n_usable5 + l_fleet"


def prep(P):
    P = P.copy()
    P["ly"] = np.log1p(P.y)
    P["l_med5"] = np.log1p(P.ucl_med5); P["l_last"] = np.log1p(P.ucl_last)
    P["l_mean10"] = np.log1p(P.ucl_mean10); P["l_fleet"] = np.log1p(P.fleet_mean_t)
    return P


def fit_year(TR):
    md = smf.mixedlm(FIX, TR, groups=TR["country"], re_formula="1", vc_formula={"unit": "0 + C(unit)"})
    for method in ("lbfgs", "powell", "nm"):
        try:
            r = md.fit(reml=True, method=method, maxiter=2000)
            if r.converged:
                return r, method
        except Exception:
            pass
    return None, "failed"


def predict(r, TR, T):
    fe = r.fe_params
    X = np.column_stack([np.ones(len(T))] + [T[c].values for c in fe.index[1:]])
    fixed = X @ fe.values
    re = r.random_effects
    c_eff = np.zeros(len(T)); u_eff = np.zeros(len(T))
    for i, (c, u) in enumerate(zip(T.country.values, T.unit.values)):
        if c in re:
            s = re[c]
            c_eff[i] = s.get("Group", s.iloc[0])
            key = f"unit[C(unit)[{u}]]"
            if key in s.index:
                u_eff[i] = s[key]
    return fixed, fixed + c_eff + u_eff


def main():
    t0 = time.time()
    P = prep(pd.read_csv(PANEL))
    rows, info = [], []
    for Y in range(2010, 2026):
        T = P[P.target_year == Y]
        TR = P[(P.target_year < Y) & (P.target_year >= Y - WINDOW)]
        r, how = fit_year(TR)
        res = T[["unit", "country", "target_year", "y"]].copy()
        if r is None:
            from statsmodels.formula.api import ols
            o = ols(FIX, TR).fit()
            res["stat:MixedLM"] = np.clip(np.expm1(o.predict(T)), 0, 100)
            res["stat:MixedLM_fixed"] = res["stat:MixedLM"]
            info.append({"target_year": Y, "fit": "failed, fixed effects only (OLS)"})
        else:
            fx, full = predict(r, TR, T)
            res["stat:MixedLM"] = np.clip(np.expm1(full), 0, 100)
            res["stat:MixedLM_fixed"] = np.clip(np.expm1(fx), 0, 100)
            vc = r.vcomp[0] if len(r.vcomp) else np.nan
            info.append({"target_year": Y, "fit": how, "n_train": len(TR), "var_country": float(r.cov_re.iloc[0, 0]),
                         "var_reactor": float(vc), "var_resid": float(r.scale),
                         **{f"b_{k}": float(v) for k, v in r.fe_params.items()},
                         **{f"p_{k}": float(v) for k, v in r.pvalues[r.fe_params.index].items()}})
        rows.append(res)
        print(f"  {Y}: {info[-1]['fit']}", flush=True)
    D = pd.concat(rows, ignore_index=True)

    FB = pd.read_csv(os.path.join(HERE, "expected_results", "forecast", "main_window15", "FB_predictions.csv.gz"))
    M = FB.merge(D[["unit", "target_year", "stat:MixedLM", "stat:MixedLM_fixed"]], on=["unit", "target_year"], how="left")
    assert len(M) == 6196 and M["stat:MixedLM"].notna().all()

    methods = ["stat:MixedLM", "stat:MixedLM_fixed", "base:median5", "base:persistence", "ml:Ridge|C", "ml:RF|C", "ml:XGB|C"]
    yc = M.y_class.values
    S = pd.DataFrame([{"Method": m, "n": len(M), "MAE": np.mean(np.abs(M[m] - M.y)),
                       "MedAE": np.median(np.abs(M[m] - M.y)), "ClassAcc": np.mean(to_class(M[m]) == yc) * 100}
                      for m in methods])

    # bootstrap of whole reactors (1000, seed 42): MixedLM minus reference
    rng = np.random.default_rng(SEED)
    uniq, inv = np.unique(M.unit.values, return_inverse=True); U = len(uniq)
    n_u = np.bincount(inv, minlength=U).astype(float)
    W = [np.bincount(rng.integers(0, U, U), minlength=U) for _ in range(1000)]
    B = []
    for m_a, m_b in [("stat:MixedLM", "base:median5"), ("stat:MixedLM", "ml:RF|C"), ("ml:RF|C", "base:median5")]:
        da = np.bincount(inv, weights=np.abs(M[m_a] - M.y) - np.abs(M[m_b] - M.y), minlength=U)
        dc = np.bincount(inv, weights=(to_class(M[m_a]) == yc).astype(float) - (to_class(M[m_b]) == yc), minlength=U)
        bm = [(w * da).sum() / (w * n_u).sum() for w in W]
        bc = [(w * dc).sum() / (w * n_u).sum() * 100 for w in W]
        B.append({"Method": m_a, "Reference": m_b, "dMAE": da.sum() / n_u.sum(),
                  "dMAE_lo": np.percentile(bm, 2.5), "dMAE_hi": np.percentile(bm, 97.5),
                  "dAcc": dc.sum() / n_u.sum() * 100, "dAcc_lo": np.percentile(bc, 2.5), "dAcc_hi": np.percentile(bc, 97.5)})
    B = pd.DataFrame(B)

    out = os.path.join(HERE, "expected_results", "mixed_effects")
    os.makedirs(out, exist_ok=True)
    D.to_csv(os.path.join(out, "ME_predictions.csv.gz"), index=False)
    S.to_csv(os.path.join(out, "ME_summary.csv"), index=False)
    B.to_csv(os.path.join(out, "ME_bootstrap.csv"), index=False)
    pd.DataFrame(info).to_csv(os.path.join(out, "ME_fits_by_year.csv"), index=False)
    json.dump({"window": WINDOW, "rows": len(D), "seconds": round(time.time() - t0)},
              open(os.path.join(out, "ME_run.json"), "w"), indent=1)
    print(S.round(4).to_string()); print(B.round(4).to_string())


if __name__ == "__main__":
    main()
