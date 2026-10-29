"""
UCL_Forecast_Tuned.py
==============================================================================
PART II ROBUSTNESS CHECK: the rolling-origin forecast of UCL_Forecast_Benchmark.py, but with the settings of the
random forest and of XGBoost searched for every test year, using only data from before that year.

READ FIRST: UCL_Tuning_Spec.py and TUNING_PRESPECIFICATION.md (committed before this script was run).

WHAT HAPPENS FOR ONE TEST YEAR Y
    1. Training rows = rows whose target year is before Y (and inside the window, if a window is used). Same as untuned.
    2. Inside those rows, the last 3 target years (Y-3, Y-2, Y-1) are put aside as an "inner test".
       Every one of the 8 candidate settings is fitted on the earlier rows only and scored by MAE on the inner test.
       Candidate 0 is the untuned setting. The lowest MAE wins (ties go to the lower number).
    3. The winner is refitted on ALL training rows (target year before Y) and predicts year Y.
    So the labels used for choosing the setting were all known before year Y began. Year Y never takes part.

HOW TO RUN (from the project folder)
    python3 research/UCL_Forecast_Tuned.py --window 15           # primary tuned run
    python3 research/UCL_Forecast_Tuned.py --window 10
    python3 research/UCL_Forecast_Tuned.py --window 0             # all earlier years
==============================================================================
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import UCL_Forecast_Benchmark as F
import UCL_Tuning_Spec as S

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = F.SEED
assert SEED == S.SEED == 42
CAND = {m: S.part2_candidates(m) for m in S.PART2_MODELS}
UNTUNED_TAG = {15: "main_window15", 10: "main_window10", 0: "main_expanding"}


def build(model, params):
    if model == "RF":
        return RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED, **params)
    if model == "XGB":
        return XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=2, tree_method="hist",
                            verbosity=0, **params)
    raise ValueError(model)


def fit_predict(model, params, Xtr, ytr, Xte):
    mdl = build(model, params).fit(Xtr, np.log1p(ytr))
    return np.clip(np.expm1(mdl.predict(Xte)), 0, 100)


def run(P, sets, models, first_test, last_test, window, chosen_rows):
    out = []
    for Y in range(first_test, last_test + 1):
        T = P[P.target_year == Y].copy()
        if T.empty:
            continue
        lo = (Y - window) if window else -10 ** 6
        TR = P[(P.target_year < Y) & (P.target_year >= lo)]
        inner_te = TR[TR.target_year >= Y - S.PART2_INNER_YEARS]
        inner_tr = TR[TR.target_year < Y - S.PART2_INNER_YEARS]
        res = T[["unit", "country", "site", "target_year", "y", "y_class", "in_paper_387"]].copy()
        for name, v in F.baselines(T).items():
            res["base:" + name] = v
        t0 = time.time()
        for sk in sets:
            cols = F.FEATURES[sk]
            for m in models:
                mae = np.array([np.abs(fit_predict(m, p, inner_tr[cols], inner_tr.y, inner_te[cols]) - inner_te.y.values).mean()
                                for p in CAND[m]])
                mae_r = np.round(mae, 9)
                win = int(np.argmin(mae_r))        # first minimum = lowest candidate number
                res[f"ml:{m}-tuned|{sk}"] = fit_predict(m, CAND[m][win], TR[cols], TR.y, T[cols])
                chosen_rows.append({"target_year": Y, "set": sk, "model": m, "candidate": win,
                                    "inner_MAE_selected": float(mae_r[win]), "inner_MAE_default": float(mae_r[0]),
                                    "params": json.dumps(CAND[m][win])})
        out.append(res)
        print(f"  target year {Y}: train rows {len(TR):5d} (inner {len(inner_tr)}+{len(inner_te)}), test rows {len(T):4d}, "
              f"{time.time() - t0:.0f} s", flush=True)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first-test", type=int, default=2010)
    ap.add_argument("--last-test", type=int, default=2025)
    ap.add_argument("--window", type=int, default=15, help="0 = all earlier target years; N = only the latest N")
    ap.add_argument("--sets", nargs="+", default=S.PART2_SETS)
    ap.add_argument("--models", nargs="+", default=S.PART2_MODELS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    P = pd.read_csv(F.PANEL)
    tag = f"window{a.window}" if a.window else "expanding"
    outdir = a.out or os.path.join(HERE, "expected_results", "tuning", "part2", tag)
    os.makedirs(outdir, exist_ok=True)
    print(f"tuned forecast '{tag}': sets {a.sets}, models {a.models}, panel rows {len(P)}", flush=True)
    t0 = time.time()
    chosen = []
    D = run(P, a.sets, a.models, a.first_test, a.last_test, a.window or None, chosen)
    D = D.round(4)                                  # the untuned predictions were stored rounded to 4 decimals; do the same
    ch = pd.DataFrame(chosen)
    ch.to_csv(os.path.join(outdir, "FT_chosen.csv"), index=False)
    D.to_csv(os.path.join(outdir, "FT_predictions.csv.gz"), index=False)
    print(f"fitting done in {time.time() - t0:.0f} s", flush=True)

    # ---- attach the untuned predictions of the same rows (for the direct comparison and the integrity check)
    integrity = {}
    up = os.path.join(HERE, "expected_results", "forecast", UNTUNED_TAG[a.window], "FB_predictions.csv.gz")
    if os.path.exists(up):
        U = pd.read_csv(up)
        keep = ["unit", "target_year"] + [c for c in U.columns if c.startswith("ml:") and c.split("|")[1] in a.sets
                                          and c[3:].split("|")[0] in a.models]
        M = D.merge(U[keep].rename(columns={c: c + "_untuned" for c in keep[2:]}), on=["unit", "target_year"], how="left")
        integrity["rows_tuned"] = int(len(D))
        integrity["rows_matched_to_untuned"] = int(M[keep[2] + "_untuned"].notna().sum()) if len(keep) > 2 else 0
        # baselines must be identical to the untuned file
        Ub = U[["unit", "target_year", "base:persistence", "base:median5"]].rename(
            columns={"base:persistence": "pers_u", "base:median5": "med_u"})
        Mb = D.merge(Ub, on=["unit", "target_year"], how="left")
        integrity["baseline_max_abs_diff_persistence"] = float(np.nanmax(np.abs(Mb["base:persistence"] - Mb.pers_u)))
        integrity["baseline_max_abs_diff_median5"] = float(np.nanmax(np.abs(Mb["base:median5"] - Mb.med_u)))
        # when the default (candidate 0) was chosen, tuned and untuned predictions must be the same
        d0 = ch[ch.candidate == 0]
        n0 = nbad = 0
        worst = 0.0
        for _, r in d0.iterrows():
            col = f"ml:{r.model}|{r.set}"
            sel = M.target_year == r.target_year
            diff = np.abs(M.loc[sel, f"ml:{r.model}-tuned|{r.set}"] - M.loc[sel, col + "_untuned"])
            n0 += 1
            worst = max(worst, float(diff.max()))
            nbad += int(diff.max() > 1e-3)
        integrity.update({"default_chosen_cases": int(n0), "default_cases_that_differ_by_more_than_0.001": int(nbad),
                          "largest_difference_in_default_cases": worst})
        for c in [c for c in M.columns if c.endswith("_untuned")]:
            D[c[:-len("_untuned")] + "@untuned"] = M[c].values
    json.dump(integrity, open(os.path.join(outdir, "FT_integrity.json"), "w"), indent=1)
    print("integrity:", integrity, flush=True)

    # ---- scores (same functions as the untuned run)
    tuned_cols = [c for c in D.columns if "-tuned|" in c]
    untuned_cols = [c for c in D.columns if c.endswith("@untuned")]
    Dscore = D.drop(columns=untuned_cols)
    periods = {"2010-2025": D, "2018-2025": D[D.target_year >= 2018], "2010-2017": D[D.target_year <= 2017]}
    summ, boots = [], []
    for pname, d in periods.items():
        if d.empty:
            continue
        dd = d.drop(columns=untuned_cols)
        s = F.score_table(dd)
        s.insert(0, "Period", pname)
        ref = s.loc[s.Method == "base:persistence", "MAE"].iloc[0]
        s["MAE_skill_vs_persistence_pct"] = (1 - s.MAE / ref) * 100
        summ.append(s)
        for refname in ["base:persistence", "base:median5", "base:mean5", "base:country_mean3"]:
            b = F.bootstrap_vs(dd, refname, [c for c in F.method_columns(dd) if c != refname])
            b.insert(0, "Period", pname)
            boots.append(b)
        # tuned versus the SAME model untuned (direct comparison on the same rows)
        for c in tuned_cols:
            u = c.replace("-tuned", "") + "@untuned"
            if u in d.columns:
                z = dd.copy()
                z[c.replace("-tuned", "")] = d[u].values
                b = F.bootstrap_vs(z, c.replace("-tuned", ""), [c])
                b["Reference"] = c.replace("-tuned", "") + " (untuned)"
                b.insert(0, "Period", pname)
                boots.append(b)
    pd.concat(summ, ignore_index=True).round(4).to_csv(os.path.join(outdir, "FT_summary.csv"), index=False)
    pd.concat(boots, ignore_index=True).round(4).to_csv(os.path.join(outdir, "FT_bootstrap.csv"), index=False)
    by = []
    for Y, d in Dscore.groupby("target_year"):
        s = F.score_table(d)
        s.insert(0, "target_year", Y)
        by.append(s[["target_year", "Method", "n", "MAE", "ClassAcc"]])
    pd.concat(by).round(4).to_csv(os.path.join(outdir, "FT_by_year.csv"), index=False)
    json.dump({"args": vars(a), "rows_scored": int(len(D)), "reactors_scored": int(D.unit.nunique()),
               "seconds": round(time.time() - t0)}, open(os.path.join(outdir, "FT_run.json"), "w"), indent=2)
    s = pd.read_csv(os.path.join(outdir, "FT_summary.csv"))
    print(s[s.Period == "2010-2025"].sort_values("MAE")[["Method", "n", "MAE", "ClassAcc", "AUC_high"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
