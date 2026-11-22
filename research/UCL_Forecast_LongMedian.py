"""
UCL_Forecast_LongMedian.py
==============================================================================
POST HOC CHECK: IS THE MEDIAN OF THE LAST FIVE YEARS REALLY THE STRONGEST SIMPLE RULE?

WHY
    The paper compares the learners with a simple rule that forecasts next year's UCL by the median of the last five
    usable years (median5). The learners were given three training windows, several input sets and (in the tuning
    check) eight candidate settings, but the length of the median window was never varied. A fair question is whether a
    median over a longer window is a better simple rule, and how much of the learners' gain remains against it.
    This check was added AFTER the main results were known (after a referee-style review of the manuscript). It is
    not part of the pre-specified design and is reported as a post hoc check.

WHAT IT DOES
    1. For every scored reactor-year of the main run (same 6,196 rows as the 15-year-window run) it computes the median of
       the usable UCL values over the last N calendar years ending at the origin year, for N in WINDOWS and for all
       earlier years. Only years up to the origin year are used. N = 5 must reproduce the stored base:median5 exactly
       (integrity check).
    2. It scores each of these rules (MAE, class accuracy) on the 6,196 rows.
    3. It adds one rule whose window is CHOSEN in the same way as the settings of the tuned learners: for each test
       year Y, the window with the lowest MAE on the three target years before Y (computed from all panel rows of those
       years, never from year Y or later) is used for year Y.
    4. It compares the random forest and XGBoost with input set C (stored forecasts of the three training windows)
       with each rule: difference in MAE and in class accuracy with 95 % intervals from the same bootstrap that
       resamples whole reactors (1000 times, same seed), and the number of test years with a lower MAE.
    5. It repeats the comparison with a bootstrap that resamples the 16 TEST YEARS instead of the reactors (all rows
       of a drawn year are kept; 10,000 draws, seed 42), and reports a paired t statistic on the 16 yearly MAE
       differences. The main analysis resamples reactors, which does not allow for the fact that the errors of all methods
       move together from year to year. The year bootstrap does, but it has only 16 units, so its intervals are wide.
       Output: LM_year_bootstrap.csv.

HOW TO RUN (from the repository root; needs the stored forecasts and the raw PRIS download)
    python3 research/UCL_Forecast_LongMedian.py
    Output: research/expected_results/forecast/long_median/
==============================================================================
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import UCL_Forecast_Panel as FP            # noqa: E402  (load_unit_year: the usable-year rules of the paper)
import UCL_Forecast_Benchmark as FB        # noqa: E402  (bootstrap_vs, to_class: the same scoring as the main run)

FC = os.path.join(HERE, "expected_results", "forecast")
OUT = os.path.join(FC, "long_median")
WINDOWS = [5, 8, 10, 12, 15, 20]       # calendar years ending at the origin year; "all" = every earlier year
FIRST_TEST, LAST_TEST = 2010, 2025
INNER_YEARS = 3                           # same inner test as the tuned learners: the three target years before Y
PERIOD = "2010-2025"


def rule_name(n):
    return "base:median_all" if n == 0 else f"base:median{n}"


def median_table(series_by_unit, units, origins, n):
    """Median of the usable values in the last n calendar years ending at the origin year (n = 0: all earlier years)."""
    out = np.full(len(units), np.nan)
    for i, (u, t) in enumerate(zip(units, origins)):
        s = series_by_unit[u]
        v = s[(s.index <= t) & (s.index > t - n)] if n else s[s.index <= t]
        v = v.dropna()
        if len(v):
            out[i] = float(np.median(v.values))
    return out


def year_bootstrap(M, rules, learners, draws=10000, seed=42):
    """Bootstrap over the test years (all rows of a drawn year are kept) for the learners against each rule."""
    from scipy import stats
    years = np.sort(M.target_year.unique())
    n_y = M.groupby("target_year").size().reindex(years).values.astype(float)
    ae = {c: (M[c] - M.y).abs().groupby(M.target_year).sum().reindex(years).values for c in rules + learners}
    cr = {c: (FB.to_class(M[c].values) == M.y_class.values).astype(float) for c in rules + learners}
    cr = {c: pd.Series(v).groupby(M.target_year.values).sum().reindex(years).values for c, v in cr.items()}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(years), size=(draws, len(years)))
    tot_n = n_y[idx].sum(axis=1)
    rows = []
    for ref in rules:
        for lrn in learners:
            point_mae = ae[lrn].sum() / n_y.sum() - ae[ref].sum() / n_y.sum()
            point_acc = (cr[lrn].sum() - cr[ref].sum()) / n_y.sum() * 100
            bm = (ae[lrn][idx].sum(axis=1) - ae[ref][idx].sum(axis=1)) / tot_n
            ba = (cr[lrn][idx].sum(axis=1) - cr[ref][idx].sum(axis=1)) / tot_n * 100
            dy = ae[lrn] / n_y - ae[ref] / n_y                          # yearly MAE difference
            t, pv = stats.ttest_1samp(dy, 0.0)
            rows.append({"Reference": ref, "Method": lrn, "dMAE": point_mae, "dMAE_lo": np.percentile(bm, 2.5), "dMAE_hi": np.percentile(bm, 97.5),
                         "dClassAcc": point_acc, "dClassAcc_lo": np.percentile(ba, 2.5), "dClassAcc_hi": np.percentile(ba, 97.5),
                         "years_t": float(t), "years_p": float(pv), "Period": PERIOD})
    pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, "LM_year_bootstrap.csv"), index=False)


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = FP.load_unit_year()
    series = {u: g.set_index("year")["UCL_u"] for u, g in raw.groupby("unit")}
    P = pd.read_csv(FP.OUT_DIR + "/forecast_panel.csv") if hasattr(FP, "OUT_DIR") else None
    P = P[["unit", "country", "origin_year", "target_year", "y"]].copy()
    P["y_class"] = FB.to_class(P.y.values)

    cand = WINDOWS + [0]
    for n in cand:
        P[rule_name(n)] = median_table(series, P.unit.values, P.origin_year.values, n)

    # ---- window chosen on the three target years before Y (never on Y or later)
    ae = {n: np.abs(P[rule_name(n)].values - P.y.values) for n in cand}
    chosen, sel_rows = {}, []
    for Y in range(FIRST_TEST, LAST_TEST + 1):
        inner = ((P.target_year >= Y - INNER_YEARS) & (P.target_year < Y)).values
        score = {n: float(np.nanmean(ae[n][inner])) for n in cand}
        best = min(cand, key=lambda n: (score[n], n))          # ties: the shorter window
        chosen[Y] = best
        sel_rows.append({"target_year": Y, "chosen_window": "all" if best == 0 else best, **{f"inner_MAE_{'all' if n == 0 else n}": round(score[n], 4) for n in cand}})
    pd.DataFrame(sel_rows).to_csv(os.path.join(OUT, "LM_selected_window.csv"), index=False)
    P["base:median_selected"] = [P.loc[i, rule_name(chosen[Y])] if Y in chosen else np.nan for i, Y in zip(P.index, P.target_year)]

    # ---- score on the rows of the main run
    stored = {t: pd.read_csv(os.path.join(FC, t, "FB_predictions.csv.gz")) for t in ("main_window15", "main_window10", "main_expanding")}
    D0 = stored["main_window15"]
    key = ["unit", "target_year"]
    P5 = P[key + [rule_name(5)]].rename(columns={rule_name(5): "chk_median5"})
    others = [rule_name(n) for n in cand if n != 5]
    M = D0.merge(P[key + others + ["base:median_selected"]], on=key, how="left", validate="one_to_one")
    M = M.merge(P5, on=key, how="left", validate="one_to_one")
    assert len(M) == len(D0) == 6196 and M[others + ["chk_median5"]].notna().all().all()
    # integrity: the median over five calendar years computed here must equal the stored median5 forecast
    diff5 = float(np.abs(M["base:median5"].values - M["chk_median5"].values).max())
    assert diff5 < 1e-4, diff5          # the stored forecasts are rounded to four decimals
    M = M.drop(columns=["chk_median5"])
    for t, name in (("main_window15", "15"), ("main_window10", "10"), ("main_expanding", "all")):
        s = stored[t].set_index(key)
        for lrn in ("RF", "XGB"):
            M[f"ml:{lrn}|C ({name} years)"] = s.loc[list(zip(M.unit, M.target_year)), f"ml:{lrn}|C"].values
    # tuned learners of the tuning check (settings searched on earlier years only), set C
    for t, name in (("window15", "15"), ("window10", "10"), ("expanding", "all")):
        s = pd.read_csv(os.path.join(HERE, "expected_results", "tuning", "part2", t, "FT_predictions.csv.gz")).set_index(key)
        for lrn in ("RF", "XGB"):
            M[f"ml:{lrn}-tuned|C ({name} years)"] = s.loc[list(zip(M.unit, M.target_year)), f"ml:{lrn}-tuned|C"].values
    M["y_class"] = FB.to_class(M.y.values)

    rules = [rule_name(n) for n in cand] + ["base:median_selected"]
    learners = [c for c in M.columns if c.startswith("ml:")]
    S = FB.score_table(M[["unit", "country", "site", "target_year", "y", "y_class", "in_paper_387"] + rules + learners])
    S["Period"] = PERIOD
    S.round(4).to_csv(os.path.join(OUT, "LM_summary.csv"), index=False)

    boots, byyear = [], []
    for ref in rules:
        b = FB.bootstrap_vs(M, ref, learners)
        b["Period"] = PERIOD
        boots.append(b)
        for lrn in learners:
            e_l = M.assign(e=np.abs(M[lrn] - M.y)).groupby("target_year").e.mean()
            e_r = M.assign(e=np.abs(M[ref] - M.y)).groupby("target_year").e.mean()
            byyear.append({"Reference": ref, "Method": lrn, "years_lower": int((e_l < e_r).sum()), "years": int(len(e_l))})
    pd.concat(boots, ignore_index=True).round(4).to_csv(os.path.join(OUT, "LM_bootstrap.csv"), index=False)
    pd.DataFrame(byyear).to_csv(os.path.join(OUT, "LM_wins_by_year.csv"), index=False)
    year_bootstrap(M, rules, learners)
    json.dump({"windows": WINDOWS + ["all"], "rows": int(len(M)), "reactors": int(M.unit.nunique()),
               "median5_max_abs_difference_from_stored": diff5,
               "note": "post hoc check added after the main results; the window of the selected rule is chosen on the three target years before each test year"},
              open(os.path.join(OUT, "LM_run.json"), "w"), indent=1)
    show = S[S.Method.isin(rules + ["ml:RF|C (15 years)", "ml:XGB|C (15 years)"])][["Method", "MAE", "ClassAcc"]]
    print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
