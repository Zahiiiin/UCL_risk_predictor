"""
UCL_Independent_Check.py
==============================================================================
AN INDEPENDENT RE-COMPUTATION OF THE PART II NUMBERS THAT THE PAPER QUOTES MOST.

WHY
    The forecasting results were produced by UCL_Forecast_Panel.py and UCL_Forecast_Benchmark.py. If those scripts
    had a coding error, the paper would repeat it. This script does NOT import them. It starts from the raw PRIS
    download and the stored forecasts and recomputes, with plain loops and a different code path:

    1. for a random sample of scored rows: the target value (UCL of the target year), the persistence forecast and
       the median-of-last-five forecast, from the raw PRIS table and the three usability rules;
    2. that every scored row satisfies the row rules (target year usable, at least two usable years in the five
       years ending at the origin year, no information after the origin year);
    3. the pooled MAE of persistence, of the median rule and of the random forest (set C), the class accuracies,
       and the number of target years in which the forest has the lower MAE;
    4. the number of rows, reactors and countries;
    5. NO FUTURE DATA IN THE INPUTS: for a random sample of scored rows, EVERY input column of the forecasting table
       (all 27 columns of input sets A to D, plus the target) is recomputed with plain loops from a copy of the raw
       PRIS table that has been cut off at the origin year (all rows of later years are removed before anything is
       computed). The recomputed values must equal the stored ones. A value that had used a later year would differ.

HOW TO RUN (from the repository root)
    python3 research/UCL_Independent_Check.py
    Output: research/expected_results/forecast/INDEPENDENT_CHECK.json  and a PASS/FAIL line for each check.
==============================================================================
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RAW = os.path.join(REPO, "data", "source", "PRIS_annual_values_raw.csv")
XLSX = os.path.join(REPO, "data", "UCL_Dataset_v6_PRIS.xlsx")
FC = os.path.join(HERE, "expected_results", "forecast")
TAG = "main_window15"
N_SAMPLE = 1500
SEED = 7
N_FEATURE_SAMPLE = 600     # rows for the full input-column check (each row rebuilds the fleet and country context)
HIGH_LIMIT = 6.0


def cls(v):
    return 0 if v < 2 else (1 if v <= 6 else 2)


def main():
    raw = pd.read_csv(RAW)
    grid = pd.read_excel(XLSX, sheet_name="FINAL_DATASET").set_index("PRIS_Name")["First_Grid_Year"].to_dict()
    # one dictionary per unit: year -> (UCL, UCF)
    table, first_value_year = {}, {}
    for r in raw.itertuples(index=False):
        if pd.notna(r.UCL):
            table.setdefault(r.Unit, {})[int(r.Year)] = (float(r.UCL), float(r.UCF))
    for u, d in table.items():
        first_value_year[u] = min(d)
    # raw rows without a UCL value are simply absent from `table`

    def usable_value(unit, year):
        d = table.get(unit, {})
        if year not in d:
            return None
        ucl, ucf = d[year]
        if ucf == 0 and ucl == 0:            # complete shutdown
            return None
        g = grid.get(unit, first_value_year[unit])
        if year == g:                        # first year of grid connection
            return None
        return ucl

    D = pd.read_csv(os.path.join(FC, TAG, "FB_predictions.csv.gz"))
    out, ok_all = {}, True

    def check(name, cond, info=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(("PASS  " if cond else "FAIL  ") + name + (f"  [{info}]" if info else ""))
        out[name] = {"pass": bool(cond), "info": info}

    # ---- 1 and 2: sample of rows recomputed from the raw table
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(D), size=min(N_SAMPLE, len(D)), replace=False)
    bad_target = bad_pers = bad_med = bad_rule = 0
    for i in idx:
        r = D.iloc[i]
        unit, Y = r["unit"], int(r["target_year"])
        t = Y - 1
        y_true = usable_value(unit, Y)
        if y_true is None or abs(y_true - r["y"]) > 1e-3:
            bad_target += 1
        recent = [(yr, usable_value(unit, yr)) for yr in range(t - 4, t + 1)]
        recent = [(yr, v) for yr, v in recent if v is not None]
        if len(recent) < 2:
            bad_rule += 1
        earlier = [(yr, usable_value(unit, yr)) for yr in range(1950, t + 1)]
        earlier = [(yr, v) for yr, v in earlier if v is not None]
        if not earlier:
            bad_pers += 1
            continue
        last = earlier[-1][1]
        if abs(last - r["base:persistence"]) > 1e-3:
            bad_pers += 1
        med = float(np.median([v for _, v in recent])) if recent else np.nan
        if recent and abs(med - r["base:median5"]) > 1e-3:
            bad_med += 1
    n = len(idx)
    check("target value equals the raw PRIS UCL of the target year (sampled rows)", bad_target == 0, f"{n} rows, {bad_target} differ")
    check("persistence forecast equals the last usable UCL up to the origin year (sampled rows)", bad_pers == 0, f"{n} rows, {bad_pers} differ")
    check("median-of-five forecast equals the median of the usable values of the five years ending at the origin year (sampled rows)", bad_med == 0, f"{n} rows, {bad_med} differ")
    check("every sampled row has at least two usable years in the five years ending at the origin year", bad_rule == 0, f"{bad_rule} violate")

    # ---- 3: pooled scores recomputed from the stored forecasts
    y = D["y"].values
    res = {}
    for m in ("base:persistence", "base:median5", "ml:RF|C"):
        e = np.abs(D[m].values - y)
        res[m] = {"MAE": float(e.mean()),
                  "class_accuracy": float(np.mean([cls(a) == cls(b) for a, b in zip(D[m].values, y)]) * 100)}
    res["rows"] = int(len(D))
    res["reactors"] = int(D["unit"].nunique())
    res["countries"] = int(D["country"].nunique())
    by = D.assign(e_rf=np.abs(D["ml:RF|C"] - D["y"]), e_md=np.abs(D["base:median5"] - D["y"]), e_pe=np.abs(D["base:persistence"] - D["y"])).groupby("target_year")[["e_rf", "e_md", "e_pe"]].mean()
    res["years_rf_below_median"] = int((by.e_rf < by.e_md).sum())
    res["years_rf_below_persistence"] = int((by.e_rf < by.e_pe).sum())
    res["years"] = int(len(by))
    out["recomputed"] = res
    print(json.dumps(res, indent=1))

    # ---- compare with the stored summary table
    S = pd.read_csv(os.path.join(FC, TAG, "FB_summary.csv"))
    S = S[S.Period == "2010-2025"].set_index("Method")
    for m in ("base:persistence", "base:median5", "ml:RF|C"):
        check(f"MAE of {m} agrees with FB_summary.csv", abs(S.loc[m, "MAE"] - res[m]["MAE"]) < 5e-4, f"{res[m]['MAE']:.4f} vs {S.loc[m, 'MAE']:.4f}")
        check(f"class accuracy of {m} agrees with FB_summary.csv", abs(S.loc[m, "ClassAcc"] - res[m]["class_accuracy"]) < 5e-3, f"{res[m]['class_accuracy']:.3f} vs {S.loc[m, 'ClassAcc']:.3f}")
    ps = json.load(open(os.path.join(FC, "panel_summary.json")))
    check("scored rows equal the number in the run summary", res["rows"] == json.load(open(os.path.join(FC, TAG, "FB_run.json")))["rows_scored"], str(res["rows"]))

    # ---- 5: every input column recomputed from the raw table cut off at the origin year
    PANEL = os.path.join(HERE, "results_Forecast", "forecast_panel.csv")
    P = pd.read_csv(PANEL)
    scored = D[["unit", "target_year"]].merge(P, on=["unit", "target_year"], how="left", validate="one_to_one")
    check("every scored row is a row of the stored forecasting table", scored["origin_year"].notna().all(), f"{len(D)} rows")
    rng5 = np.random.default_rng(SEED + 1)
    sel = rng5.choice(len(scored), size=min(N_FEATURE_SAMPLE, len(scored)), replace=False)
    raw2 = raw.copy()
    raw2["Country"] = raw2["Country"].str.upper()
    raw2["Year"] = raw2["Year"].astype(int)
    full_shut = (raw2.UCF == 0) & (raw2.UCL == 0)
    gy = raw2[raw2.UCL.notna()].groupby("Unit").Year.min()
    raw2["grid"] = raw2["Unit"].map(grid).fillna(raw2["Unit"].map(gy))
    raw2["usable"] = raw2.UCL.notna() & ~full_shut & (raw2.Year != raw2["grid"])
    raw2["shut"] = full_shut
    columns = ["ucl_l0", "ucl_l1", "ucl_l2", "ucl_last", "ucl_gap", "ucl_mean3", "ucl_mean5", "ucl_mean10", "ucl_med5", "ucl_std5",
               "ucl_max5", "n_usable5", "n_usable10", "frac_high5", "delta_3_10", "puf_l0", "puf_mean3", "puf_mean5", "xuf_mean5",
               "n_shut5", "fleet_mean_t", "fleet_med_t", "fleet_mean3", "country_mean3", "country_units_t", "age", "log_mw"]
    by_unit = {u: g.set_index("Year").sort_index() for u, g in raw2.groupby("Unit")}   # sorted: the raw file lists years in descending order
    bad_cells, bad_rows, n_cells = 0, 0, 0
    detail = {}
    for i in sel:
        r = scored.iloc[i]
        unit, t = r["unit"], int(r["origin_year"])
        vis = raw2[raw2.Year <= t]                      # <-- everything after the origin year is removed here
        g = by_unit[unit]
        g = g[g.index <= t]
        U = lambda col, yr: (float(g.loc[yr, col]) if (yr in g.index and bool(g.loc[yr, "usable"]) and pd.notna(g.loc[yr, col])) else None)
        def window(col, w):
            return [v for v in (U(col, yr) for yr in range(t - w + 1, t + 1)) if v is not None]
        mean = lambda v: float(np.mean(v)) if v else np.nan
        hist = [(yr, U("UCL", yr)) for yr in sorted(g.index) if U("UCL", yr) is not None]
        w5, w10, w3 = window("UCL", 5), window("UCL", 10), window("UCL", 3)
        # fleet and country context: usable values of all units, years up to t only
        us = vis[vis.usable]
        fm = us.groupby("Year").UCL.mean()
        yrs3 = [yr for yr in range(t - 2, t + 1) if yr in fm.index]
        c3 = us[(us.Country == r["country"]) & (us.Year >= t - 2)]
        mwv = g["MW"].dropna()
        mw_last = float(mwv.iloc[-1]) if len(mwv) else np.nan
        want = {
            "ucl_l0": U("UCL", t), "ucl_l1": U("UCL", t - 1), "ucl_l2": U("UCL", t - 2),
            "ucl_last": hist[-1][1] if hist else np.nan, "ucl_gap": (t - hist[-1][0]) if hist else np.nan,
            "ucl_mean3": mean(w3), "ucl_mean5": mean(w5), "ucl_mean10": mean(w10),
            "ucl_med5": float(np.median(w5)) if w5 else np.nan,
            "ucl_std5": float(np.std(w5, ddof=1)) if len(w5) >= 3 else np.nan,
            "ucl_max5": max(w5) if w5 else np.nan,
            "n_usable5": float(len(w5)), "n_usable10": float(len(w10)),
            "frac_high5": (sum(v > HIGH_LIMIT for v in w5) / len(w5)) if w5 else np.nan,
            "delta_3_10": (mean(w3) - mean(w10)) if (w3 and w10) else np.nan,
            "puf_l0": U("PUF", t), "puf_mean3": mean(window("PUF", 3)), "puf_mean5": mean(window("PUF", 5)),
            "xuf_mean5": mean(window("XUF", 5)),
            "n_shut5": float(sum(bool(g.loc[yr, "shut"]) for yr in range(t - 4, t + 1) if yr in g.index)),
            "fleet_mean_t": float(fm.loc[t]) if t in fm.index else np.nan,
            "fleet_med_t": float(us[us.Year == t].UCL.median()) if (us.Year == t).any() else np.nan,
            "fleet_mean3": float(np.mean([fm.loc[yr] for yr in yrs3])) if yrs3 else np.nan,
            "country_mean3": float(c3.UCL.sum() / len(c3)) if len(c3) else np.nan,
            "country_units_t": float(vis[(vis.Country == r["country"]) & (vis.Year == t)].Unit.nunique()),
            "age": t - float(g["grid"].iloc[0]),
            "log_mw": float(np.log(mw_last)) if (len(mwv) and mw_last > 0) else np.nan,
        }
        row_bad = 0
        for c in columns:
            a, b = r[c], want[c]
            n_cells += 1
            same = (pd.isna(a) and (b is None or pd.isna(b))) or (pd.notna(a) and b is not None and pd.notna(b) and abs(float(a) - float(b)) <= 1e-6 + 1e-6 * abs(float(b)))
            if not same:
                row_bad += 1
                bad_cells += 1
                detail[c] = detail.get(c, 0) + 1
        bad_rows += int(row_bad > 0)
    check("all 27 input columns of input sets A to D equal values recomputed from the raw table cut off at the origin year (sampled rows)",
          bad_cells == 0, f"{len(sel)} rows, {n_cells} values, {bad_cells} differ" + (f"; by column {detail}" if detail else ""))

    json.dump(out, open(os.path.join(FC, "INDEPENDENT_CHECK.json"), "w"), indent=1)
    print("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
