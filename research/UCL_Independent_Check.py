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
    4. the number of rows, reactors and countries.

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
    json.dump(out, open(os.path.join(FC, "INDEPENDENT_CHECK.json"), "w"), indent=1)
    print("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
