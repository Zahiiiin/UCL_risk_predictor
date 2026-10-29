"""
UCL_Forecast_Extra.py
==============================================================================
EXTRA CHECKS ON THE STORED PART II FORECASTS (no model is refitted here).

WHY
    The main forecast runs (UCL_Forecast_Benchmark.py) store every forecast in FB_predictions.csv.gz.
    The paper needs a few more numbers that can be computed from those stored forecasts alone:

    1. Differences between input sets of one model, with 95% bootstrap intervals (reactors resampled).
    2. The headline comparison with the target year 2025 left out, with the year 2020 left out, and with
       the target years 2018 to 2024 only (the years of the Part I label window).
    3. Bias: the mean signed error (forecast minus true value) of each method.
    4. Where the forecasts gain: errors grouped by the reactor's own recent level (the median of its last five
       usable years, which is known at forecast time), in three groups: below 2, 2 to 6, above 6 percent.

HOW TO RUN (from the repository root)
    python3 research/UCL_Forecast_Extra.py
    Output: research/expected_results/forecast/EXTRA_CHECKS.json

All numbers in the paper that come from this file are read from EXTRA_CHECKS.json.
==============================================================================
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import UCL_Forecast_Benchmark as FB  # noqa: E402

FC = os.path.join(HERE, "expected_results", "forecast")
TAG = "main_window15"
RF_C, XGB_C, RIDGE_C = "ml:RF|C", "ml:XGB|C", "ml:Ridge|C"


def mae(D, c):
    return float((D[c] - D.y).abs().mean())


def boot(D, ref, methods, B=1000):
    b = FB.bootstrap_vs(D, ref, methods, B=B)
    return {r.Method: {"dMAE": float(r.dMAE), "lo": float(r.dMAE_lo), "hi": float(r.dMAE_hi),
                       "dAcc": float(r.dClassAcc), "acc_lo": float(r.dClassAcc_lo), "acc_hi": float(r.dClassAcc_hi)}
            for r in b.itertuples()}


def main():
    D = pd.read_csv(os.path.join(FC, TAG, "FB_predictions.csv.gz"))
    out = {"tag": TAG, "rows": int(len(D)), "note": "computed from stored forecasts of the 15-year window; bootstrap seed 42, 1000 resamples of whole reactors"}

    # 1. input sets of one model
    sets = {}
    for m in ("Ridge", "RF", "XGB"):
        sets[m] = {
            "C_minus_A": boot(D, f"ml:{m}|A", [f"ml:{m}|C"])[f"ml:{m}|C"],
            "D_minus_C": boot(D, f"ml:{m}|C", [f"ml:{m}|D"])[f"ml:{m}|D"],
            "B_minus_A": boot(D, f"ml:{m}|A", [f"ml:{m}|B"])[f"ml:{m}|B"],
        }
    out["set_differences"] = sets

    # 2. headline comparison on other row selections
    sel = {"without_2025": D[D.target_year <= 2024], "without_2020": D[D.target_year != 2020], "2018_to_2024": D[(D.target_year >= 2018) & (D.target_year <= 2024)]}
    rob = {}
    for k, d in sel.items():
        r = {"rows": int(len(d)), "MAE": {c: mae(d, c) for c in ("base:persistence", "base:median5", RF_C, XGB_C, RIDGE_C)}}
        r["RF_C_vs_median5"] = boot(d, "base:median5", [RF_C])[RF_C]
        r["RF_C_vs_persistence"] = boot(d, "base:persistence", [RF_C])[RF_C]
        rob[k] = r
    out["selections"] = rob

    # 3. bias
    cols = [c for c in D.columns if c.startswith("base:") or c.startswith("ml:")]
    out["mean_signed_error"] = {c: float((D[c] - D.y).mean()) for c in cols}
    out["mean_true"] = float(D.y.mean())

    # 4. where the forecasts gain: groups by the reactor's own recent level (median of last five usable years)
    lvl = pd.cut(D["base:median5"], [-1e-9, 2.0, 6.0, 1e9], labels=["below 2", "2 to 6", "above 6"], right=False)
    lvl = np.where(D["base:median5"] < 2, "below 2", np.where(D["base:median5"] <= 6, "2 to 6", "above 6"))
    g = {}
    for k in ("below 2", "2 to 6", "above 6"):
        d = D[lvl == k]
        g[k] = {"rows": int(len(d)), "share_rows": float(len(d) / len(D)), "mean_true": float(d.y.mean()),
                "MAE": {c: mae(d, c) for c in ("base:persistence", "base:median5", RF_C)},
                "bias": {c: float((d[c] - d.y).mean()) for c in ("base:persistence", "base:median5", RF_C)}}
        g[k]["RF_C_vs_median5"] = boot(d, "base:median5", [RF_C])[RF_C]
    out["by_recent_level"] = g
    json.dump(out, open(os.path.join(FC, "EXTRA_CHECKS.json"), "w"), indent=1)
    print("saved", os.path.join(FC, "EXTRA_CHECKS.json"))
    print(json.dumps({k: out[k] for k in ("selections", "by_recent_level")}, indent=1)[:6000])


if __name__ == "__main__":
    main()
