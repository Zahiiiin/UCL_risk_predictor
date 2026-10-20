"""
UCL_Forecast_Panel.py
=====================
OPTION B, STEP 1: build the reactor-year forecasting table (the "panel").

WHAT THIS SCRIPT DOES (in plain words)
--------------------------------------
The paper classifies one number per reactor (its 2018-2024 average).  Here we ask a different question:

    "At the end of year t, how much unplanned capability loss (UCL) will this reactor have in year t+1?"

So the table has ONE ROW PER REACTOR AND ORIGIN YEAR t.
  * The columns ("features") describe what was known at the end of year t.
  * The answer ("target", column y) is the UCL of the NEXT year, t+1.

THE ONE RULE THAT PROTECTS THE RESULTS
--------------------------------------
A feature in the row for origin year t may only use data from years <= t.  Nothing from t+1 or later.
If a feature saw the future, the model would look clever but would fail in real use ("leakage").
The function check_no_leakage() at the bottom re-computes a sample of features from scratch
using only the allowed years and stops with an error if even one value differs.

WHICH YEARS COUNT (same rules as the paper's label, see tools/build_dataset_v6.py)
----------------------------------------------------------------------------------
A reactor-year is USABLE if
  1. PRIS published a UCL value for it,
  2. the unit was not completely shut down for planned reasons (UCF = 0 and UCL = 0), and
  3. it is not the year of first grid connection (a partial year).
For the 387 reactors of the paper we use their First_Grid_Year.  For the other units in the raw file
(mostly reactors that were shut down before 2025) the design table is not available, so the first year
with a published value is used instead.  This is an approximation and is stated in the paper.

WHO IS IN THE TABLE
-------------------
All 649 units in the raw PRIS file, from their first usable year until their last year in the file.
Reactors that were permanently shut down are kept for the years in which they operated.  This is on purpose:
if we kept only today's operating reactors, the table would hide the reactors that failed and the results would
look too good ("survivorship bias").  The 387 paper reactors are flagged (column in_paper_387).

Run:   python3 research/UCL_Forecast_Panel.py
Output: research/results_Forecast/forecast_panel.csv     (the table, about 17,000 rows)
        research/expected_results/forecast/panel_summary.json  (small summary that is committed to git)
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RAW = os.path.join(REPO, "data", "source", "PRIS_annual_values_raw.csv")
XLSX = os.path.join(REPO, "data", "UCL_Dataset_v6_PRIS.xlsx")
OUT_DIR = os.path.join(HERE, "results_Forecast")
SUM_DIR = os.path.join(HERE, "expected_results", "forecast")

LOW_LIMIT, HIGH_LIMIT = 2.0, 6.0     # same class limits as the paper (percent)
MIN_USABLE_IN_5 = 2                  # a row needs at least 2 usable years in the last 5 (origin year included)


def ucl_class(v):
    """0 = Low (<2), 1 = Medium (2 to 6), 2 = High (>6).  Works on numbers and arrays."""
    v = np.asarray(v, dtype=float)
    return np.where(v < LOW_LIMIT, 0, np.where(v <= HIGH_LIMIT, 1, 2))


def site_name(unit, country):
    """Heuristic 'site' = unit name without its unit number or letter, e.g. BRUCE-3 -> BRUCE.
    Only used for the grouped robustness test.  It is NOT an official IAEA site code."""
    s = re.sub(r"\s*\(.*?\)", "", unit)                 # drop "(Taiwan, China)" and similar
    s = re.sub(r"[-\s]+[A-Z]?\d*[A-Z]?$", "", s).strip(" -")
    return f"{country}|{s}" if s else f"{country}|{unit}"


def load_unit_year():
    """Raw PRIS table -> one row per unit and year, with the 'usable' flag and clean copies of the indicators."""
    raw = pd.read_csv(RAW).rename(columns={"Unit": "unit", "Country": "country", "Year": "year"})
    raw["country"] = raw["country"].str.upper()
    final = pd.read_excel(XLSX, sheet_name="FINAL_DATASET")
    grid = final.set_index("PRIS_Name")["First_Grid_Year"]
    first_rep = raw[raw.UCL.notna()].groupby("unit").year.min()
    raw["grid_year"] = raw["unit"].map(grid)
    raw["grid_year"] = raw["grid_year"].fillna(raw["unit"].map(first_rep))
    raw["in_paper_387"] = raw["unit"].isin(set(grid.index)).astype(int)
    raw["no_value"] = raw.UCL.isna()
    raw["fully_shut"] = (raw.UCF == 0) & (raw.UCL == 0)
    raw["usable"] = ~raw.no_value & ~raw.fully_shut & (raw.year != raw.grid_year)
    for c in ["UCL", "PUF", "XUF"]:
        raw[c + "_u"] = np.where(raw.usable, raw[c], np.nan)      # NaN when the year is not usable
    raw["shut_flag"] = raw.fully_shut.astype(float)
    return raw


def build_panel(raw):
    """Create the rows.  Every feature at origin year t is computed from years <= t only."""
    years = np.arange(raw.year.min(), raw.year.max() + 1)
    # fleet context per year: average over ALL units with a usable value in that year
    fleet_mean = raw[raw.usable].groupby("year").UCL.mean().reindex(years)
    fleet_med = raw[raw.usable].groupby("year").UCL.median().reindex(years)
    fleet_mean3 = fleet_mean.rolling(3, min_periods=1).mean()
    # country context per year: average over all units of the same country (usable values)
    cy = raw[raw.usable].groupby(["country", "year"]).UCL.agg(["sum", "count"])
    cy = cy.reindex(pd.MultiIndex.from_product([sorted(raw.country.unique()), years], names=["country", "year"]),
                    fill_value=0)
    cy3 = cy.groupby(level=0).rolling(3, min_periods=1).sum().droplevel(0)       # sums over t-2..t
    country_mean3 = (cy3["sum"] / cy3["count"].replace(0, np.nan))
    country_units = raw[raw.year.isin(years)].groupby(["country", "year"]).unit.nunique()

    rows = []
    for unit, g in raw.groupby("unit", sort=True):
        g = g.set_index("year").reindex(years)
        country = raw.loc[raw.unit == unit, "country"].iloc[0]
        grid_year = float(raw.loc[raw.unit == unit, "grid_year"].iloc[0])
        in387 = int(raw.loc[raw.unit == unit, "in_paper_387"].iloc[0])
        u, puf, xuf = g["UCL_u"], g["PUF_u"], g["XUF_u"]
        shut = g["shut_flag"].fillna(0.0)
        mw = g["MW"].ffill()

        def roll(s, w, how, minp=1):
            return getattr(s.rolling(w, min_periods=minp), how)()

        n5 = u.notna().astype(float).rolling(5, min_periods=1).sum()
        n10 = u.notna().astype(float).rolling(10, min_periods=1).sum()
        high5 = (u > HIGH_LIMIT).astype(float).where(u.notna()).rolling(5, min_periods=1).sum()
        last_year_with_value = pd.Series(np.where(u.notna(), years, np.nan), index=years).ffill()
        f = pd.DataFrame({
            "unit": unit, "country": country, "origin_year": years, "target_year": years + 1,
            "in_paper_387": in387,
            # --- own history of UCL (usable years only) -------------------------------------------
            "ucl_l0": u.values, "ucl_l1": u.shift(1).values, "ucl_l2": u.shift(2).values,
            "ucl_last": u.ffill().values,
            "ucl_gap": (years - last_year_with_value).values,          # years since that last usable value
            "ucl_mean3": roll(u, 3, "mean").values, "ucl_mean5": roll(u, 5, "mean").values,
            "ucl_mean10": roll(u, 10, "mean").values, "ucl_med5": roll(u, 5, "median").values,
            "ucl_std5": roll(u, 5, "std", 3).values, "ucl_max5": roll(u, 5, "max").values,
            "n_usable5": n5.values, "n_usable10": n10.values,
            "frac_high5": (high5 / n5.replace(0, np.nan)).values,       # share of the last 5 usable years above 6 %
            "delta_3_10": (roll(u, 3, "mean") - roll(u, 10, "mean")).values,   # recent minus long-run level
            # --- own planned and external losses ---------------------------------------------------
            "puf_l0": puf.values, "puf_mean3": roll(puf, 3, "mean").values, "puf_mean5": roll(puf, 5, "mean").values,
            "xuf_mean5": roll(xuf, 5, "mean").values,
            "n_shut5": shut.rolling(5, min_periods=1).sum().values,     # years of complete planned shutdown in last 5
            # --- static facts known at t ------------------------------------------------------------
            "age": (years - grid_year),
            "log_mw": np.log(mw.where(mw > 0)).values,
            # --- context known at t -----------------------------------------------------------------
            "fleet_mean_t": fleet_mean.values, "fleet_med_t": fleet_med.values, "fleet_mean3": fleet_mean3.values,
            # --- the answer (next year) -------------------------------------------------------------
            "y": u.shift(-1).values,
        })
        rows.append(f)
    P = pd.concat(rows, ignore_index=True)
    P["country_mean3"] = [country_mean3.get((c, y), np.nan) for c, y in zip(P.country, P.origin_year)]
    P["country_units_t"] = [country_units.get((c, y), np.nan) for c, y in zip(P.country, P.origin_year)]
    P["site"] = [site_name(u, c) for u, c in zip(P.unit, P.country)]
    # keep only rows that can be used: next year is usable AND the unit has some recent history
    keep = P.y.notna() & (P.n_usable5 >= MIN_USABLE_IN_5) & P.origin_year.between(years.min(), years.max() - 1)
    P = P[keep].reset_index(drop=True)
    P["y_class"] = ucl_class(P.y)
    return P


def check_no_leakage(raw, P, n_check=400, seed=42):
    """Recompute some features from scratch using ONLY years <= origin_year, and compare with the table."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(P), size=min(n_check, len(P)), replace=False)
    bad = 0
    for i in idx:
        r = P.iloc[i]
        g = raw[(raw.unit == r.unit) & (raw.year <= r.origin_year)].sort_values("year")
        u = g.set_index("year")["UCL_u"]
        last5 = u[u.index > r.origin_year - 5].dropna()
        last3 = u[u.index > r.origin_year - 3].dropna()
        tests = [
            (r.ucl_mean5, last5.mean() if len(last5) else np.nan),
            (r.ucl_mean3, last3.mean() if len(last3) else np.nan),
            (r.ucl_last, u.dropna().iloc[-1] if u.notna().any() else np.nan),
            (r.n_usable5, float(len(last5))),
        ]
        for got, want in tests:
            if not (np.isnan(got) and np.isnan(want)) and not np.isclose(got, want, atol=1e-9):
                bad += 1
        # the target must be the UCL of year origin_year + 1 and nothing else
        t = raw[(raw.unit == r.unit) & (raw.year == r.origin_year + 1)]
        want_y = t.UCL_u.iloc[0] if len(t) else np.nan
        if not np.isclose(r.y, want_y, atol=1e-9):
            bad += 1
    return bad, len(idx)


def main():
    raw = load_unit_year()
    P = build_panel(raw)
    bad, n = check_no_leakage(raw, P)
    print(f"leakage check: {n} random rows recomputed from scratch, {bad} mismatches")
    if bad:
        sys.exit("STOP: the leakage check failed")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(SUM_DIR, exist_ok=True)
    P.to_csv(os.path.join(OUT_DIR, "forecast_panel.csv"), index=False)
    ty = P.groupby("target_year").agg(rows=("y", "size"), reactors=("unit", "nunique"),
                                       mean_y=("y", "mean"), median_y=("y", "median"),
                                       share_high=("y_class", lambda s: float((s == 2).mean())))
    summary = {
        "rows": int(len(P)), "units": int(P.unit.nunique()), "countries": int(P.country.nunique()),
        "sites_heuristic": int(P.site.nunique()),
        "rows_in_paper_387": int(P.in_paper_387.sum()), "units_in_paper_387": int(P[P.in_paper_387 == 1].unit.nunique()),
        "target_years": [int(P.target_year.min()), int(P.target_year.max())],
        "leakage_check": {"rows_checked": int(n), "mismatches": int(bad)},
        "class_share_all_rows": {k: float((P.y_class == v).mean()) for k, v in [("Low", 0), ("Medium", 1), ("High", 2)]},
        "per_target_year": {int(k): {c: (float(v) if c != "rows" and c != "reactors" else int(v)) for c, v in r.items()}
                            for k, r in ty.loc[2005:].iterrows()},
    }
    json.dump(summary, open(os.path.join(SUM_DIR, "panel_summary.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_target_year"}, indent=2))


if __name__ == "__main__":
    main()
