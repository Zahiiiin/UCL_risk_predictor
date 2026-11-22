"""
UCL_Forecast_Report.py
======================
OPTION B, STEP 3: turn the stored forecasting results into one readable file.

It only READS the files that UCL_Forecast_Benchmark.py wrote (nothing is re-fitted) and writes
research/expected_results/forecast/FORECAST_RESULTS.md.  Every number in that file is copied from a result file,
none is typed by hand.

Run:  python3 research/UCL_Forecast_Report.py
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "expected_results", "forecast")
PERIOD = "2010-2025"
NAMES = {"base:persistence": "Persistence (last usable year)", "base:mean5": "Mean of last 5 years",
         "base:median5": "Median of last 5 years", "base:country_mean3": "Country average (3 years)",
         "base:fleet_mean": "World average this year"}
RUNS = [("main_expanding", "All earlier years"), ("main_window15", "Latest 15 years"), ("main_window10", "Latest 10 years")]


def nm(m):
    if m in NAMES:
        return NAMES[m]
    model, s = m[3:].split("|")
    return f"{model}, feature set {s}"


def load(tag, f):
    return pd.read_csv(os.path.join(R, tag, f))


def md(df, floatfmt=None):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def f1(x): return f"{x:.2f}"
def f0(x): return f"{x:.1f}"
def ci(d, lo, hi): return f"{d:+.2f} ({lo:+.2f} to {hi:+.2f})"
def cia(d, lo, hi): return f"{d:+.1f} ({lo:+.1f} to {hi:+.1f})"


def main():
    L = []
    ps = json.load(open(os.path.join(R, "panel_summary.json")))
    L.append("# Option B results: forecasting next-year UCL of each reactor\n")
    L.append("All numbers are read from the result files in this folder by `research/UCL_Forecast_Report.py`. "
             "Scores are pooled over target years 2010 to 2025 unless a table says otherwise. MAE is the mean absolute "
             "error in percentage points of UCL (smaller is better). Class accuracy is the share of reactor-years whose "
             "predicted Low/Medium/High class (limits 2 and 6 percent) equals the true class of that year.\n")
    L.append(f"**Panel.** {ps['rows']:,} reactor-year rows, {ps['units']} reactors, {ps['countries']} countries, "
             f"target years {ps['target_years'][0]} to {ps['target_years'][1]}. Leakage check: {ps['leakage_check']['rows_checked']} "
             f"random rows recomputed from scratch, {ps['leakage_check']['mismatches']} mismatches.\n")

    # ---------------------------------------------------------------- table 1 main
    methods = ["base:persistence", "base:mean5", "base:median5", "base:country_mean3",
               "ml:Ridge|A", "ml:Ridge|C", "ml:RF|A", "ml:RF|C", "ml:XGB|A", "ml:XGB|C"]
    rows = []
    S = {t: load(t, "FB_summary.csv") for t, _ in RUNS}
    for m in methods:
        r = {"Method": nm(m)}
        for t, lab in RUNS:
            s = S[t][(S[t].Period == PERIOD) & (S[t].Method == m)].iloc[0]
            r[f"MAE, {lab.lower()}"] = f1(s.MAE)
        s = S["main_window15"]
        s = s[(s.Period == PERIOD) & (s.Method == m)].iloc[0]
        r["Class acc. % (15 yr)"] = f0(s.ClassAcc)
        r["AUC high (15 yr)"] = f"{s.AUC_high:.3f}"
        rows.append(r)
    L.append("## Table 1. Rolling-origin forecasts, three training windows\n")
    L.append("Feature sets: A = own UCL history; C = A + own planned/external losses + world and country context. "
             "Window = how many past target years the model may learn from. None of the windows was tuned on the test years.\n")
    L.append(md(pd.DataFrame(rows)) + "\n")

    # ---------------------------------------------------------------- table 2 differences
    rows = []
    for t, lab in RUNS:
        B = load(t, "FB_bootstrap.csv")
        for ref, rl in [("base:median5", "median of last 5"), ("base:persistence", "persistence")]:
            for m in ["ml:Ridge|C", "ml:RF|C", "ml:XGB|C"]:
                b = B[(B.Period == PERIOD) & (B.Reference == ref) & (B.Method == m)].iloc[0]
                rows.append({"Window": lab, "Model": nm(m), "Compared with": rl,
                             "Change in MAE (95% CI)": ci(b.dMAE, b.dMAE_lo, b.dMAE_hi),
                             "Change in class acc., points (95% CI)": cia(b.dClassAcc, b.dClassAcc_lo, b.dClassAcc_hi)})
    L.append("## Table 2. Differences with 95% intervals (reactors resampled 1000 times)\n")
    L.append("Negative change in MAE = better than the comparison. Rows of one reactor are not independent, so whole reactors are resampled. "
             "No correction for the number of comparisons was applied.\n")
    L.append(md(pd.DataFrame(rows)) + "\n")

    # ---------------------------------------------------------------- table 3 ablation
    rows = []
    s = S["main_window15"]
    s = s[s.Period == PERIOD].set_index("Method")
    for model in ["Ridge", "RF", "XGB"]:
        r = {"Model": model}
        for k in "ABCD":
            r[f"Set {k}"] = f1(s.loc[f"ml:{model}|{k}", "MAE"])
        rows.append(r)
    L.append("## Table 3. What the models look at (MAE, 15-year window)\n")
    L.append("A = own UCL history; B = A + own planned/external losses; C = B + world and country context; D = C + age and size.\n")
    L.append(md(pd.DataFrame(rows)) + "\n")

    # ---------------------------------------------------------------- table 4 periods
    rows = []
    for per in ["2010-2017", "2018-2025"]:
        for m in ["base:persistence", "base:median5", "ml:RF|C"]:
            x = S["main_window15"]
            x = x[(x.Period == per) & (x.Method == m)].iloc[0]
            rows.append({"Target years": per, "Method": nm(m), "MAE": f1(x.MAE), "Class acc. %": f0(x.ClassAcc)})
    L.append("## Table 4. Early and late half of the test period (15-year window)\n")
    L.append(md(pd.DataFrame(rows)) + "\n")

    # ---------------------------------------------------------------- table 5 grouped
    rows = []
    for tag, lab in [("main_window15", "Same countries and sites allowed in training"),
                     ("grouped-site_window15", "New site (no unit of the site in training)"),
                     ("grouped-country_window15", "New country (no unit of the country in training)")]:
        x = load(tag, "FB_summary.csv")
        x = x[x.Period == PERIOD].set_index("Method")
        r = {"Test": lab}
        for m in ["ml:RF|A", "ml:RF|C", "ml:XGB|A", "ml:XGB|C", "ml:Ridge|A", "ml:Ridge|C"]:
            r[nm(m).replace("feature set ", "set ")] = f1(x.loc[m, "MAE"])
        rows.append(r)
    L.append("## Table 5. Do the models work for places they have not seen? (MAE, 15-year window)\n")
    L.append("Reference rules on the same rows: persistence "
             f"{f1(s.loc['base:persistence', 'MAE'])}, median of last 5 {f1(s.loc['base:median5', 'MAE'])}. "
             "Sites are guessed from unit names (heuristic, not an IAEA code). Six folds; every training row of the held-out group is removed.\n")
    L.append(md(pd.DataFrame(rows)) + "\n")
    B = load("grouped-country_window15", "FB_bootstrap.csv")
    rows = []
    for m in ["ml:RF|A", "ml:XGB|A", "ml:Ridge|A", "ml:RF|C", "ml:Ridge|C"]:
        b = B[(B.Period == PERIOD) & (B.Reference == "base:median5") & (B.Method == m)].iloc[0]
        rows.append({"New-country model": nm(m), "Change in MAE vs median of last 5 (95% CI)": ci(b.dMAE, b.dMAE_lo, b.dMAE_hi),
                     "Change in class acc., points (95% CI)": cia(b.dClassAcc, b.dClassAcc_lo, b.dClassAcc_hi)})
    L.append(md(pd.DataFrame(rows)) + "\n")

    # ---------------------------------------------------------------- table 6 per-year wins
    rows = []
    for t, lab in RUNS:
        y = load(t, "FB_by_year.csv")
        piv = y.pivot(index="target_year", columns="Method", values="MAE")
        for m in ["ml:RF|C", "ml:XGB|C", "ml:Ridge|C"]:
            rows.append({"Window": lab, "Model": nm(m), "Years better than median of last 5": f"{int((piv[m] < piv['base:median5']).sum())} of {len(piv)}",
                         "Years better than persistence": f"{int((piv[m] < piv['base:persistence']).sum())} of {len(piv)}"})
    L.append("## Table 6. In how many of the 16 test years does the model have the lower MAE?\n")
    L.append(md(pd.DataFrame(rows)) + "\n")

    # ---------------------------------------------------------------- table 7 design
    x = load("subset387_window15", "FB_summary.csv")
    x = x[x.Period == PERIOD].set_index("Method")
    rows = []
    for model in ["Ridge", "RF", "XGB"]:
        rows.append({"Model": model, "MAE without design facts (set D)": f1(x.loc[f"ml:{model}|D", "MAE"]),
                     "MAE with reactor type and construction time (set E)": f1(x.loc[f"ml:{model}|E", "MAE"])})
    L.append("## Table 7. Do design facts help? (the 387 reactors of the paper, 15-year window)\n")
    L.append(f"Reference rules on these rows: persistence {f1(x.loc['base:persistence', 'MAE'])}, "
             f"median of last 5 {f1(x.loc['base:median5', 'MAE'])}.\n")
    L.append(md(pd.DataFrame(rows)) + "\n")

    open(os.path.join(R, "FORECAST_RESULTS.md"), "w").write("\n".join(L))
    print("wrote", os.path.join(R, "FORECAST_RESULTS.md"))


if __name__ == "__main__":
    main()
