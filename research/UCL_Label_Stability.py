"""
UCL_Label_Stability.py
==============================================================================
How stable is the class of a reactor from one part of the label window to another?

WHY THIS CHECK EXISTS
    The label is the mean annual UCL over 2018 to 2024. Annual UCL jumps around (one long
    outage moves a year a lot), so even a perfect model cannot reproduce the class exactly.
    This script measures the noise directly, with no model: it splits the label window into
    an early block (2018 to 2021) and a late block (2022 to 2024), computes the class of each
    block from the same PRIS values, and counts how often the two classes agree.
    The same comparison is made between the 2014 to 2017 history block and the label itself,
    which is what the "persistence" baseline of the benchmark does.

RULES (the same as for the label)
    A year counts if PRIS published a UCL value, the unit was not fully shut down that year,
    and it is not the year of first grid connection (column Used = "yes" in PRIS_YEARLY).
    A reactor needs at least 2 usable years in each block to be compared.

HOW TO RUN (from the research folder)
    python UCL_Label_Stability.py
    Writes results_LabelStability/label_stability.json and label_stability_blocks.csv
==============================================================================
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
XLSX = os.path.join(ROOT, "data", "UCL_Dataset_v6_PRIS.xlsx")
OUT = os.path.join(HERE, "results_LabelStability")
CLASSES = ["Low", "Medium", "High"]


def to_class(v):
    return np.where(v < 2, "Low", np.where(v <= 6, "Medium", "High"))


def block_mean(y, a, b, min_years=2):
    s = y[(y.Year >= a) & (y.Year <= b) & (y.Used == "yes")].groupby("Reactor_Name").UCL.agg(["mean", "count"])
    return s[s["count"] >= min_years]["mean"]


def compare(x, y, label):
    d = pd.concat([x.rename("first"), y.rename("second")], axis=1).dropna()
    c1, c2 = to_class(d["first"].round(2).values), to_class(d["second"].round(2).values)
    idx = {c: i for i, c in enumerate(CLASSES)}
    i1, i2 = np.array([idx[c] for c in c1]), np.array([idx[c] for c in c2])
    cm = np.zeros((3, 3), int)
    for a, b in zip(i1, i2):
        cm[a, b] += 1
    rho = stats.spearmanr(d["first"], d["second"])
    return {
        "comparison": label, "n_reactors": int(len(d)),
        "class_agreement_pct": round(float((i1 == i2).mean() * 100), 1),
        "within_one_class_pct": round(float((np.abs(i1 - i2) <= 1).mean() * 100), 1),
        "low_high_swaps": int((np.abs(i1 - i2) == 2).sum()),
        "weighted_kappa": round(float(cohen_kappa_score(i1, i2, weights="linear")), 3),
        "spearman_rho_of_means": round(float(rho.correlation), 3),
        "confusion_rows_first_cols_second": cm.tolist(),
    }, d


def main():
    os.makedirs(OUT, exist_ok=True)
    yearly = pd.read_excel(XLSX, sheet_name="PRIS_YEARLY")
    final = pd.read_excel(XLSX, sheet_name="FINAL_DATASET").set_index("Reactor_Name")
    early = block_mean(yearly, 2018, 2021)
    late = block_mean(yearly, 2022, 2024)
    h2 = block_mean(yearly, 2014, 2017)
    h1 = block_mean(yearly, 2010, 2013)
    full = final["UCL_Value"]

    res = {}
    r, blocks = compare(early, late, "label window, early block 2018-2021 vs late block 2022-2024")
    res["early_vs_late"] = r
    r, _ = compare(h2, full, "history block 2014-2017 vs label (2018-2024)")
    res["history_2014_2017_vs_label"] = r
    r, _ = compare(h1, full, "history block 2010-2013 vs label (2018-2024)")
    res["history_2010_2013_vs_label"] = r
    r, _ = compare(late, full, "late block 2022-2024 vs label (2018-2024); the blocks overlap, shown for scale only")
    res["late_block_vs_label_overlapping"] = r

    # class of the label that each block reproduces, for reactors in the early/late comparison
    blocks.round(3).to_csv(os.path.join(OUT, "label_stability_blocks.csv"))
    with open(os.path.join(OUT, "label_stability.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    for k, v in res.items():
        print(k, {a: b for a, b in v.items() if a != "confusion_rows_first_cols_second"})
        print("   confusion (rows first, cols second):", v["confusion_rows_first_cols_second"])


if __name__ == "__main__":
    main()
