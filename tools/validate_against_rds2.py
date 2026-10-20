"""
validate_against_rds2.py
==============================================================================
Checks the PRIS annual values (the source of the label) against the figures that the
IAEA prints in the RDS-2 books, and writes the numbers used in the paper.

WHAT IT CHECKS
    1. UCF 2013-2017       RDS-2 2018, Table 14 (5-year average)   vs the mean of the PRIS annual UCF
    2. UCF "2010-2020"     RDS-2 2021, Table 14                    vs PRIS 2010-2020 AND vs PRIS lifetime to 2020
    3. UCF "2014-2024"     RDS-2 2025, Table 14                    vs PRIS 2014-2024 AND vs PRIS lifetime to 2024
    4. PCL 2021-2023       RDS-2 2024, Table 18 (by type and size) vs the mean PRIS planned unavailability
                           (PUF) of the dataset reactors in the same group

    The book values come from the version 5 workbook (sheets UCL_LONGITUDAL and PCL_REFERENCE),
    which was typed from the printed tables and corrected against them. The PRIS values come from
    data/source/PRIS_annual_values_raw.csv.

HOW TO RUN (from the repository root)
    python tools/validate_against_rds2.py
    Writes research/expected_results/validation/ (a per-reactor table and a summary file).
==============================================================================
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "research", "expected_results", "validation")
V5 = os.path.join(ROOT, "data", "UCL_Dataset_v5_longitudinal.xlsx")
V6 = os.path.join(ROOT, "data", "UCL_Dataset_v6_PRIS.xlsx")
RAW = os.path.join(ROOT, "data", "source", "PRIS_annual_values_raw.csv")


def mean_years(df, a, b):
    x = df[(df.Year >= a) & (df.Year <= b)].UCF.dropna()
    return x.mean() if len(x) else np.nan


def agreement(book, pris):
    m = pd.DataFrame({"b": book, "p": pris}).dropna()
    e = (m.b - m.p).abs()
    return {"n": int(len(m)), "mean_abs_diff": round(float(e.mean()), 2), "median_abs_diff": round(float(e.median()), 2),
            "share_within_0.5": round(float((e <= 0.5).mean() * 100), 1), "share_within_1": round(float((e <= 1.0).mean() * 100), 1),
            "pearson_r": round(float(np.corrcoef(m.b, m.p)[0, 1]), 4)}


def main():
    os.makedirs(OUT, exist_ok=True)
    final = pd.read_excel(V6, sheet_name="FINAL_DATASET")
    book = pd.read_excel(V5, sheet_name="UCL_LONGITUDAL")
    pcl_ref = pd.read_excel(V5, sheet_name="PCL_REFERENCE")
    raw = pd.read_csv(RAW)

    by_unit = {u: g.sort_values("Year") for u, g in raw.groupby("Unit")}
    rows = []
    for _, r in final.iterrows():
        g = by_unit[r.PRIS_Name]
        # years up to and including the end year, all years PRIS holds (lifetime)
        rows.append({
            "Reactor_Name": r.Reactor_Name,
            "PRIS_2013_2017": mean_years(g, 2013, 2017),
            "PRIS_2010_2020": mean_years(g, 2010, 2020),
            "PRIS_life_to_2020": mean_years(g, 0, 2020),
            "PRIS_2014_2024": mean_years(g, 2014, 2024),
            "PRIS_life_to_2024": mean_years(g, 0, 2024),
        })
    pr = pd.DataFrame(rows).merge(book, on="Reactor_Name", how="left")
    pr = pr.rename(columns={"UCF_2013_2017": "Book_2013_2017", "UCF_2010_2020": "Book_2010_2020",
                            "UCF_2014_2024": "Book_2014_2024"})
    pr.round(3).to_csv(os.path.join(OUT, "UCF_book_vs_PRIS.csv"), index=False)

    summary = {
        "n_reactors_in_dataset": int(len(final)),
        "ucf_2013_2017_vs_PRIS_2013_2017": agreement(pr.Book_2013_2017, pr.PRIS_2013_2017),
        "ucf_2010_2020_vs_PRIS_period": agreement(pr.Book_2010_2020, pr.PRIS_2010_2020),
        "ucf_2010_2020_vs_PRIS_lifetime_to_2020": agreement(pr.Book_2010_2020, pr.PRIS_life_to_2020),
        "ucf_2014_2024_vs_PRIS_period": agreement(pr.Book_2014_2024, pr.PRIS_2014_2024),
        "ucf_2014_2024_vs_PRIS_lifetime_to_2024": agreement(pr.Book_2014_2024, pr.PRIS_life_to_2024),
    }

    # planned loss by type and size, 2021 to 2023
    m = final[["PRIS_Name", "Type", "Capacity_Category"]].merge(raw, left_on="PRIS_Name", right_on="Unit")
    m = m[(m.Year >= 2021) & (m.Year <= 2023)]
    out = []
    for _, r in pcl_ref.iterrows():
        cat = r.Capacity_Category
        sub = m[(m.Type == r.Type) & ((m.Capacity_Category == cat) | (str(cat).upper() == "ALL"))]
        bk = r["PCL_RDS2_2024_ed"]
        out.append({"Type": r.Type, "Size": cat, "Units": int(sub.PRIS_Name.nunique()), "Unit_years": int(sub.PUF.notna().sum()),
                    "RDS2_2024_Table18_PCL": bk, "PRIS_mean_PUF": round(float(sub.PUF.mean()), 2) if sub.PUF.notna().any() else np.nan})
    pc = pd.DataFrame(out)
    pc["Difference"] = (pc.PRIS_mean_PUF - pc.RDS2_2024_Table18_PCL).round(2)
    pc.to_csv(os.path.join(OUT, "PCL_2021_2023_book_vs_PRIS.csv"), index=False)
    ok = pc.dropna(subset=["RDS2_2024_Table18_PCL", "PRIS_mean_PUF"])
    ok = ok[ok.Units >= 5]
    summary["pcl_2021_2023"] = {"groups_compared": int(len(ok)), "mean_abs_diff": round(float(ok.Difference.abs().mean()), 2),
                                "largest_abs_diff": round(float(ok.Difference.abs().max()), 2),
                                "rule": "groups with at least 5 dataset reactors"}
    # fleet-wide annual UCL in the raw PRIS table (all reporting units, unweighted)
    yr = raw.dropna(subset=["UCL"]).groupby("Year").UCL.agg(["mean", "median", "count"])
    summary["fleet_ucl_by_year_all_reporting_units"] = {
        str(y): {"mean": round(float(yr.loc[y, "mean"]), 2), "median": round(float(yr.loc[y, "median"]), 2), "units": int(yr.loc[y, "count"])}
        for y in (1990, 2000, 2010, 2020, 2023, 2024)}
    summary["raw_table"] = {"unit_years": int(len(raw)), "units": int(raw.Unit.nunique()), "first_year": int(raw.Year.min()), "last_year": int(raw.Year.max())}
    with open(os.path.join(OUT, "validation_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    print(pc.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
