"""
Build version 6 of the UCL dataset: the label and the history features come from the IAEA PRIS annual values.

Run from the repository root:
    python tools/build_dataset_v6.py --design data/UCL_Dataset_v5_longitudinal.xlsx \
        --pris data/source/PRIS_annual_values_raw.csv --out data

Inputs
  --design   workbook with the corrected design columns (UCL_Dataset_v5_longitudinal.xlsx: FINAL_DATASET and RAW_DATA)
  --pris     data/source/PRIS_annual_values_raw.csv (one row per unit and year, as downloaded from the PRIS Data Browser)
  --out      output folder (workbook, flat CSV, data dictionary are written here)

Rules (all of them are constants below, so nothing is hidden):
  A unit-year is USABLE if PRIS published a UCL value for it, the unit was not completely shut down for planned
  reasons in that year (UCF = 0 and UCL = 0), and it is not the year of first grid connection.
  Label       UCL_Value = mean annual UCL over usable years 2018-2024, at least 3 usable years.
  History     mean UCL and mean UCF over usable years in 2010-2013 and in 2014-2017, at least 2 usable years per block.
  Classes     Low < 2, Medium 2 to 6 (inclusive), High > 6 (decided on the value rounded to 2 decimals).
"""
import argparse
import re
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.properties import PageSetupProperties

ap = argparse.ArgumentParser()
ap.add_argument("--design", required=True)
ap.add_argument("--pris", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--xlsx-name", default="UCL_Dataset_v6_PRIS.xlsx")
ap.add_argument("--csv-name", default="UCL_Dataset_flat.csv")
ap.add_argument("--md-name", default="DATA_DICTIONARY.md")
args = ap.parse_args()

# ----------------------------------------------------------------------------- rules
LABEL_FIRST, LABEL_LAST, LABEL_MIN_YEARS = 2018, 2024, 3
HIST_BLOCKS = [(2010, 2013), (2014, 2017)]
HIST_MIN_YEARS = 2
LOW_BELOW, HIGH_ABOVE = 2.0, 6.0
CLASSES = ["Low", "Medium", "High"]
# Dataset spelling -> PRIS spelling (only where they differ after upper-casing and tidying hyphens and spaces)
ALIAS = {"ZAPORIZHZHYA": "ZAPORIZHZHIA", "KHMELNYTSKYY": "KHMELNYTSKYI", "COMANCHEPEAK": "COMANCHE PEAK"}


def norm(s):
    s = str(s).upper().replace("’", "'")
    s = re.sub(r"\s*-\s*", "-", s)
    return re.sub(r"\s+", " ", s).strip()


def pris_key(dataset_name):
    k = norm(dataset_name)
    for a, b in ALIAS.items():
        k = k.replace(a, b)
    return k


def klass(v):
    return "Low" if v < LOW_BELOW else ("Medium" if v <= HIGH_ABOVE else "High")


# ----------------------------------------------------------------------------- read design columns
fin = pd.read_excel(args.design, sheet_name="FINAL_DATASET")
fin = fin[fin["Reactor_Name"].notna() & fin["UCL_Class"].isin(CLASSES)].copy().reset_index(drop=True)
raw = pd.read_excel(args.design, sheet_name="RAW_DATA")
raw = raw[raw["Net_Capacity_MW"].notna()].copy().reset_index(drop=True)
for d in (fin, raw):
    d["Reactor_Name"] = d["Reactor_Name"].astype(str).str.strip()
assert len(fin) == len(raw) == 407 and (fin.Reactor_Name.values == raw.Reactor_Name.values).all()
assert fin.Reactor_Name.is_unique
for c in ["Country", "Type", "NSSS_Supplier", "Reactor_Model"]:
    fin[c] = fin[c].astype(str).str.strip().str.upper()
design = fin.drop(columns=["UCL_Value", "UCL_Class"]).merge(raw[["Reactor_Name", "Thermal_Capacity_MW", "Const_Start_Year"]], on="Reactor_Name", how="left")
assert design.Thermal_Capacity_MW.notna().all() and design.Const_Start_Year.notna().all()

# ----------------------------------------------------------------------------- read PRIS and match
pr = pd.read_csv(args.pris)
need = {"Country", "Year", "Unit", "EAF", "XUF", "FLR", "PUF", "UCF", "UCL", "UUF", "MW", "MWh"}
assert need <= set(pr.columns), need - set(pr.columns)
assert not pr.duplicated(["Unit", "Year"]).any()
pr["key"] = pr.Unit.map(norm)
assert pr.groupby("key").Unit.nunique().max() == 1
design["key"] = design.Reactor_Name.map(pris_key)
assert design.key.is_unique
missing = sorted(set(design.key) - set(pr.key))
assert not missing, ("not found in PRIS", missing)
spelled = pr.drop_duplicates("key").set_index("key").Unit
design["PRIS_Name"] = design.key.map(spelled)

p = pr[pr.key.isin(set(design.key))].merge(design[["key", "Reactor_Name", "First_Grid_Year"]], on="key")
p["fully_shut"] = (p.UCF == 0) & (p.UCL == 0)
p["no_value"] = p.UCL.isna()
p["grid_year"] = p.Year == p.First_Grid_Year
p["usable"] = ~p.no_value & ~p.fully_shut & ~p.grid_year
p["reason"] = np.select([p.no_value, p.fully_shut, p.grid_year],
                        ["No UCL value published", "Fully shut down (UCF = 0 and UCL = 0)", "Year of first grid connection"], "")


def block_mean(a, b, col):
    w = p[p.usable & p.Year.between(a, b)].groupby("Reactor_Name")[col].agg(["mean", "count"])
    return w


lab = block_mean(LABEL_FIRST, LABEL_LAST, "UCL")
lab = lab[lab["count"] >= LABEL_MIN_YEARS]
design = design.merge(lab.rename(columns={"mean": "UCL_Value", "count": "UCL_Years_Used"}), left_on="Reactor_Name", right_index=True, how="left")
design["labelled"] = design.UCL_Value.notna()
design["UCL_Value"] = design.UCL_Value.round(2) + 0.0
design["UCL_Class"] = [klass(v) if pd.notna(v) else "" for v in design.UCL_Value]

hist = design[["Reactor_Name"]].copy()
for a, b in HIST_BLOCKS:
    for col in ("UCL", "UCF"):
        w = block_mean(a, b, col)
        w = w[w["count"] >= HIST_MIN_YEARS]
        hist = hist.merge(w["mean"].rename(f"{col}_{a}_{b}"), left_on="Reactor_Name", right_index=True, how="left")
    cnt = p[p.usable & p.Year.between(a, b)].groupby("Reactor_Name").Year.count().rename(f"Years_{a}_{b}")
    hist = hist.merge(cnt, left_on="Reactor_Name", right_index=True, how="left")
    hist[f"Years_{a}_{b}"] = hist[f"Years_{a}_{b}"].fillna(0).astype(int)
    below = hist[f"Years_{a}_{b}"] < HIST_MIN_YEARS
    for col in ("UCL", "UCF"):
        hist.loc[below, f"{col}_{a}_{b}"] = np.nan
for col in [c for c in hist.columns if c.startswith(("UCL_", "UCF_"))]:
    hist[col] = hist[col].round(2) + 0.0
hist["Model2_Subset"] = (hist.UCL_2010_2013.notna() & hist.UCL_2014_2017.notna()).astype(int)

# ----------------------------------------------------------------------------- tables
lab_names = set(design.loc[design.labelled, "Reactor_Name"])
FINAL_COLS = ["Reactor_Name", "Country", "Type", "Net_Capacity_MW", "Capacity_Category", "First_Grid_Year",
              "Reactor_Age", "Operational_Years", "UCL_Value", "UCL_Class", "NSSS_Supplier", "Reactor_Model",
              "Construction_Duration", "Thermal_Efficiency_Pct", "UCL_Years_Used", "PRIS_Name"]
final = design.loc[design.labelled, FINAL_COLS].copy().reset_index(drop=True)
final["UCL_Years_Used"] = final.UCL_Years_Used.astype(int)
history = hist[hist.Reactor_Name.isin(lab_names)].reset_index(drop=True)
assert (final.Reactor_Name.values == history.Reactor_Name.values).all()

yearly = p[p.Year.between(HIST_BLOCKS[0][0], LABEL_LAST)].copy()


def window_name(y):
    for a, b in HIST_BLOCKS:
        if a <= y <= b:
            return f"History {a}-{b}"
    return f"Label {LABEL_FIRST}-{LABEL_LAST}"


yearly["Window"] = yearly.Year.map(window_name)
yearly["Used"] = np.where(yearly.usable, "yes", "no")
yearly = yearly.rename(columns={"Unit": "PRIS_Name"})
yearly = yearly[["Reactor_Name", "PRIS_Name", "Year", "Window", "UCL", "UCF", "PUF", "UUF", "XUF", "EAF", "Used", "reason"]]
yearly = yearly.rename(columns={"reason": "Reason_not_used"})
order = {n: i for i, n in enumerate(design.Reactor_Name)}
yearly = yearly.sort_values(["Reactor_Name", "Year"], key=lambda s: s.map(order) if s.name == "Reactor_Name" else s).reset_index(drop=True)

dropped = design.loc[~design.labelled, ["Reactor_Name", "Country", "Type", "First_Grid_Year"]].copy()
nuse = p[p.usable & p.Year.between(LABEL_FIRST, LABEL_LAST)].groupby("Reactor_Name").Year.count()
dropped["Usable_Years_2018_2024"] = dropped.Reactor_Name.map(nuse).fillna(0).astype(int)
dropped["Reason"] = f"Fewer than {LABEL_MIN_YEARS} usable years in {LABEL_FIRST}-{LABEL_LAST}"
n_drop = len(dropped)
NINE = [("SHIDAO BAY-1", "HTGR technology, no IAEA PCL reference, insufficient operational history"),
        ("AKADEMIK LOMONOSOV-1", "Floating power unit"), ("AKADEMIK LOMONOSOV-2", "Floating power unit"),
        ("BILIBINO-2", "Very small Arctic unit (11 MWe), no meaningful peer group"),
        ("BILIBINO-3", "Very small Arctic unit (11 MWe), no meaningful peer group"),
        ("BILIBINO-4", "Very small Arctic unit (11 MWe), no meaningful peer group"),
        ("ZHANGZHOU-1", "Only first-year commissioning data existed"),
        ("FLAMANVILLE-3", "Only first-year commissioning data existed"),
        ("MOCHOVCE-3", "Only first-year commissioning data existed")]
excluded = pd.concat([
    pd.DataFrame({"Reactor_Name": [n for n, _ in NINE], "Country": "", "Type": "", "First_Grid_Year": pd.NA,
                  "Usable_Years_2018_2024": pd.NA, "Reason": ["Not in the 407-reactor population: " + w for _, w in NINE]}),
    dropped.assign(Reason=dropped.Reason)], ignore_index=True)

# ----------------------------------------------------------------------------- consistency checks (stop on any disagreement)
assert len(final) == 407 - n_drop
assert (final.Reactor_Age == 2024 - final.First_Grid_Year).all()
assert (final.Operational_Years == np.minimum(10, final.Reactor_Age)).all()
assert (final.Capacity_Category == np.where(final.Net_Capacity_MW < 600, "<600 MWe", ">=600 MWe")).all()
assert final.UCL_Value.between(0, 100).all() and (final.UCL_Years_Used >= LABEL_MIN_YEARS).all()
assert (final.UCL_Class == final.UCL_Value.map(klass)).all()
cnt = final.UCL_Class.value_counts()
n_c = final.Country.nunique()
n_m2 = int(history.Model2_Subset.sum())
near = int(((final.UCL_Value - LOW_BELOW).abs() <= 0.005).sum() + ((final.UCL_Value - HIGH_ABOVE).abs() <= 0.005).sum())

flat = final.merge(design[["Reactor_Name", "Thermal_Capacity_MW", "Const_Start_Year"]], on="Reactor_Name").merge(
    history.drop(columns=["Years_2010_2013", "Years_2014_2017"]), on="Reactor_Name")
FLAT_COLS = ["Reactor_Name", "PRIS_Name", "Country", "Type", "Capacity_Category", "Net_Capacity_MW", "Thermal_Capacity_MW",
             "Thermal_Efficiency_Pct", "NSSS_Supplier", "Reactor_Model", "Const_Start_Year", "First_Grid_Year",
             "Construction_Duration", "Reactor_Age", "Operational_Years", "UCL_2010_2013", "UCL_2014_2017",
             "UCF_2010_2013", "UCF_2014_2017", "Model2_Subset", "UCL_Value", "UCL_Years_Used", "UCL_Class"]
flat = flat[FLAT_COLS]

# ----------------------------------------------------------------------------- documentation text
TITLE = f"UCL dataset (version 6): unplanned capability loss class of {len(final)} nuclear power reactors"
INTRO = [
    ("What this is",
     f"One row per nuclear power reactor ({len(final)} reactors in {n_c} countries). It supports the paper 'Machine Learning "
     "Classification of Unplanned Capability Loss Risk in Nuclear Power Plants: Evidence from IAEA RDS-2 Data' and the web app "
     "UCL Risk Predictor. Design columns come from IAEA RDS-2 and PRIS reactor details. The label and the history columns come from "
     "the annual performance values that the IAEA PRIS Data Browser publishes for every reactor."),
    ("Target",
     f"UCL_Value is the mean of the annual unplanned capability loss factor (UCL) published in PRIS for {LABEL_FIRST} to {LABEL_LAST}, in percent. "
     f"Classes: Low below {LOW_BELOW:g} percent, Medium {LOW_BELOW:g} to {HIGH_ABOVE:g} percent, High above {HIGH_ABOVE:g} percent. "
     f"Classes in this file: Low {cnt['Low']}, Medium {cnt['Medium']}, High {cnt['High']}. The class limits were chosen by the authors; "
     f"{near} reactors lie within 0.005 of a limit."),
    ("Which years count",
     "A year is skipped when PRIS has no UCL value for it, when the reactor was completely shut down for planned reasons that year "
     "(UCF = 0 and UCL = 0), and in the year of first grid connection (a partial year). A reactor needs at least "
     f"{LABEL_MIN_YEARS} usable years in {LABEL_FIRST} to {LABEL_LAST}; {n_drop} reactors of the earlier 407 do not have them and are listed in the EXCLUDED sheet. "
     "UCL_Years_Used shows how many years went into each average. The PRIS_YEARLY sheet shows every year and whether it was used."),
    ("History",
     f"HISTORY holds the mean UCL and the mean UCF for 2010 to 2013 and for 2014 to 2017 (at least {HIST_MIN_YEARS} usable years per block, same skipping rules). "
     f"All of these years lie before the label window. {n_m2} reactors have both UCL blocks (Model2_Subset = 1); these form the Model 2 subset."),
    ("Which sheets the scripts read", "Only FINAL_DATASET and HISTORY. The other sheets document where the numbers come from."),
    ("Not an input", "UCL_Value, UCL_Class and UCL_Years_Used describe the label window and are never model inputs."),
    ("Population",
     "Reactors listed as operational at the end of 2024 in RDS-2 (2025 edition), minus nine that were left out (see EXCLUDED), minus the "
     f"{n_drop} that lack enough usable years. Reactors that were permanently shut down before the end of 2024 are not included, so the data describe the current fleet."),
    ("Checks",
     "The PRIS annual values reproduce figures that IAEA prints: the 2013 to 2017 UCF of RDS-2 2018 Table 14 for 356 reactors (median absolute difference 0.03 points) "
     "and the 2021 to 2023 planned capability loss by reactor type and size of RDS-2 2024 Table 18 (mean absolute difference 0.39 points over six groups). "
     "tools/audit_dataset.py re-computes the label and the history columns from the raw download and checks the other columns; its last report is data/AUDIT_REPORT.txt."),
    ("Note on RDS-2 Table 14",
     "Version 5 of this dataset used the Table 14 columns headed 2010-2020 (2021 edition) and 2014-2024 (2025 edition). Compared with the annual PRIS values, "
     "those columns match averages over the whole life of each reactor up to 2020 and 2024 (mean absolute difference 0.60 and 0.45 points) and not the stated periods "
     "(5.23 and 5.27 points). They were therefore not used here."),
]
SOURCES = [
    ("UCL_Value, UCL_Years_Used, history columns, PRIS_YEARLY",
     "IAEA PRIS Data Browser, Performance indicators report (https://pris.iaea.org), annual UCL, UCF, PUF, UUF, XUF and EAF per reactor and year; downloaded October 2026. The unfiltered download is in data/source/PRIS_annual_values_raw.csv."),
    ("Net_Capacity_MW, Thermal_Capacity_MW, Capacity_Category, Thermal_Efficiency_Pct", "IAEA RDS-2, 2025 edition, Table 14, and PRIS reactor details"),
    ("First_Grid_Year, Const_Start_Year, Construction_Duration, Reactor_Age, Operational_Years", "IAEA RDS-2 (2018 and 2025 editions, Table 14) and PRIS reactor details; the last three are computed from the first two"),
    ("Country, Type, NSSS_Supplier, Reactor_Model", "IAEA PRIS reactor details and RDS-2 Table 14"),
]
SHEETS = [
    ("FINAL_DATASET", str(len(final)), "The table the scripts read. One row per reactor, with the label."),
    ("HISTORY", str(len(history)), "Mean UCL and UCF for 2010 to 2013 and 2014 to 2017 per reactor, joined to FINAL_DATASET by Reactor_Name."),
    ("PRIS_YEARLY", str(len(yearly)), "Every reactor-year from 2010 to 2024 that the numbers above are built from, with a flag for whether it was used and why not."),
    ("EXCLUDED", str(len(excluded)), "Reactors that are not in FINAL_DATASET and the reason."),
]
COLS = {
    "FINAL_DATASET": [
        ("Reactor_Name", "Reactor name. Unique. Join key. Spelling follows earlier versions of this dataset; PRIS_Name gives the PRIS spelling."),
        ("Country", "Country name as listed by PRIS, in capitals."),
        ("Type", "Reactor type: PWR, BWR, PHWR, GCR, LWGR or FBR."),
        ("Net_Capacity_MW", "Net electrical capacity in MW."),
        ("Capacity_Category", "'<600 MWe' or '>=600 MWe'."),
        ("First_Grid_Year", "Year of first grid connection."),
        ("Reactor_Age", "Age in years in 2024 (2024 minus First_Grid_Year)."),
        ("Operational_Years", "Years since first grid connection up to 2024, at most 10. Kept from version 5 as a maturity feature; its cap of 10 is no longer the label window."),
        ("UCL_Value", f"Mean annual unplanned capability loss, {LABEL_FIRST} to {LABEL_LAST}, percent, rounded to 2 decimals. The numeric label."),
        ("UCL_Class", "Low, Medium or High. This is the label the models learn."),
        ("NSSS_Supplier", "Supplier of the nuclear steam supply system, as listed by PRIS."),
        ("Reactor_Model", "Design model name, as listed by PRIS."),
        ("Construction_Duration", "Years from the start of construction to first grid connection."),
        ("Thermal_Efficiency_Pct", "Net electrical capacity divided by thermal capacity, in percent. The scripts use a constant 33.0 instead unless they are told to use this column."),
        ("UCL_Years_Used", f"Number of usable years in {LABEL_FIRST} to {LABEL_LAST} behind UCL_Value (3 to 7)."),
        ("PRIS_Name", "Name of the reactor in the PRIS table. Differs from Reactor_Name for ZAPORIZHZHIA-1 to 6, KHMELNYTSKYI-1 and 2, and COMANCHE PEAK-2."),
    ],
    "HISTORY": [
        ("Reactor_Name", "Join key."),
        ("UCL_2010_2013, UCL_2014_2017", "Mean annual UCL in percent for the block. Blank if fewer than 2 usable years."),
        ("UCF_2010_2013, UCF_2014_2017", "Mean annual unit capability factor in percent for the block. Same rules."),
        ("Years_2010_2013, Years_2014_2017", "Number of usable years in the block."),
        ("Model2_Subset", "1 if both UCL blocks are available, otherwise 0."),
    ],
    "PRIS_YEARLY": [
        ("Reactor_Name, PRIS_Name, Year", "Reactor, its PRIS spelling and the calendar year."),
        ("Window", "Which block the year belongs to."),
        ("UCL, UCF, PUF, UUF, XUF, EAF", "PRIS annual factors in percent: unplanned capability loss, unit capability factor, planned unavailability, unplanned unavailability, external unavailability, energy availability factor. Blank where PRIS has no value."),
        ("Used", "yes if the year went into the averages, otherwise no."),
        ("Reason_not_used", "No UCL value published; Fully shut down (UCF = 0 and UCL = 0); or Year of first grid connection."),
    ],
    "EXCLUDED": [
        ("Reactor_Name, Country, Type, First_Grid_Year", "Identification (blank for the nine reactors that were never in the population)."),
        ("Usable_Years_2018_2024", "Usable years found, for reactors that were in the earlier 407."),
        ("Reason", "Why the reactor is not in FINAL_DATASET."),
    ],
}
CHANGES = ("Version 6 replaces the label of version 5. Version 5 estimated unplanned loss as 100 - UCF - PCL, with PCL a planned-loss average for the reactor's "
           "type and size class, and it agreed with the real PRIS unplanned loss for only 58 percent of reactors. Version 6 uses the PRIS annual UCL directly. "
           "The UCF history columns, PCL_REFERENCE and RAW_DATA of version 5 are retired. Design columns are unchanged from the corrected version 5, "
           "which fixed grid year and construction duration of Tianwan-1 and -2, net capacity of Tarapur-3 and -4, Daya Bay-1, Dukovany-2 to 4, Ringhals-4 and Forsmark-1, "
           "and thermal capacity of Rostov-2 against the printed RDS-2 tables.")
TERMS = ("The numbers come from IAEA publications and the IAEA PRIS database, which remain the property of the IAEA; please cite them together with the paper. "
         "This workbook is a derived research dataset, shared so that readers can check the paper's results. Check the IAEA terms of use before reusing the "
         "underlying data for any other purpose. The MIT licence in the repository root covers the code only.")

# ----------------------------------------------------------------------------- write workbook
FONT = "Arial"
f_norm, f_bold = Font(name=FONT, size=10), Font(name=FONT, size=10, bold=True)
thin = Side(style="thin", color="000000")
TWO_DP = {"UCL_Value", "Thermal_Efficiency_Pct", "UCL_2010_2013", "UCL_2014_2017", "UCF_2010_2013", "UCF_2014_2017",
          "UCL", "UCF", "PUF", "UUF", "XUF", "EAF"}


def put_table(ws, df):
    cols = list(df.columns)
    for j, c in enumerate(cols, 1):
        cell = ws.cell(row=1, column=j, value=c)
        cell.font = f_bold
        cell.border = Border(bottom=thin)
        num = pd.api.types.is_numeric_dtype(df[c])
        cell.alignment = Alignment(horizontal="right" if num else "left", indent=0 if num else 1)
    for i, row in enumerate(df.itertuples(index=False), 2):
        for j, v in enumerate(row, 1):
            c = cols[j - 1]
            if v is pd.NA or v is None or (isinstance(v, float) and np.isnan(v)):
                v = None
            elif isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (float, np.floating)):
                v = float(v) if c in TWO_DP or not float(v).is_integer() else int(v)
            cell = ws.cell(row=i, column=j, value=v)
            cell.font = f_norm
            if isinstance(v, str):
                cell.alignment = Alignment(horizontal="left", indent=1)
            if isinstance(v, float):
                cell.number_format = "0.00"
    for j, c in enumerate(cols, 1):
        vals = [len(str(x)) for x in df.iloc[:, j - 1].tolist() if pd.notna(x)]
        ws.column_dimensions[get_column_letter(j)].width = max(8, min(max([len(c)] + vals) + 2, 60))
    ws.freeze_panes = "A2"


wb = Workbook()
ws = wb.active
ws.title = "README"
r = 1


def line(a="", b="", c="", bold=False):
    global r
    for j, v in enumerate((a, b, c), 1):
        if v != "":
            cell = ws.cell(row=r, column=j, value=v)
            cell.font = f_bold if bold else f_norm
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    r += 1


line(TITLE, bold=True)
ws.cell(row=1, column=1).alignment = Alignment(wrap_text=False, vertical="top")
line()
for k, v in INTRO:
    line(k, v)
line(); line("Where the numbers come from", bold=True)
for k, v in SOURCES:
    line(k, v)
line(); line("Sheets", bold=True)
for n, rows, what in SHEETS:
    line(n, what, f"{rows} rows")
line(); line("Columns", bold=True)
for sheet, items in COLS.items():
    line(sheet, bold=True)
    for k, v in items:
        line("   " + k, v)
line(); line("Changes from the previous release", bold=True); line("", CHANGES)
line(); line("Terms of use", bold=True); line("", TERMS)
ws.column_dimensions["A"].width = 40
ws.column_dimensions["B"].width = 110
ws.column_dimensions["C"].width = 12

for name, df in [("FINAL_DATASET", final), ("HISTORY", history), ("PRIS_YEARLY", yearly), ("EXCLUDED", excluded)]:
    put_table(wb.create_sheet(name), df)
for s in wb.worksheets:
    s.page_setup.orientation = "landscape"
    s.page_setup.fitToWidth = 1
    s.page_setup.fitToHeight = 0
    s.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
    if s.title != "README":
        s.print_title_rows = "1:1"
wb.properties.title = TITLE
wb.properties.creator = "UCL risk predictor project"
import os
os.makedirs(args.out, exist_ok=True)
wb.save(os.path.join(args.out, args.xlsx_name))
flat.to_csv(os.path.join(args.out, args.csv_name), index=False, encoding="utf-8", lineterminator="\n", float_format="%.10g")

md = ["# Data dictionary", "", f"File: `{args.xlsx_name}` (plus the flat copy `{args.csv_name}`)", ""]
for k, v in INTRO:
    md += [f"**{k}.** {v}", ""]
md += ["## Where the numbers come from", "", "| Quantity | Source |", "|---|---|"] + [f"| {k} | {v} |" for k, v in SOURCES] + [""]
md += ["## Sheets", "", "| Sheet | Rows | What it holds |", "|---|---|---|", "| `README` | - | The same information as this file, inside the workbook. |"]
md += [f"| `{n}` | {rows} | {what} |" for n, rows, what in SHEETS] + [""]
for sheet, items in COLS.items():
    md += [f"## `{sheet}` columns", "", "| Column | Meaning |", "|---|---|"] + [f"| `{k.strip()}` | {v} |" for k, v in items] + [""]
md += [f"## `{args.csv_name}`", "",
       f"One row per labelled reactor with the design columns, the history columns and the label in one table ({len(FLAT_COLS)} columns), for readers who prefer a plain text file. "
       "It is for reading and checking. The scripts read the workbook, not the CSV.", ""]
md += ["## Features built by the code", "",
       "Supplier group, model family, age group, operational category, country encoding and the other model inputs are computed by the scripts "
       "(`ucl_core.py` and the files in `research/`) from the columns above. They are not stored here.", ""]
md += ["## Changes from the previous release", "", CHANGES, ""]
md += ["## Terms of use", "", TERMS, ""]
open(os.path.join(args.out, args.md_name), "w", encoding="utf-8").write("\n".join(md))

print("labelled reactors:", len(final), "| countries:", n_c, "| classes:", cnt.reindex(CLASSES).to_dict())
print("dropped:", n_drop, "| Model 2 subset (both UCL blocks):", n_m2, "| within 0.005 of a class limit:", near)
print("PRIS_YEARLY rows:", len(yearly), "| used:", int((yearly.Used == 'yes').sum()))
