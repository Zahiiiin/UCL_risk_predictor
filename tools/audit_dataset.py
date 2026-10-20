"""
Audit of the UCL dataset (version 6). It re-computes the label and the history columns from the raw PRIS download with
code that is written separately from the builder (plain Python, no pandas), and checks the design columns for
internal consistency and against PRIS. Run it after every change to the data:

    python tools/audit_dataset.py --xlsx data/UCL_Dataset_v6_PRIS.xlsx --csv data/UCL_Dataset_flat.csv \
        --pris data/source/PRIS_annual_values_raw.csv

Every line starts with PASS, WARN or FAIL. A FAIL means the dataset is wrong. A WARN means a value deserves a look
but is not necessarily wrong. The script exits with a non-zero code if anything FAILED.
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from openpyxl import load_workbook

ap = argparse.ArgumentParser()
ap.add_argument("--xlsx", required=True)
ap.add_argument("--csv", required=True)
ap.add_argument("--pris", required=True)
ap.add_argument("--report", default="", help="optional path of a text file to write the report to")
a = ap.parse_args()

LABEL = (2018, 2024)
LABEL_MIN = 3
BLOCKS = [(2010, 2013), (2014, 2017)]
BLOCK_MIN = 2
ALIAS = {"ZAPORIZHZHYA": "ZAPORIZHZHIA", "KHMELNYTSKYY": "KHMELNYTSKYI", "COMANCHEPEAK": "COMANCHE PEAK"}
OUT = []
FAILS = 0


def say(level, msg):
    global FAILS
    if level == "FAIL":
        FAILS += 1
    OUT.append(f"{level}  {msg}")
    print(OUT[-1], flush=True)


def check(cond, ok_msg, bad_msg):
    say("PASS" if cond else "FAIL", ok_msg if cond else bad_msg)
    return cond


def key(name):
    s = str(name).upper().replace("’", "'")
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s+", " ", s).strip()
    for x, y in ALIAS.items():
        s = s.replace(x, y)
    return s


def num(v):
    if v is None or v == "":
        return None
    return float(v)


def sheet_rows(wb, name):
    ws = wb[name]
    rows = list(ws.iter_rows(values_only=True))
    head = list(rows[0])
    return head, [dict(zip(head, r)) for r in rows[1:] if any(c is not None for c in r)]


# ------------------------------------------------------------------ load
wb = load_workbook(a.xlsx, read_only=True, data_only=True)
check(set(wb.sheetnames) == {"README", "FINAL_DATASET", "HISTORY", "PRIS_YEARLY", "EXCLUDED"},
      "workbook has the five expected sheets", f"unexpected sheets: {wb.sheetnames}")
_, fin = sheet_rows(wb, "FINAL_DATASET")
_, his = sheet_rows(wb, "HISTORY")
_, yrs = sheet_rows(wb, "PRIS_YEARLY")
_, exc = sheet_rows(wb, "EXCLUDED")
with open(a.csv, newline="", encoding="utf-8") as fh:
    flat = list(csv.DictReader(fh))
with open(a.pris, newline="", encoding="utf-8") as fh:
    pris = list(csv.DictReader(fh))
print(f"loaded: {len(fin)} FINAL_DATASET rows, {len(his)} HISTORY rows, {len(yrs)} PRIS_YEARLY rows, {len(exc)} EXCLUDED rows, "
      f"{len(flat)} flat CSV rows, {len(pris)} raw PRIS rows\n")

# ------------------------------------------------------------------ 1. structure
names = [r["Reactor_Name"] for r in fin]
check(len(names) == len(set(names)) and all(names), "Reactor_Name is present and unique", "duplicate or blank Reactor_Name")
check([r["Reactor_Name"] for r in his] == names, "HISTORY lists the same reactors in the same order", "HISTORY and FINAL_DATASET differ")
check([r["Reactor_Name"] for r in flat] == names, "flat CSV lists the same reactors in the same order", "flat CSV and FINAL_DATASET differ")
required = ["Country", "Type", "Net_Capacity_MW", "Capacity_Category", "First_Grid_Year", "Reactor_Age", "Operational_Years",
            "UCL_Value", "UCL_Class", "NSSS_Supplier", "Reactor_Model", "Construction_Duration", "Thermal_Efficiency_Pct",
            "UCL_Years_Used", "PRIS_Name"]
blank = [(r["Reactor_Name"], c) for r in fin for c in required if r.get(c) in (None, "")]
check(not blank, "no blank cell in any FINAL_DATASET column", f"blank cells: {blank[:10]}")
check({r["Type"] for r in fin} <= {"PWR", "BWR", "PHWR", "GCR", "LWGR", "FBR"}, "reactor types are the six expected ones",
      f"unexpected types {sorted({r['Type'] for r in fin})}")
check({r["UCL_Class"] for r in fin} <= {"Low", "Medium", "High"}, "classes are Low, Medium or High", "unexpected class label")
cls_count = {c: sum(1 for r in fin if r["UCL_Class"] == c) for c in ("Low", "Medium", "High")}
say("INFO", f"{len(fin)} reactors, {len({r['Country'] for r in fin})} countries, classes {cls_count}")

# ------------------------------------------------------------------ 2. design columns, internal consistency
flat_by = {r["Reactor_Name"]: r for r in flat}
bad = []
for r in fin:
    f = flat_by[r["Reactor_Name"]]
    age = 2024 - int(r["First_Grid_Year"])
    if int(r["Reactor_Age"]) != age:
        bad.append((r["Reactor_Name"], "Reactor_Age", r["Reactor_Age"], age))
    if int(r["Operational_Years"]) != min(10, age):
        bad.append((r["Reactor_Name"], "Operational_Years", r["Operational_Years"], min(10, age)))
    cat = "<600 MWe" if float(r["Net_Capacity_MW"]) < 600 else ">=600 MWe"
    if r["Capacity_Category"] != cat:
        bad.append((r["Reactor_Name"], "Capacity_Category", r["Capacity_Category"], cat))
    cd = int(r["First_Grid_Year"]) - int(float(f["Const_Start_Year"]))
    if int(r["Construction_Duration"]) != cd:
        bad.append((r["Reactor_Name"], "Construction_Duration", r["Construction_Duration"], cd))
    eff = float(r["Net_Capacity_MW"]) / float(f["Thermal_Capacity_MW"]) * 100
    if abs(float(r["Thermal_Efficiency_Pct"]) - eff) > 0.011:
        bad.append((r["Reactor_Name"], "Thermal_Efficiency_Pct", r["Thermal_Efficiency_Pct"], round(eff, 2)))
check(not bad, "Reactor_Age, Operational_Years, Capacity_Category, Construction_Duration and Thermal_Efficiency_Pct follow from their inputs",
      f"{len(bad)} inconsistencies, first: {bad[:6]}")
rng = {"Net_Capacity_MW": (50, 1800), "First_Grid_Year": (1950, 2024), "Construction_Duration": (1, 45), "Thermal_Efficiency_Pct": (20, 40)}
for c, (lo, hi) in rng.items():
    odd = [(r["Reactor_Name"], r[c]) for r in fin if not lo <= float(r[c]) <= hi]
    say("PASS" if not odd else "WARN", f"{c} inside {lo}..{hi}" if not odd else f"{c} outside {lo}..{hi}: {odd}")
diff = []
for r in fin:
    f = flat_by[r["Reactor_Name"]]
    for c in ["Country", "Type", "Capacity_Category", "NSSS_Supplier", "Reactor_Model", "PRIS_Name", "UCL_Class"]:
        if str(r[c]) != f[c]:
            diff.append((r["Reactor_Name"], c))
    for c in ["Net_Capacity_MW", "First_Grid_Year", "Reactor_Age", "Operational_Years", "Construction_Duration", "Thermal_Efficiency_Pct",
              "UCL_Value", "UCL_Years_Used"]:
        if abs(float(r[c]) - float(f[c])) > 1e-9:
            diff.append((r["Reactor_Name"], c))
check(not diff, "flat CSV agrees with FINAL_DATASET in every shared column", f"differences: {diff[:8]}")

# ------------------------------------------------------------------ 3. raw PRIS download: identities
bad_id, dup, seen = [], 0, set()
for r in pris:
    k = (r["Unit"], r["Year"])
    if k in seen:
        dup += 1
    seen.add(k)
    ucf, puf, uuf, ucl = num(r["UCF"]), num(r["PUF"]), num(r["UUF"]), num(r["UCL"])
    if None not in (ucf, puf, uuf) and abs(ucf + puf + uuf - 100) > 0.011:
        bad_id.append((r["Unit"], r["Year"], "UCF+PUF+UUF"))
    if None not in (ucl, uuf) and abs(ucl - uuf) > 0.011:
        bad_id.append((r["Unit"], r["Year"], "UCL!=UUF"))
check(dup == 0, "raw PRIS download has no duplicate unit-year", f"{dup} duplicate unit-years")
check(not bad_id, "in the raw PRIS download UCF + PUF + UUF = 100 and UCL = UUF in every row", f"{len(bad_id)} rows break an identity, first: {bad_id[:5]}")
neg = [(r["Unit"], r["Year"], r["UCF"]) for r in pris if num(r["UCF"]) is not None and num(r["UCF"]) < 0]
say("PASS" if not neg else "WARN", "no negative UCF in the raw download" if not neg else f"{len(neg)} unit-years with negative UCF in the raw download (PRIS values, kept as published): {neg[:6]}")

# ------------------------------------------------------------------ 4. recompute the label and history independently
by_unit = defaultdict(dict)
for r in pris:
    by_unit[key(r["Unit"])][int(r["Year"])] = r
miss = [r["Reactor_Name"] for r in fin if key(r["Reactor_Name"]) not in by_unit]
check(not miss, "every labelled reactor is found in the raw PRIS download", f"not found: {miss}")
excl_names = {e["Reactor_Name"] for e in exc}


def usable_values(unit_key, grid_year, y0, y1, col):
    out = []
    for y in range(y0, y1 + 1):
        row = by_unit[unit_key].get(y)
        if row is None:
            continue
        ucl, ucf = num(row["UCL"]), num(row["UCF"])
        if ucl is None:
            continue
        if ucf == 0 and ucl == 0:
            continue
        if y == grid_year:
            continue
        out.append(num(row[col]))
    return out


def klass(v):
    return "Low" if v < 2 else ("Medium" if v <= 6 else "High")


bad_lab, bad_hist = [], []
his_by = {r["Reactor_Name"]: r for r in his}
for r in fin:
    k, g = key(r["Reactor_Name"]), int(r["First_Grid_Year"])
    vals = usable_values(k, g, LABEL[0], LABEL[1], "UCL")
    if len(vals) < LABEL_MIN:
        bad_lab.append((r["Reactor_Name"], "fewer than the minimum usable years"))
        continue
    m = sum(vals) / len(vals)          # unrounded mean; the stored value must be this mean rounded to 2 decimals
    stored = float(r["UCL_Value"])
    if abs(m - stored) > 0.00501 or len(vals) != int(r["UCL_Years_Used"]) or klass(stored) != r["UCL_Class"]:
        bad_lab.append((r["Reactor_Name"], round(m, 4), r["UCL_Value"], len(vals), r["UCL_Years_Used"], klass(stored), r["UCL_Class"]))
    if klass(m) != klass(stored):
        say("WARN", f"{r['Reactor_Name']}: unrounded mean {m:.4f} and rounded value {stored} fall in different classes")
    h = his_by[r["Reactor_Name"]]
    for (y0, y1) in BLOCKS:
        for col in ("UCL", "UCF"):
            v = usable_values(k, g, y0, y1, col)
            exp = sum(v) / len(v) if len(v) >= BLOCK_MIN else None
            got = h[f"{col}_{y0}_{y1}"]
            if (exp is None) != (got is None) or (exp is not None and abs(exp - float(got)) > 0.00501):
                bad_hist.append((r["Reactor_Name"], f"{col}_{y0}_{y1}", exp, got))
        n = len(usable_values(k, g, y0, y1, "UCL"))
        if int(h[f"Years_{y0}_{y1}"]) != n:
            bad_hist.append((r["Reactor_Name"], f"Years_{y0}_{y1}", n, h[f"Years_{y0}_{y1}"]))
    both = h["UCL_2010_2013"] is not None and h["UCL_2014_2017"] is not None
    if int(h["Model2_Subset"]) != int(both):
        bad_hist.append((r["Reactor_Name"], "Model2_Subset", int(both), h["Model2_Subset"]))
check(not bad_lab, "UCL_Value, UCL_Years_Used and UCL_Class re-computed from the raw PRIS download agree for every reactor",
      f"{len(bad_lab)} disagreements, first: {bad_lab[:5]}")
check(not bad_hist, "all history columns re-computed from the raw PRIS download agree for every reactor",
      f"{len(bad_hist)} disagreements, first: {bad_hist[:5]}")
edge = [(r["Reactor_Name"], r["UCL_Value"]) for r in fin if min(abs(float(r["UCL_Value"]) - 2), abs(float(r["UCL_Value"]) - 6)) <= 0.05]
say("INFO", f"reactors within 0.05 of a class limit (their class could flip with a tiny data change): {len(edge)} {edge}")

# reactors that were dropped must really have too few usable years, and nobody may be missing
dropped = [e for e in exc if str(e["Reason"]).startswith("Fewer than")]
bad_drop = []
for e in dropped:
    k = key(e["Reactor_Name"])
    # grid year of dropped reactors is stored in the EXCLUDED sheet
    v = usable_values(k, int(e["First_Grid_Year"]), LABEL[0], LABEL[1], "UCL")
    if len(v) >= LABEL_MIN or len(v) != int(e["Usable_Years_2018_2024"]):
        bad_drop.append((e["Reactor_Name"], len(v), e["Usable_Years_2018_2024"]))
check(not bad_drop, f"the {len(dropped)} reactors dropped for too few usable years really have fewer than {LABEL_MIN}", f"dropped reactors that qualify: {bad_drop}")
check(len(fin) + len(dropped) == 407, "labelled plus dropped reactors add up to the 407-reactor population", f"{len(fin)} + {len(dropped)} != 407")

# ------------------------------------------------------------------ 5. PRIS_YEARLY equals the raw download
bad_y, used_expected = [], 0
for r in yrs:
    raw = by_unit[key(r["Reactor_Name"])].get(int(r["Year"]))
    if raw is None or raw["Unit"] != r["PRIS_Name"]:
        bad_y.append((r["Reactor_Name"], r["Year"], "row not in raw download"))
        continue
    for c in ("UCL", "UCF", "PUF", "UUF", "XUF", "EAF"):
        x, y = num(raw[c]), r[c]
        if (x is None) != (y is None) or (x is not None and abs(x - float(y)) > 1e-9):
            bad_y.append((r["Reactor_Name"], r["Year"], c, x, y))
    fg = next(int(f["First_Grid_Year"]) for f in fin if f["Reactor_Name"] == r["Reactor_Name"]) if r["Reactor_Name"] in names else None
    if fg is None:
        fg = next(int(e["First_Grid_Year"]) for e in exc if e["Reactor_Name"] == r["Reactor_Name"])
    ucl, ucf = num(raw["UCL"]), num(raw["UCF"])
    use = ucl is not None and not (ucf == 0 and ucl == 0) and int(r["Year"]) != fg
    if use != (r["Used"] == "yes"):
        bad_y.append((r["Reactor_Name"], r["Year"], "Used flag", use, r["Used"]))
    used_expected += use
check(not bad_y, f"all {len(yrs)} PRIS_YEARLY rows equal the raw download and carry the right Used flag", f"{len(bad_y)} problems, first: {bad_y[:5]}")
expected_rows = sum(1 for e in list(fin) + [x for x in dropped] for y in by_unit[key(e["Reactor_Name"])] if 2010 <= y <= 2024)
check(len(yrs) == expected_rows, "PRIS_YEARLY contains every 2010-2024 year that exists in the download for the 407 reactors",
      f"PRIS_YEARLY has {len(yrs)} rows, the download has {expected_rows}")

# ------------------------------------------------------------------ 6. design columns against PRIS (second source)
mw_bad, yr_bad, yr_late, ct_bad = [], [], [], []
for r in fin:
    rows = by_unit[key(r["Reactor_Name"])]
    if key(r["Reactor_Name"]) != key(r["PRIS_Name"]) and key(r["PRIS_Name"]) not in by_unit:
        ct_bad.append((r["Reactor_Name"], "PRIS_Name not found"))
    latest = max(y for y in rows if y <= 2024)
    mw = num(rows[latest]["MW"])
    if mw is not None and abs(mw - float(r["Net_Capacity_MW"])) > 10:
        mw_bad.append((r["Reactor_Name"], r["Net_Capacity_MW"], mw, latest))
    with_energy = [y for y in rows if (num(rows[y]["MWh"]) or 0) > 0]
    first_energy = min(with_energy) if with_energy else None
    if first_energy is not None and int(r["First_Grid_Year"]) > first_energy:
        yr_bad.append((r["Reactor_Name"], r["First_Grid_Year"], first_energy))
    if first_energy is not None and first_energy - int(r["First_Grid_Year"]) > 3:
        yr_late.append((r["Reactor_Name"], r["First_Grid_Year"], first_energy, min(rows)))
check(not ct_bad, "every PRIS_Name is found in the raw download", f"{ct_bad}")
say("PASS" if not mw_bad else "WARN", "Net_Capacity_MW is within 10 MW of the PRIS reference unit power (latest year) for every reactor"
    if not mw_bad else f"{len(mw_bad)} reactors where Net_Capacity_MW differs by more than 10 MW from PRIS reference unit power (name, dataset, PRIS, year): {mw_bad}")
check(not yr_bad, "no reactor has a First_Grid_Year later than the first PRIS year with electricity produced",
      f"First_Grid_Year after the first PRIS year with energy: {yr_bad}")
say("PASS" if not yr_late else "WARN", "First_Grid_Year is at most 3 years before the first PRIS year with electricity produced for every reactor"
    if not yr_late else f"{len(yr_late)} reactors where the first PRIS year with energy is more than 3 years after First_Grid_Year "
    f"(name, grid year, first year with energy, first year in PRIS table; PRIS annual data can start late for older units): {yr_late}")
pris_country = {}
for r in pris:
    pris_country.setdefault(key(r["Unit"]), r["Country"])
SHORT = {"RUSSIAN FEDERATION": "RUSSIA", "UNITED STATES OF AMERICA": "USA", "UNITED KINGDOM": "UK",
         "UNITED ARAB EMIRATES": "UAE", "NETHERLANDS KINGDOM OF THE": "NETHERLANDS"}


def cn(s):
    s = str(s).upper().replace(",", "").strip()
    return SHORT.get(s, s)


cbad = [(r["Reactor_Name"], r["Country"], pris_country[key(r["Reactor_Name"])]) for r in fin
        if cn(r["Country"]) != cn(pris_country[key(r["Reactor_Name"])]) and cn(r["Country"]) not in SHORT.values()]
cbad += [(r["Reactor_Name"], r["Country"], pris_country[key(r["Reactor_Name"])]) for r in fin
         if cn(r["Country"]) in SHORT.values() and cn(r["Country"]) != cn(pris_country[key(r["Reactor_Name"])])]
say("PASS" if not cbad else "WARN", "Country agrees with the PRIS country for every reactor (after the usual short names)"
    if not cbad else f"{len(cbad)} country differences to look at: {cbad[:8]}")

# ------------------------------------------------------------------ summary
print()
say("INFO", f"{FAILS} FAIL line(s)")
if a.report:
    open(a.report, "w", encoding="utf-8").write("\n".join(OUT) + "\n")
sys.exit(1 if FAILS else 0)
