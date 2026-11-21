"""
US_NRC_Daily.py
===============
Reads the NRC Power Reactor Status Reports (one text file per year, 2005 to 2025, saved by the author on
21 November 2026 from nrc.gov), maps the NRC unit names to the PRIS unit names and builds
(1) a clean daily table and (2) unit-year descriptors of the daily power history.

The daily report gives one power value (percent of rated thermal power) per unit and day.
It says nothing about WHY power was low, so planned and unplanned outages cannot be told apart directly.
"""
import glob, os, re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "nrc_power_status")
OUT = os.path.join(HERE, "results_US")

MANUAL = {"Arkansas Nuclear 1": "ANO-1", "Arkansas Nuclear 2": "ANO-2", "Callaway": "CALLAWAY-1", "Clinton": "CLINTON-1",
          "Columbia Generating Station": "COLUMBIA", "D.C. Cook 1": "COOK-1", "D.C. Cook 2": "COOK-2",
          "Davis-Besse": "DAVIS BESSE-1", "Duane Arnold": "DUANE ARNOLD-1", "Fort Calhoun": "FORT CALHOUN-1",
          "La Salle 1": "LASALLE-1", "La Salle 2": "LASALLE-2", "River Bend Station 1": "RIVER BEND-1",
          "Saint Lucie 1": "ST. LUCIE-1", "Saint Lucie 2": "ST. LUCIE-2", "Summer": "SUMMER-1", "Wolf Creek 1": "WOLF CREEK"}


def pris_name(n):
    if n in MANUAL:
        return MANUAL[n]
    m = re.fullmatch(r"(.+?) (\d+)", n)
    return (m.group(1).upper() + "-" + m.group(2)) if m else n.upper()


def load_daily():
    L = []
    for f in sorted(glob.glob(os.path.join(RAW, "*PowerStatus.txt"))):
        d = pd.read_csv(f, sep="|", dtype=str)
        d.columns = [c.strip() for c in d.columns]
        L.append(d)
    D = pd.concat(L, ignore_index=True)
    D["nrc_unit"] = D.Unit.str.strip()
    D["date"] = pd.to_datetime(D.ReportDt.str.split(" ").str[0], format="%m/%d/%Y")
    D["power"] = pd.to_numeric(D.Power.str.strip())
    D["unit"] = D.nrc_unit.map(pris_name)
    n0 = len(D)
    D = D.drop_duplicates(["unit", "date"]).sort_values(["unit", "date"]).reset_index(drop=True)
    return D[["unit", "nrc_unit", "date", "power"]], n0 - len(D)


def episodes(p, cond):
    """Lengths of runs of consecutive days where cond is true."""
    runs, n = [], 0
    for c in cond:
        if c:
            n += 1
        elif n:
            runs.append(n); n = 0
    if n:
        runs.append(n)
    return runs


def unit_year(D):
    rows = []
    for (u, y), g in D.groupby(["unit", D.date.dt.year]):
        g = g.sort_values("date")
        full = pd.date_range(f"{y}-01-01", f"{y}-12-31")
        p = g.set_index("date").power.reindex(full)
        ok = p.notna()
        pv = p[ok].values
        zero = episodes(pv, pv == 0)
        red = episodes(pv, (pv > 0) & (pv < 90))
        short = [r for r in zero if r <= 14]
        long_ = [r for r in zero if r > 14]
        rows.append({"unit": u, "year": y, "n_days": int(ok.sum()), "mean_power": pv.mean(),
                     "loss_total": 100 - pv.mean(),
                     "days_zero": int((pv == 0).sum()),
                     "n_shutdowns": len(zero), "n_short_shutdowns": len(short), "days_short_shutdowns": sum(short),
                     "n_long_shutdowns": len(long_), "days_long_shutdowns": sum(long_),
                     "longest_shutdown": max(zero) if zero else 0,
                     "n_reductions": len(red), "days_reduced": int(((pv > 0) & (pv < 90)).sum()),
                     "loss_partial": float(np.where(pv > 0, 100 - pv, 0).sum() / len(pv)),
                     "loss_short_zero": 100 * sum(short) / len(pv)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    D, ndup = load_daily()
    UY = unit_year(D)
    UY.to_csv(os.path.join(OUT, "US_unit_year_daily_descriptors.csv"), index=False)
    print("rows", len(D), "duplicates removed", ndup, "units", D.unit.nunique(), "unit-years", len(UY))
