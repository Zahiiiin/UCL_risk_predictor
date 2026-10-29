"""
UCL_Paper_Figures_V6.py
==============================================================================
Draws the NEW figures of the combined paper (overview, Part II forecasting, tuning check).
The Part I figures (label, PRIS against RDS-2, benchmark, feature sets, confusion matrices, country, importance, stability)
are the ones made by UCL_Paper_Figures.py and are re-used unchanged.

WHAT THIS SCRIPT DOES NOT DO
    It trains no model and changes no number. Every value that is plotted is read from
        research/expected_results/forecast/      (Part II results)
        research/expected_results/tuning/        (tuning check, when present)
        data/UCL_Dataset_v6_PRIS.xlsx            (counts in the overview)

HOW TO RUN (from the repository root)
    python3 research/UCL_Paper_Figures_V6.py            # all figures that have their inputs
Pictures go to research/paper_figures/ (PNG, 900 DPI, sized for a 6.5 inch text width), same style as the other figures.
==============================================================================
"""
import json
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import UCL_Paper_Figures as PF
from UCL_Paper_Figures import AMBER, GREEN, GREY, LIGHT, NAVY, RED, STEEL, panel_label, save  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.join(HERE, "expected_results", "forecast")
TU = os.path.join(HERE, "expected_results", "tuning")
PER = "2010-2025"

COL = {"base:persistence": "0.55", "base:median5": AMBER, "ml:RF|C": NAVY, "ml:XGB|C": STEEL, "ml:Ridge|C": GREEN,
       "ml:RF|A": NAVY, "ml:XGB|A": STEEL, "ml:Ridge|A": GREEN}
NAME = {"base:persistence": "Persistence", "base:median5": "Median of last 5 years", "ml:RF|C": "Random forest (set C)",
        "ml:XGB|C": "XGBoost (set C)", "ml:Ridge|C": "Ridge (set C)"}


def rd(tag, f):
    return pd.read_csv(os.path.join(FC, tag, f))


# ----------------------------------------------------------------------------------------------
# Fig 1  Overview of the two parts
# ----------------------------------------------------------------------------------------------
def fig_overview():
    ps = json.load(open(os.path.join(FC, "panel_summary.json")))
    hist = pd.read_excel(PF.XLSX, sheet_name="HISTORY")
    n_all, n_sub = len(PF.FINAL), int((hist.Model2_Subset == 1).sum())
    W, H = 6.5, 4.70
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    FS = 7.0

    def box(x0, x1, y0, y1, text, bold_first=False, wrap=40, fc="white", lw=0.8, bold=False):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fc, ec="black", lw=1.2 if bold else lw))
        lines = ["\n".join(textwrap.wrap(p, wrap)) for p in text.split("\n")]
        if bold_first:
            first, rest = lines[0], "\n".join(lines[1:])
            n1, n2 = first.count("\n") + 1, rest.count("\n") + 1 if rest else 0
            lh = FS * 1.2 / 72.0
            tot = (n1 + n2) * lh
            ytop = (y0 + y1) / 2 + tot / 2
            ax.text((x0 + x1) / 2, ytop, first, ha="center", va="top", fontsize=FS + 0.8, fontweight="bold", linespacing=1.15)
            if rest:
                ax.text((x0 + x1) / 2, ytop - n1 * lh * 1.04 - 0.03, rest, ha="center", va="top", fontsize=FS, linespacing=1.15)
        else:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, "\n".join(lines), ha="center", va="center", fontsize=FS,
                    fontweight="bold" if bold else "normal", linespacing=1.15)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black", shrinkA=0, shrinkB=0, mutation_scale=6.5))

    L, R, G = 0.12, W - 0.12, 0.13
    mid = W / 2
    gap = 0.14
    xL0, xL1 = L, mid - gap / 2
    xR0, xR1 = mid + gap / 2, R
    cL, cR = (xL0 + xL1) / 2, (xR0 + xR1) / 2
    t1 = H - 0.08
    t0 = t1 - 0.5
    box(L, R, t0, t1, "IAEA PRIS: UCL, UCF, planned and external losses of every reactor and year (October 2026 download)\n"
        "IAEA RDS-2, 2025 edition: country, type, capacity, grid date, supplier and model of each operating reactor", wrap=118)
    d1 = t0 - G
    d0 = d1 - 0.95
    arrow(cL, t0, cL, d1)
    arrow(cR, t0, cR, d1)
    box(xL0, xL1, d0, d1, f"PART I: which UCL class does a reactor fall in?\n{n_all} reactors in {PF.FINAL.Country.nunique()} countries operating at the end of 2024. "
        f"Label: mean UCL 2018 to 2024 in three classes (below 2%, 2 to 6%, above 6%); {n_sub} reactors also have history from 2010 to 2017", bold_first=True, wrap=50, fc="#E7EEF5")
    box(xR0, xR1, d0, d1, f"PART II: what will a reactor's UCL be next year?\n{ps['rows']:,} reactor-years from {ps['units']} reactors in {ps['countries']} countries, "
        "including reactors since closed. Target: UCL of the next year; inputs: only what was known at the end of the year", bold_first=True, wrap=50, fc="#E7EEF5")
    m1 = d0 - G
    m0 = m1 - 0.9
    arrow(cL, d0, cL, m1)
    arrow(cR, d0, cR, m1)
    box(xL0, xL1, m0, m1, "Model 1: 13 design inputs. Model 2: plus 5 inputs from past UCL\n"
        "Logistic regression, random forest, extra trees, XGBoost, stacking\n"
        "Repeated 5-fold cross-validation (25 test folds); scaling, country encoding and oversampling inside each training fold", wrap=52)
    box(xR0, xR1, m0, m1, "Feature sets A to E: own UCL history, planned losses, world and country context, age and size, design\n"
        "Ridge, random forest, XGBoost\n"
        "Rolling origin: for each year 2010 to 2025, learn from earlier years only, then forecast that year", wrap=52)
    c1 = m0 - G
    c0 = c1 - 0.72
    arrow(cL, m0, cL, c1)
    arrow(cR, m0, cR, c1)
    box(xL0, xL1, c0, c1, "Compared with: majority class, country majority, persistence of past UCL\n"
        "Also: leave-one-country-out, permutation importance, label stability, class limits, PRIS against printed tables", wrap=52)
    box(xR0, xR1, c0, c1, "Compared with: last year's value, mean and median of the last 5 years, country and world average\n"
        "Also: training windows, new sites and new countries, design facts, error by year", wrap=52)
    b1 = c0 - G
    b0 = b1 - 0.46
    arrow(cL, c0, cL, b1)
    arrow(cR, c0, cR, b1)
    box(L, R, b0, b1, "Robustness check, specified before it was run: the settings of each model searched on training data only (nested), "
        "then compared with the fixed settings of the main analysis", wrap=118, fc="#F4F1E6")
    q1 = b0 - G
    q0 = q1 - 0.36
    arrow(mid, b0, mid, q1)
    box(L, R, q0, q1, "One question: how much of a reactor's unplanned capability loss can public IAEA data predict beyond simple rules?", bold=True, wrap=118)
    assert q0 >= 0.06, "overview does not fit: raise H (bottom edge at %.2f)" % q0
    save(fig, "V6_Fig01_overview")


# ----------------------------------------------------------------------------------------------
# Part II figures
# ----------------------------------------------------------------------------------------------
def fig_by_year():
    y = rd("main_window15", "FB_by_year.csv")
    mae = y.pivot(index="target_year", columns="Method", values="MAE")
    yrs = mae.index.values
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.75), gridspec_kw={"width_ratios": [1.35, 1]})
    for m, ls, mk in (("base:persistence", "--", "s"), ("base:median5", "-.", "^"), ("ml:RF|C", "-", "o")):
        a.plot(yrs, mae[m].values, ls=ls, marker=mk, ms=3, lw=1.1, color=COL[m], label=NAME[m])
    a.set_xlabel("Target year")
    a.set_ylabel("Mean absolute error (UCL points)")
    a.set_xticks(yrs[::3])
    a.legend(loc="upper left", frameon=False, fontsize=6.5)
    panel_label(a, "(a)")
    d = (mae["ml:RF|C"] - mae["base:median5"]).values
    d2 = (mae["ml:RF|C"] - mae["base:persistence"]).values
    w = 0.38
    b.bar(yrs - w / 2, d2, width=w, color="0.65", label="vs persistence")
    b.bar(yrs + w / 2, d, width=w, color=AMBER, label="vs median of last 5")
    b.axhline(0, color="black", lw=0.6)
    b.set_xlabel("Target year")
    b.set_ylabel("Change in MAE, random forest (points)")
    b.set_xticks(yrs[::3])
    b.legend(loc="lower right", frameon=False, fontsize=6.5)
    panel_label(b, "(b)")
    fig.tight_layout(w_pad=1.6)
    save(fig, "V6_Fig10_forecast_by_year")


def fig_intervals():
    rows = []
    for tag, lab in (("main_expanding", "All earlier years"), ("main_window15", "Latest 15 years"), ("main_window10", "Latest 10 years")):
        B = rd(tag, "FB_bootstrap.csv")
        B = B[B.Period == PER]
        for m in ("ml:RF|C", "ml:XGB|C", "ml:Ridge|C"):
            r_med = B[(B.Method == m) & (B.Reference == "base:median5")].iloc[0]
            r_per = B[(B.Method == m) & (B.Reference == "base:persistence")].iloc[0]
            rows.append((lab, m, r_per, r_med))
    n = len(rows)
    ypos = np.array([10 - ((i // 3) * 4 + i % 3) for i in range(n)], dtype=float)   # three groups of three rows, one blank line between groups
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 3.25), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1]})
    titles = ["MAE, minus persistence", "MAE, minus median of last 5", "Class accuracy, minus median of last 5"]
    for k, ax in enumerate(axs):
        for yv, (lab, m, rp, rm) in zip(ypos, rows):
            if k == 0:
                d, lo, hi = rp.dMAE, rp.dMAE_lo, rp.dMAE_hi
            elif k == 1:
                d, lo, hi = rm.dMAE, rm.dMAE_lo, rm.dMAE_hi
            else:
                d, lo, hi = rm.dClassAcc, rm.dClassAcc_lo, rm.dClassAcc_hi
            ax.plot([lo, hi], [yv, yv], color=COL[m], lw=1.6, solid_capstyle="butt")
            ax.plot([d], [yv], marker="o", ms=3.6, color=COL[m], mec="white", mew=0.4)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(titles[k], fontsize=6.6, pad=14)
        ax.set_xlabel("difference in MAE (UCL points)" if k < 2 else "difference (percentage points)", fontsize=6.8)
        ax.text(0.0, 1.13, "(%s)" % "abc"[k], transform=ax.transAxes, ha="left", va="bottom", fontsize=9, fontweight="bold")
    labels = [NAME[m].replace(" (set C)", "") for (_, m, _, _) in rows]
    axs[0].set_yticks(ypos)
    axs[0].set_yticklabels(labels, fontsize=6.5)
    for g, lab in enumerate(("All earlier years", "Latest 15 years", "Latest 10 years")):
        yy = ypos[g * 3:(g + 1) * 3]
        axs[0].text(-0.62, yy.mean(), lab, transform=axs[0].get_yaxis_transform(), rotation=90, ha="center", va="center", fontsize=6.8,
                    fontweight="bold")
    axs[0].spines["left"].set_visible(True)
    fig.tight_layout(w_pad=1.0)
    fig.subplots_adjust(left=0.205)
    save(fig, "V6_Fig11_forecast_intervals")


def fig_new_places():
    tests = [("main_window15", "Known countries\nand sites"), ("grouped-site_window15", "New site"), ("grouped-country_window15", "New country")]
    methods = ["ml:Ridge|A", "ml:Ridge|C", "ml:RF|A", "ml:RF|C", "ml:XGB|A", "ml:XGB|C"]
    lab = {"ml:Ridge|A": "Ridge A", "ml:Ridge|C": "Ridge C", "ml:RF|A": "RF A", "ml:RF|C": "RF C", "ml:XGB|A": "XGB A", "ml:XGB|C": "XGB C"}
    S = {t: rd(t, "FB_summary.csv") for t, _ in tests}
    S = {t: s[s.Period == PER].set_index("Method") for t, s in S.items()}
    pers = S["main_window15"].loc["base:persistence", "MAE"]
    med = S["main_window15"].loc["base:median5", "MAE"]
    fig, ax = plt.subplots(figsize=(6.5, 2.95))
    w = 0.13
    x = np.arange(len(tests))
    for i, m in enumerate(methods):
        vals = [S[t].loc[m, "MAE"] for t, _ in tests]
        hatch = "" if m.endswith("A") else "////"
        ax.bar(x + (i - 2.5) * w, vals, width=w, color=COL[m], edgecolor="white" if not hatch else COL[m], hatch=hatch, lw=0.4,
               label=lab[m], alpha=1.0 if not hatch else 0.55)
        for xi, v in zip(x + (i - 2.5) * w, vals):
            ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=5.3, rotation=90, zorder=5,
                    bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"))
    ax.axhline(pers, color="0.35", ls="--", lw=0.9)
    ax.axhline(med, color=AMBER, ls="-.", lw=0.9)
    ax.plot([], [], color="0.35", ls="--", lw=0.9, label=f"Persistence {pers:.2f}")
    ax.plot([], [], color=AMBER, ls="-.", lw=0.9, label=f"Median of last 5 {med:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([t[1] for t in tests])
    ax.set_ylabel("Mean absolute error (UCL points)")
    ax.set_ylim(3.0, 4.95)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.24), frameon=False, fontsize=6.3, handlelength=1.4, columnspacing=1.0)
    fig.tight_layout()
    save(fig, "V6_Fig12_forecast_new_places")



# ----------------------------------------------------------------------------------------------
# Tuning check figures
# ----------------------------------------------------------------------------------------------
def fig_tuning_part1():
    un = pd.read_csv(os.path.join(HERE, "expected_results", "benchmark", "BM_summary.csv"))
    tu = pd.read_csv(os.path.join(TU, "part1", "BT_summary.csv"))
    models = ["LR", "RF", "ETC", "XGB", "Stacking"]
    panels = [("A", "Model 1, all reactors", ["Baseline: country majority"]), ("D", "Model 2, 324 reactors",
              ["Baseline: persistence (past UCL)", "Baseline: country majority"])]
    # baselines of set D come from the untuned file of set B (same reactors, same folds); the persistence rule needs history inputs only
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.7), sharey=True)
    for k, (ax, (sl, title, bl)) in enumerate(zip(axs, panels)):
        for i, m in enumerate(models):
            u = float(un[un.Set.str.startswith(sl + ".") & (un.Model == m)].accuracy.iloc[0])
            t = float(tu[tu.Set.str.startswith(sl + ".") & (tu.Model == m + "-tuned")].accuracy.iloc[0])
            ax.plot([u, t], [i, i], color="0.6", lw=1.0, zorder=1)
            ax.plot([u], [i], marker="o", ms=4.5, mfc="white", mec=NAVY, mew=1.0, ls="", zorder=3, label="Fixed settings" if i == 0 else None)
            ax.plot([t], [i], marker="o", ms=4.5, color=NAVY, ls="", zorder=3, label="Tuned settings" if i == 0 else None)
        for j, key in enumerate(bl):
            bs = "B" if sl == "D" else sl
            v = float(un[un.Set.str.startswith(bs + ".") & (un.Model == key)].accuracy.iloc[0])
            ax.axvline(v, color=AMBER if "country" in key else "0.35", ls="-." if "country" in key else "--", lw=0.9)
            ax.text(v + (-0.08 if "country" in key else 0.08), -0.95, ("country rule " if "country" in key else "persistence ") + f"{v:.1f}", rotation=0,
                    ha="right" if "country" in key else "left", va="center", fontsize=5.8, color=AMBER if "country" in key else "0.35")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Accuracy over 25 test folds (%)", fontsize=7)
        ax.set_title(title, fontsize=7.2, pad=12)
        panel_label(ax, "(%s)" % "ab"[k])
        ax.set_ylim(4.5, -1.25)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(w_pad=1.2, rect=(0, 0.07, 1, 1))
    save(fig, "V6_Fig13_tuning_part1")


def fig_tuning_part2():
    wins = [("window15", "main_window15", "Latest 15 years"), ("window10", "main_window10", "Latest 10 years"), ("expanding", "main_expanding", "All earlier years")]
    combos = [("RF", "A"), ("RF", "C"), ("XGB", "A"), ("XGB", "C")]
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.6), sharey=True)
    allv = []
    for k, (ax, (tt, ut, title)) in enumerate(zip(axs, wins)):
        su = rd(ut, "FB_summary.csv")
        su = su[su.Period == PER].set_index("Method")
        st = pd.read_csv(os.path.join(TU, "part2", tt, "FT_summary.csv"))
        st = st[st.Period == PER].set_index("Method")
        x = np.arange(len(combos))
        vu = [su.loc[f"ml:{m}|{s}", "MAE"] for m, s in combos]
        vt = [st.loc[f"ml:{m}-tuned|{s}", "MAE"] for m, s in combos]
        allv += vu + vt + [su.loc["base:median5", "MAE"]]
        # dots, not bars: the axis does not start at zero, so a bar length would mislead
        for xi, a, c in zip(x, vu, vt):
            ax.plot([xi, xi], [a, c], color="0.6", lw=1.0, zorder=1)
        ax.plot(x, vu, marker="o", ms=4.5, mfc="white", mec=NAVY, mew=1.0, ls="", zorder=3, label="Fixed settings")
        ax.plot(x, vt, marker="o", ms=4.5, color=NAVY, ls="", zorder=3, label="Tuned settings")
        med = su.loc["base:median5", "MAE"]
        ax.axhline(med, color=AMBER, ls="-.", lw=0.9, label=f"Median of last 5 years ({med:.2f})")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{m}\n{s}" for m, s in combos], fontsize=6.5)
        ax.set_xlim(-0.5, len(combos) - 0.5)
        ax.set_title(title, fontsize=7.2, pad=12)
        panel_label(ax, "(%s)" % "abc"[k])
    lo, hi = min(allv), max(allv)
    pad = 0.12 * (hi - lo)
    axs[0].set_ylim(lo - pad, hi + 3 * pad)
    axs[0].set_ylabel("Mean absolute error (UCL points)", fontsize=7)
    axs[0].legend(loc="upper left", frameon=False, fontsize=5.8)
    fig.tight_layout(w_pad=0.8)
    save(fig, "V6_Fig14_tuning_part2")

# ----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    fig_overview()
    fig_by_year()
    fig_intervals()
    fig_new_places()
    if os.path.exists(os.path.join(TU, "part1", "BT_summary.csv")):
        fig_tuning_part1()
    if all(os.path.exists(os.path.join(TU, "part2", w, "FT_summary.csv")) for w in ("window15", "window10", "expanding")):
        fig_tuning_part2()
