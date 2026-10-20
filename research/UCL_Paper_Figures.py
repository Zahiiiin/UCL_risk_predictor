"""
UCL_Paper_Figures.py
==============================================================================
Draws the data figures of the paper (Figures 1 to 9) from the files in this repository.

WHAT THIS SCRIPT DOES NOT DO
    It trains no model and changes no number. Every value that is plotted is read from
        data/UCL_Dataset_v6_PRIS.xlsx                       (label, design columns, yearly PRIS values)
        research/expected_results/benchmark/                (BM_summary.csv, BM_confusion_matrices.json)
        research/expected_results/validation/               (UCF_book_vs_PRIS.csv, validation_summary.json)
        research/expected_results/label_stability/          (label_stability.json)
        research/expected_results/importance/               (PI_groups.csv)
    The two screenshots of the web application (Figures 10 and 11) are taken from the running
    application and are not made here.

HOW TO RUN (from the repository root)
    python research/UCL_Paper_Figures.py
    The pictures are written to research/paper_figures/ (PNG, 900 DPI, sized for a 6.5 inch text width).

STYLE (same for every figure)
    Times New Roman if it is installed (otherwise Liberation Serif, which has the same letter widths),
    no title inside the picture, no grid lines, a small legend that does not touch the data.
==============================================================================
"""
import json
import os
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP = os.path.join(HERE, "expected_results")
OUT = os.path.join(HERE, "paper_figures")
os.makedirs(OUT, exist_ok=True)
DPI = 900


def pick_font():
    have = {f.name for f in fm.fontManager.ttflist}
    for name in ("Times New Roman", "Liberation Serif", "Tinos", "DejaVu Serif"):
        if name in have:
            return name
    return "serif"


FONT = pick_font()
mpl.rcParams.update({
    "font.family": "serif", "font.serif": [FONT, "DejaVu Serif"], "font.size": 8,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "legend.frameon": True, "legend.edgecolor": "0.6", "legend.fancybox": False, "legend.framealpha": 1.0,
    "mathtext.fontset": "stix", "savefig.dpi": DPI, "figure.dpi": 150,
})

NAVY, STEEL, LIGHT = "#1F3556", "#4A7BA7", "#AFC5DA"
GREEN, AMBER, RED = "#2B5E3A", "#B07A1A", "#7A1B1B"
CLS_COL = {"Low": GREEN, "Medium": AMBER, "High": RED}
GREY = "0.4"
MODELS = ["LR", "RF", "ETC", "XGB", "Stacking"]


def save(fig, name):
    path = os.path.join(OUT, name + ".png")
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print("saved", path)


def panel_label(ax, text, dx=-0.02, dy=1.03):
    ax.text(dx, dy, text, transform=ax.transAxes, fontweight="bold", fontsize=9, ha="left", va="bottom")


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
XLSX = os.path.join(ROOT, "data", "UCL_Dataset_v6_PRIS.xlsx")
FINAL = pd.read_excel(XLSX, sheet_name="FINAL_DATASET")
YEARLY = pd.read_excel(XLSX, sheet_name="PRIS_YEARLY")
BM = pd.read_csv(os.path.join(EXP, "benchmark", "BM_summary.csv"))
BM["SetKey"] = BM["Set"].str[0]
CM = json.load(open(os.path.join(EXP, "benchmark", "BM_confusion_matrices.json")))["rows_are_true_cols_are_predicted"]
VAL = pd.read_csv(os.path.join(EXP, "validation", "UCF_book_vs_PRIS.csv"))
STAB = json.load(open(os.path.join(EXP, "label_stability", "label_stability.json")))
STAB_BLOCKS = pd.read_csv(os.path.join(HERE, "results_LabelStability", "label_stability_blocks.csv")) \
    if os.path.exists(os.path.join(HERE, "results_LabelStability", "label_stability_blocks.csv")) else None


def bm(setkey, model, metric="accuracy"):
    r = BM[(BM.SetKey == setkey) & (BM.Model == model)].iloc[0]
    return float(r[metric]), float(r[metric + "_sd"])


# ----------------------------------------------------------------------------------------------
# Fig 1  Methodology flowchart
# ----------------------------------------------------------------------------------------------
def fig1():
    """Study overview. Wide and low, so that the text stays readable at the width of the page (6.3 inches)."""
    W, H = 6.4, 4.15
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    FS = 7.2

    def box(x0, x1, y0, y1, text, bold=False, wrap=40, lw=0.8):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="white", ec="black", lw=1.2 if bold else lw))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, "\n".join(textwrap.fill(p, wrap) for p in text.split("\n")), ha="center", va="center",
                fontsize=FS, fontweight="bold" if bold else "normal", linespacing=1.15)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black", shrinkA=0, shrinkB=0, mutation_scale=6.5))

    L, R = 0.12, W - 0.12
    G = 0.24
    # row 1: source and usable-year rule
    y1b, y1t = H - 0.12 - 0.5, H - 0.12
    box(L, R, y1b, y1t, "IAEA PRIS annual values for every reactor and year (downloaded October 2026; 649 units, 1969 to 2025)\n"
        "Usable reactor-year: UCL published, reactor not completely shut down, not the year of first grid connection", wrap=120, bold=False)
    # row 2: label and two input groups
    y2t = y1b - G
    y2b = y2t - 0.78
    cw = (R - L - 2 * 0.16) / 3
    xs0 = [L, L + cw + 0.16, L + 2 * (cw + 0.16)]
    texts = ["Label (never an input)\nMean annual UCL 2018 to 2024, at least 3 usable years\nLow < 2 %,\nMedium 2 to 6 %, High > 6 %",
             "Design inputs (13)\nType, country, capacity, age, supplier, model family, construction time and others (Appendix A)",
             "History inputs (5)\nMean UCL 2010 to 2013 and 2014 to 2017, their mean, their change and the trend"]
    for x0, t in zip(xs0, texts):
        arrow(x0 + cw / 2, y1b, x0 + cw / 2, y2t)
        box(x0, x0 + cw, y2b, y2t, t, wrap=34)
    # row 3: dataset
    y3t = y2b - G
    y3b = y3t - 0.36
    for x0 in xs0:
        arrow(x0 + cw / 2, y2b, x0 + cw / 2, y3t)
    box(L, R, y3b, y3t, "Dataset: 387 reactors in 31 countries (Model 1); 324 of them with both history blocks (Model 2)", wrap=120, bold=True)
    # row 4: evaluation
    y4t = y3b - G
    y4b = y4t - 0.82
    wa, wb, wc, gap = 1.95, 2.35, 1.6, 0.16
    xa0 = L
    xb0 = xa0 + wa + gap
    xc0 = xb0 + wb + gap
    xc1 = R
    wc = xc1 - xc0
    arrow(xa0 + wa / 2, y3b, xa0 + wa / 2, y4t)
    box(xa0, xa0 + wa, y4b, y4t, "Repeated stratified 5-fold cross-validation, 5 repeats = 25 test folds\nInside each training fold only: country encoding, scaling, SMOTE (k\u00a0=\u00a03)", wrap=36)
    ymid = (y4b + y4t) / 2
    arrow(xa0 + wa, ymid, xb0, ymid)
    hh = (y4t - y4b - 0.08) / 2
    box(xb0, xb0 + wb, y4t - hh, y4t, "Baselines (no learning): majority class, country majority, persistence of past UCL", wrap=44)
    box(xb0, xb0 + wb, y4b, y4b + hh, "Classifiers (untuned): LR, RF, ETC, XGB, stacking", wrap=44)
    arrow(xb0 + wb, ymid, xc0, ymid)
    box(xc0, xc1, y4b, y4t, "Scores on the 25 test folds: accuracy, balanced accuracy, macro F1, per-class F1, AUC; corrected paired t-test", wrap=28)
    # row 5: further analyses and final models
    y5t = y4b - G
    y5b = max(0.1, y5t - 0.56)
    wl = 4.05
    arrow(xc0 + wc / 2, y4b, xc0 + wc / 2, y5t) if False else None
    arrow(xb0 + wb / 2, y4b, xb0 + wb / 2, y5t)
    box(L, L + wl, y5b, y5t, "Further analyses: PRIS against printed RDS-2, feature-set ablation, leave-one-country-out, permutation importance, label stability, sensitivity checks", wrap=82)
    arrow(xc0 + wc / 2, y4b, xc0 + wc / 2, y5t)
    box(L + wl + 0.16, R, y5b, y5t, "Final models trained on all reactors for the web application", wrap=30)
    assert y5b >= 0.08, "flowchart does not fit: raise H (bottom edge at %.2f)" % y5b
    save(fig, "Fig01_methodology_flowchart")


# ----------------------------------------------------------------------------------------------
# Fig 2  The label
# ----------------------------------------------------------------------------------------------
def fig2():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
    v = FINAL.UCL_Value.values
    bins = np.arange(0, 20.5, 0.5)
    counts, edges = np.histogram(np.clip(v, 0, 20), bins=bins)
    cols = [CLS_COL["Low"] if e < 2 else (CLS_COL["Medium"] if e < 6 else CLS_COL["High"]) for e in edges[:-1]]
    a.bar(edges[:-1], counts, width=0.5, align="edge", color=cols, edgecolor="white", linewidth=0.3)
    for x in (2, 6):
        a.axvline(x, color="black", ls="--", lw=0.7)
    n = {c: int((FINAL.UCL_Class == c).sum()) for c in ("Low", "Medium", "High")}
    ymax = counts.max()
    a.text(1.45, ymax * 0.93, "Low\n%d" % n["Low"], ha="center", va="top", fontsize=7.5, color=CLS_COL["Low"], fontweight="bold")
    a.text(4.0, ymax * 0.62, "Medium\n%d" % n["Medium"], ha="center", va="top", fontsize=7.5, color=CLS_COL["Medium"], fontweight="bold")
    a.text(11.5, ymax * 0.36, "High\n%d" % n["High"], ha="center", va="top", fontsize=7.5, color=CLS_COL["High"], fontweight="bold")
    a.set_xlim(0, 20)
    a.set_xlabel("Mean annual UCL, 2018 to 2024 (%)")
    a.set_ylabel("Number of reactors")
    n_over = int((v > 20).sum())
    a.text(19.6, ymax * 0.2, "%d reactors above 20 %%\nare drawn in the last bar" % n_over, ha="right", va="bottom", fontsize=6.5)
    panel_label(a, "(a)")

    order = ["PWR", "PHWR", "BWR", "GCR", "LWGR", "FBR"]
    ct = pd.crosstab(FINAL.Type, FINAL.UCL_Class).reindex(order)[["Low", "Medium", "High"]]
    share = ct.div(ct.sum(axis=1), axis=0) * 100
    y = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    for c in ("Low", "Medium", "High"):
        b.barh(y, share[c].values, left=left, color=CLS_COL[c], edgecolor="white", linewidth=0.4, height=0.68, label=c)
        left += share[c].values
    b.set_yticks(y)
    b.set_yticklabels(["%s (%d)" % (t, ct.loc[t].sum()) for t in order])
    b.set_xlim(0, 100)
    b.set_xlabel("Share of reactors of the type (%)")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False, handlelength=1.0, columnspacing=1.2)
    panel_label(b, "(b)", dx=-0.3)
    fig.tight_layout(w_pad=1.5)
    save(fig, "Fig02_label_distribution")


# ----------------------------------------------------------------------------------------------
# Fig 3  PRIS against the printed RDS-2 values
# ----------------------------------------------------------------------------------------------
def fig3():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.4))
    panels = [("Book_2013_2017", "PRIS_2013_2017", "RDS-2 2018 Table 14, UCF 2013 to 2017 (%)", "PRIS mean annual UCF, 2013 to 2017 (%)", "(a)"),
              ("Book_2014_2024", "PRIS_2014_2024", "RDS-2 2025 Table 14, UCF 2014 to 2024 (%)", "PRIS mean annual UCF, 2014 to 2024 (%)", "(b)"),
              ("Book_2014_2024", "PRIS_life_to_2024", "RDS-2 2025 Table 14, UCF 2014 to 2024 (%)", "PRIS mean annual UCF, whole life to 2024 (%)", "(c)")]
    for ax, (bk, pr, xl, yl, lab) in zip(axs, panels):
        d = VAL[[bk, pr]].dropna()
        ax.scatter(d[bk], d[pr], s=5, color=STEEL, alpha=0.65, linewidths=0)
        ax.plot([0, 100], [0, 100], color="black", lw=0.6)
        lo = 40 if bk != "Book_2013_2017" else 0
        ax.set_xlim(lo, 100)
        ax.set_ylim(lo, 100)
        ax.set_xlabel(xl, fontsize=6.8)
        ax.set_ylabel(yl, fontsize=6.8)
        e = (d[bk] - d[pr]).abs()
        ax.text(0.04, 0.96, "n = %d\nmean |diff| = %.2f\nmedian |diff| = %.2f" % (len(d), e.mean(), e.median()),
                transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
                bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none"))
        ax.set_aspect("equal", adjustable="box")
        panel_label(ax, lab, dx=-0.18)
    fig.tight_layout(w_pad=0.8)
    save(fig, "Fig03_PRIS_vs_RDS2")


# ----------------------------------------------------------------------------------------------
# Fig 9  Label stability
# ----------------------------------------------------------------------------------------------
def fig4():
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.1))
    yearly = YEARLY[YEARLY.Used == "yes"]
    lab = FINAL.set_index("Reactor_Name").UCL_Value

    def block(a, b):
        s = yearly[(yearly.Year >= a) & (yearly.Year <= b)].groupby("Reactor_Name").UCL.agg(["mean", "count"])
        return s[s["count"] >= 2]["mean"]

    early, late, h2 = block(2018, 2021), block(2022, 2024), block(2014, 2017)
    ticks = [0, 1, 2, 4, 6, 10, 20, 40]
    for ax, (x, y_, xl, yl, key, lab_) in zip(axs, [
            (early, late, "Mean UCL 2018 to 2021 (%)", "Mean UCL 2022 to 2024 (%)", "early_vs_late", "(a)"),
            (h2, lab, "Mean UCL 2014 to 2017 (%)", "Label: mean UCL 2018 to 2024 (%)", "history_2014_2017_vs_label", "(b)")]):
        d = pd.concat([x.rename("x"), y_.rename("y")], axis=1).dropna()
        ax.scatter(d.x, d.y, s=5, color=STEEL, alpha=0.6, linewidths=0)
        for v in (2, 6):
            ax.axvline(v, color="black", ls="--", lw=0.6)
            ax.axhline(v, color="black", ls="--", lw=0.6)
        ax.plot([0, 45], [0, 45], color=GREY, lw=0.5)
        for axis in (ax.xaxis, ax.yaxis):
            pass
        ax.set_xscale("symlog", linthresh=1, linscale=0.8)
        ax.set_yscale("symlog", linthresh=1, linscale=0.8)
        ax.set_xlim(0, 45)
        ax.set_ylim(0, 45)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_yticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        r = STAB[key]
        ax.set_title("n = %d reactors; same class: %.0f %%; Low/High swaps: %d" % (r["n_reactors"], r["class_agreement_pct"], r["low_high_swaps"]),
                     fontsize=6.8, pad=4)
        ax.set_aspect("equal", adjustable="box")
        panel_label(ax, lab_, dx=-0.16, dy=1.08)
    fig.tight_layout(w_pad=1.0)
    save(fig, "Fig09_label_stability")


# ----------------------------------------------------------------------------------------------
# Fig 7  Fleet-mean annual UCL by country group
# ----------------------------------------------------------------------------------------------
def fig5():
    yearly = YEARLY[YEARLY.Used == "yes"].merge(FINAL[["Reactor_Name", "Country"]], on="Reactor_Name")
    groups = [("FRANCE", "France", RED, "-"), ("USA", "United States", NAVY, "-"), ("CHINA", "China", GREEN, "-")]
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for key, name, col, ls in groups:
        s = yearly[yearly.Country == key].groupby("Year").UCL.mean()
        ax.plot(s.index, s.values, color=col, lw=1.2, ls=ls, marker="o", ms=2.5, label="%s (%d reactors)" % (name, (FINAL.Country == key).sum()))
    rest = yearly[~yearly.Country.isin([g[0] for g in groups])].groupby("Year").UCL.mean()
    ax.plot(rest.index, rest.values, color=GREY, lw=1.2, ls="--", marker="s", ms=2.5,
            label="All other countries (%d reactors)" % (~FINAL.Country.isin([g[0] for g in groups])).sum())
    for v in (2, 6):
        ax.axhline(v, color="black", ls=":", lw=0.5)
    ax.axvspan(2017.5, 2024.5, color="0.93", zorder=0)
    ax.set_ylim(-0.4, 17.2)
    ax.text(2021, 16.9, "label window", ha="center", va="top", fontsize=6.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean annual UCL of the reactors (%)")
    ax.set_xlim(2010, 2024.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.3, frameon=False, columnspacing=1.0, handlelength=2.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "Fig07_country_UCL_by_year.png"), dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("saved Fig07")


# ----------------------------------------------------------------------------------------------
# Fig 4  Benchmark: accuracy of baselines and classifiers
# ----------------------------------------------------------------------------------------------
def fig6():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 3.0), sharex=True)
    # panel (a): 387 reactors, design features only
    rows_a = [("Majority class", ("A", "Baseline: majority class"), None), ("Country majority", ("A", "Baseline: country majority"), None)] + \
             [(m, ("A", m), None) for m in MODELS]
    for ax, rows, lab_, n_ in ((a, rows_a, "(a)", 387),):
        y = np.arange(len(rows))[::-1]
        for yi, (name, (sk, mdl), _) in zip(y, rows):
            m, s = bm(sk, mdl)
            base = mdl.startswith("Baseline")
            ax.errorbar(m, yi, xerr=s, fmt="o" if not base else "s", ms=4.5, color=GREY if base else NAVY, ecolor=GREY if base else NAVY,
                        elinewidth=0.7, capsize=2, capthick=0.7, zorder=3)
            ax.text(m, yi + 0.33, "%.1f" % m, ha="center", va="bottom", fontsize=6.3)
        cm_ = bm("A", "Baseline: country majority")[0]
        ax.axvline(cm_, color=GREY, ls="--", lw=0.7, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_xlabel("Accuracy over 25 test folds (%)")
        ax.set_ylim(-0.7, len(rows) - 0.2)
        panel_label(ax, lab_, dx=-0.35)
        ax.text(0.02, 1.0, "Design inputs only, %d reactors" % n_, transform=ax.transAxes, fontsize=6.8, va="bottom")
    # panel (b): 324 reactors, design vs design + history
    rows_b = [("Majority class", "Baseline: majority class"), ("Country majority", "Baseline: country majority"),
              ("Persistence of past UCL", "Baseline: persistence (past UCL)")] + [(m, m) for m in MODELS]
    y = np.arange(len(rows_b))[::-1]
    for yi, (name, mdl) in zip(y, rows_b):
        if mdl.startswith("Baseline"):
            m, s = bm("B", mdl)
            b.errorbar(m, yi, xerr=s, fmt="s", ms=4.5, color=GREY, ecolor=GREY, elinewidth=0.7, capsize=2, capthick=0.7, zorder=3)
            b.text(m, yi + 0.33, "%.1f" % m, ha="center", va="bottom", fontsize=6.3)
        else:
            m1, s1 = bm("B", mdl)
            m2, s2 = bm("D", mdl)
            b.errorbar(m1, yi + 0.12, xerr=s1, fmt="o", ms=4.5, mfc="white", color=STEEL, ecolor=STEEL, elinewidth=0.7, capsize=2, capthick=0.7, zorder=3)
            b.errorbar(m2, yi - 0.12, xerr=s2, fmt="o", ms=4.5, color=NAVY, ecolor=NAVY, elinewidth=0.7, capsize=2, capthick=0.7, zorder=3)
    cm_ = bm("B", "Baseline: country majority")[0]
    b.axvline(cm_, color=GREY, ls="--", lw=0.7, zorder=1)
    b.set_yticks(y)
    b.set_yticklabels([r[0] for r in rows_b])
    b.set_xlabel("Accuracy over 25 test folds (%)")
    b.set_ylim(-0.7, len(rows_b) - 0.2)
    fig.legend(handles=[Line2D([], [], marker="o", mfc="white", color=STEEL, ls="", ms=4.5, label="Model 1 (design inputs, 13)"),
                        Line2D([], [], marker="o", color=NAVY, ls="", ms=4.5, label="Model 2 (design + history inputs, 18)"),
                        Line2D([], [], marker="s", color=GREY, ls="", ms=4.5, label="Baseline"),
                        Line2D([], [], color=GREY, ls="--", lw=0.8, label="Country majority baseline")],
               loc="lower center", ncol=4, fontsize=6.5, frameon=False, columnspacing=1.4, handletextpad=0.3)
    b.text(0.02, 1.0, "324 reactors with history", transform=b.transAxes, fontsize=6.8, va="bottom")
    panel_label(b, "(b)", dx=-0.35)
    a.set_xlim(48, 80)
    fig.tight_layout(w_pad=1.2, rect=(0, 0.07, 1, 1))
    save(fig, "Fig04_benchmark_accuracy")


# ----------------------------------------------------------------------------------------------
# Fig 5  Feature sets
# ----------------------------------------------------------------------------------------------
def fig7():
    sets = [("B", "Design (13)"), ("C", "Design without country (12)"), ("E", "History only (5)"),
            ("F", "Design + UCL 2010-2013 (14)"), ("G", "Design + UCL 2014-2017 (14)"),
            ("H", "Design + both UCL blocks (15)"), ("D", "Model 2: design + history (18)"),
            ("I", "Model 2 without country (17)")]
    fig, ax = plt.subplots(figsize=(6.0, 3.1))
    y = np.arange(len(sets))[::-1]
    off = {"LR": 0.22, "RF": 0.0, "ETC": -0.22}
    sty = {"LR": ("^", "#B07A1A"), "RF": ("o", NAVY), "ETC": ("s", STEEL)}
    for yi, (sk, name) in zip(y, sets):
        for mdl in ("LR", "RF", "ETC"):
            m, s = bm(sk, mdl)
            ax.errorbar(m, yi + off[mdl], xerr=s, fmt=sty[mdl][0], ms=4, color=sty[mdl][1], ecolor=sty[mdl][1],
                        elinewidth=0.6, capsize=1.6, capthick=0.6, zorder=3)
    cm_ = bm("B", "Baseline: country majority")[0]
    pm_ = bm("B", "Baseline: persistence (past UCL)")[0]
    ax.axvline(cm_, color=GREY, ls="--", lw=0.7, zorder=1)
    ax.axvline(pm_, color=GREY, ls=":", lw=0.9, zorder=1)
    ax.text(cm_ + 0.2, len(sets) - 0.1, "country majority %.1f" % cm_, fontsize=6.3, va="bottom", ha="left")
    ax.text(pm_ - 0.2, len(sets) - 0.1, "persistence %.1f" % pm_, fontsize=6.3, va="bottom", ha="right")
    ax.set_yticks(y)
    ax.set_yticklabels([s[1] for s in sets])
    ax.set_xlabel("Accuracy over 25 test folds (%), 324 reactors")
    ax.set_ylim(-0.7, len(sets) + 0.15)
    ax.legend(handles=[Line2D([], [], marker=sty[m][0], color=sty[m][1], ls="", ms=4, label=n) for m, n in
                       (("LR", "Logistic regression"), ("RF", "Random forest"), ("ETC", "Extra trees"))],
              loc="lower left", fontsize=6.5)
    fig.tight_layout()
    save(fig, "Fig05_feature_sets")


# ----------------------------------------------------------------------------------------------
# Fig 6  Confusion matrices
# ----------------------------------------------------------------------------------------------
def fig8():
    items = [("Country majority (baseline)", "B. Design (13), 324 Model 2 reactors | Baseline: country majority"),
             ("Model 1: random forest, design (13)", "B. Design (13), 324 Model 2 reactors | RF"),
             ("Model 2: random forest, design + history (18)", "D. Model 2: design + history (18), 324 reactors | RF")]
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.5))
    cls = ["Low", "Medium", "High"]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("blu", ["#FFFFFF", "#4A7BA7"])
    for ax, (title, key), lab in zip(axs, items, ("(a)", "(b)", "(c)")):
        cm = np.array(CM[key])
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, "%.0f %%\n(%d)" % (pct[i, j], cm[i, j]), ha="center", va="center", fontsize=6.4,
                        color="white" if pct[i, j] > 60 else "black")
        ax.set_xticks(range(3))
        ax.set_xticklabels(cls)
        ax.set_yticks(range(3))
        ax.set_yticklabels(cls)
        ax.set_xlabel("Predicted class")
        if lab == "(a)":
            ax.set_ylabel("True class")
        ax.set_title(title, fontsize=6.6, pad=3)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.6)
        panel_label(ax, lab, dx=-0.16, dy=1.12)
    fig.tight_layout(w_pad=0.8)
    save(fig, "Fig06_confusion_matrices")


# ----------------------------------------------------------------------------------------------
# Fig 8  Permutation importance
# ----------------------------------------------------------------------------------------------
def fig9():
    path = os.path.join(EXP, "importance", "PI_groups.csv")
    if not os.path.exists(path):
        print("PI_groups.csv not found, Fig 8 skipped")
        return
    pi = pd.read_csv(path)
    short = {
        "Country": "Country",
        "Age (age, age squared, age group)": "Age (3 inputs)",
        "Reactor type, family and supplier": "Type, family, supplier",
        "Size (net capacity, capacity category)": "Size (2 inputs)",
        "Build (construction duration, operational years and category)": "Build (4 inputs)",
        "Past UCL (all five history inputs)": "Past UCL (5 inputs)",
    }
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.6))
    for ax, (mdl, lab) in zip(axs, (("Model 1 (design, 13)", "(a)"), ("Model 2 (design + history, 18)", "(b)"))):
        d = pi[pi.Model == mdl].sort_values("mean", ascending=True)
        hist = d.Group.str.startswith("Past")
        ax.barh(range(len(d)), d["mean"], xerr=d["std"], color=[NAVY if h else LIGHT for h in hist], edgecolor="black", linewidth=0.4,
                error_kw=dict(elinewidth=0.6, capsize=1.6, capthick=0.6, ecolor="0.3"))
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([short[g] for g in d.Group], fontsize=7)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlabel("Fall in accuracy (points)")
        ax.set_title(mdl, fontsize=7, pad=3)
        panel_label(ax, lab, dx=-0.62, dy=1.1)
    axs[1].legend(handles=[Patch(fc=NAVY, ec="black", lw=0.4, label="History input"), Patch(fc=LIGHT, ec="black", lw=0.4, label="Design input")],
                  loc="lower right", fontsize=6.5)
    fig.tight_layout(w_pad=1.6)
    save(fig, "Fig08_permutation_importance")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9):
        f()
