"""
UCL_Model2_Pipeline.py
==============================================================================
Model 2: design features + UCL history from 2010-2017 (18 features)
Paper: "Machine Learning Classification of Unplanned Capability Loss Risk in
        Nuclear Power Plants" (target journal: Nuclear Engineering and Design)
Data : IAEA PRIS / RDS-2, file UCL_Dataset_v6_PRIS.xlsx
==============================================================================

WHAT THIS SCRIPT DOES
    Predicts the UCL risk class (Low / Medium / High) of a reactor, where the
    label is the mean PRIS unplanned capability loss in 2018-2024, from
        13 design and identity features (the same ones as Model 1), plus
         5 history features built from the reactor's own PRIS record before the label window:
            UCL 2010-2013   (period 1, mean annual UCL)
            UCL 2014-2017   (period 2, mean annual UCL)
            change between the two, their mean, and a trend direction
            (+1 UCL rose by more than 2 points, -1 fell by more than 2, 0 otherwise)

    Both history periods end before 2018, so the history never overlaps the
    label window. The label itself (UCL_Value, UCL_Class, UCL_Years_Used) is
    NEVER an input. The script stops with an error if one of those columns ever
    sneaks into the feature list.

HOW TO RUN
    pip install -r requirements-research.txt
    python UCL_Model2_Pipeline.py
    python UCL_Model2_Pipeline.py --data path/to/UCL_Dataset_v6_PRIS.xlsx
    python UCL_Model2_Pipeline.py --quick        (fast smoke test, numbers differ)

    Full run takes roughly 10 to 30 minutes on a normal laptop (the random
    searches are the slow part). Results go to the folder results_Model2/.

NUMBERS YOU SHOULD SEE
    See expected_results/model2/run_summary.json for the reference run. The
    numbers match when the script runs with the library versions pinned in
    requirements-research.txt. The versions used for the reference run are
    written to results_Model2/run_summary.json.

THINGS A REVIEWER SHOULD KNOW (also listed in the repository README)
    1. Thermal_Efficiency is the constant 33.0 for every reactor, because the
       earlier release of the dataset had no thermal power column. It therefore
       carries no information. Run with --pris-thermal to use the
       Thermal_Efficiency_Pct column instead.
    2. Scaling and the Country encoding are computed on all reactors before the
       train/test split (as in the paper). The encoding is out-of-fold, and trees
       ignore scaling, but a stricter design would fit both on the training part only.
    3. SMOTE is applied to the training part only. The cross-validation used
       inside the random search runs on that balanced training set, so the CV
       scores are optimistic. Test-set scores are not affected.
    4. Model 2 only includes reactors with at least two usable PRIS years in each
       history period (column Model2_Subset of the HISTORY sheet). Reactors that
       were shut down in 2010-2017, such as most Japanese units, drop out.
       UCL_Model2_History_Ablation.py measures how much each period contributes.
    5. Model 1 is validated with a temporal split by operational category. That split
       is impossible for Model 2, because every reactor in the Model 2 subset began
       operating by 2011. Model 2 therefore uses a cohort split instead: train on reactors
       first connected to the grid in 1990 or earlier, test on later ones (cut-off chosen
       by the authors, see COHORT_CUTOFF).
==============================================================================
"""
import argparse
import copy
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from imblearn.over_sampling import SMOTE
from scipy.stats import binomtest
from scipy.stats import chi2 as chi2_dist
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc as sk_auc, confusion_matrix,
                             f1_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)

# ------------------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------------------
RANDOM_STATE = 42
COHORT_CUTOFF = 1990   # cohort split: train on reactors first connected up to this year (our choice)
TEST_SIZE = 0.20
CV_FOLDS = 5
HPO_ITER = 40          # random-search iterations per model
N_BOOTSTRAP = 500      # bootstrap resamples of the test set
FIG_DPI = 900          # journal quality; use --dpi 300 for quick looks
CLASS_LIST = ["Low", "Medium", "High"]
CLASS_SHORT = ["Low", "Med", "High"]

# Colours (readable in print and for colour-blind readers)
C_M1 = "#8DA0B8"       # Model 1: soft steel blue
C_M2 = "#1F3A5F"       # Model 2: dark navy
C_TEMP = "#1F3A5F"     # temporal features
C_DESIGN = "#B8C2D0"   # design features
C_CLASS = {"Low": "#2E7D5B", "Medium": "#C28A1E", "High": "#A43A3A"}

# NSSS supplier -> technology group (same map as Model 1)
NSSS_GROUP_MAP = {
    "WH": "Western_PWR", "CE": "Western_PWR", "B&W": "Western_PWR",
    "WH/MHI": "Western_PWR", "ACECOWE": "Western_PWR", "ACLF": "Western_PWR",
    "FAEA": "Western_PWR",
    "FRAM": "French_PWR", "ORANO": "French_PWR", "KWU": "French_PWR",
    "SIEMENS": "French_PWR", "S/KWU": "French_PWR", "GETSCO": "French_PWR",
    "GE": "GE_BWR", "HITACHI": "GE_BWR", "TOSHIBA": "GE_BWR",
    "ABB ATOM": "GE_BWR", "ASEASTAL": "GE_BWR",
    "AEM": "Russian_VVER", "JSC ASE": "Russian_VVER", "IZ": "Russian_VVER",
    "MAEP": "Russian_VVER", "AEE": "Russian_VVER", "PAIP": "Russian_VVER",
    "PAA": "Russian_VVER", "SKODA": "Russian_VVER", "CZEC": "Russian_VVER",
    "CNNC": "Chinese", "DEC": "Chinese", "CFHI": "Chinese",
    "NPIC": "Chinese", "SHE": "Chinese",
    "AECL": "Canadian_CANDU", "OH/AECL": "Canadian_CANDU",
    "AECL/DHI": "Canadian_CANDU", "AECL/DAE": "Canadian_CANDU",
    "NPCIL": "Canadian_CANDU",
    "DHICKOPC": "Korean_Other", "DHICKAEC": "Korean_Other",
    "KEPCO": "Korean_Other", "MHI": "Korean_Other", "NPC": "Korean_Other",
    "NNC": "Korean_Other", "PPC": "Korean_Other",
}

FEAT_M1 = [
    "Type_enc", "Country_OOF", "Net_Capacity_MW", "CapCat_enc",
    "Reactor_Age", "Operational_Years", "Age_squared", "NSSS_Group_enc",
    "Construction_Duration", "Thermal_Efficiency", "Model_Family_enc",
    "Age_Group", "Op_Category",
]
FEAT_TEMPORAL = ["UCL_P1", "UCL_P2", "UCL_change_P1_P2",
                 "UCL_mean_2period", "UCL_trend_dir"]
FEAT_M2 = FEAT_M1 + FEAT_TEMPORAL

FEAT_LABELS = {
    "Type_enc": "Reactor type", "Country_OOF": "Country (UCL tendency)",
    "Net_Capacity_MW": "Net capacity (MW)", "CapCat_enc": "Capacity category",
    "Reactor_Age": "Reactor age (yr)", "Operational_Years": "Operational years",
    "Age_squared": "Age squared", "NSSS_Group_enc": "NSSS technology group",
    "Construction_Duration": "Construction duration (yr)",
    "Thermal_Efficiency": "Thermal efficiency (%)",
    "Model_Family_enc": "Model family", "Age_Group": "Age group",
    "Op_Category": "Operational category",
    "UCL_P1": "UCL 2010-2013", "UCL_P2": "UCL 2014-2017",
    "UCL_change_P1_P2": "UCL change (P1 to P2)",
    "UCL_mean_2period": "UCL mean (2 periods)",
    "UCL_trend_dir": "UCL trend direction",
}

# Columns that define the target and must never be model inputs
FORBIDDEN_FEATURES = {"UCL_Value", "UCL_Class", "UCL_Years_Used", "Target"}


# ------------------------------------------------------------------------------
# SMALL HELPERS
# ------------------------------------------------------------------------------
def say(text=""):
    print(text, flush=True)


def find_dataset(cli_path):
    """Look for the dataset next to the script, in ../data, or where --data says."""
    here = os.path.dirname(os.path.abspath(__file__))
    name = "UCL_Dataset_v6_PRIS.xlsx"
    candidates = [cli_path] if cli_path else []
    candidates += [name, os.path.join(here, name),
                   os.path.join(here, "..", "data", name),
                   os.path.join("data", name)]
    for p in candidates:
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        "Could not find " + name + ". Put it in the data/ folder or run:\n"
        "    python UCL_Model2_Pipeline.py --data path/to/" + name)


def set_figure_style():
    """Journal style: serif font, no grid, no figure title, boxed axes."""
    wanted = ["Times New Roman", "Liberation Serif", "Tinos", "DejaVu Serif"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    family = next((w for w in wanted if w in have), "serif")
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [family],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.grid": False, "axes.linewidth": 0.7,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "legend.frameon": False, "figure.dpi": 100,
        "savefig.bbox": "tight", "pdf.fonttype": 42,
    })
    return family


def save_fig(fig, out_dir, name, dpi):
    path = os.path.join(out_dir, name + ".png")
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    say("  saved figure " + name + ".png")


def model_family(text):
    """Model-name rule used in Model 2: only names containing CANDU count as CANDU."""
    m = str(text).upper()
    if "VVER" in m or "WWER" in m:
        return "VVER"
    if "BWR" in m:
        return "BWR"
    if "CANDU" in m:
        return "CANDU"
    if "AP-1000" in m or "AP1000" in m:
        return "AP1000"
    if "EPR" in m:
        return "EPR"
    if "AGR" in m or "GCR" in m:
        return "GCR"
    if "BN" in m or "FBR" in m:
        return "FBR"
    return "Other"


def age_group(age):
    return 0 if age <= 10 else 1 if age <= 25 else 2 if age <= 40 else 3


def trend_direction(change):
    """+1 UCL rose, -1 UCL fell, 0 about the same (more than 2 points counts as a change)."""
    return 1 if change > 2 else -1 if change < -2 else 0


# ------------------------------------------------------------------------------
# STEP 1: LOAD AND CLEAN THE DATA
# ------------------------------------------------------------------------------
def load_data(xlsx):
    """Returns (model-2 data, number of labelled reactors, number of countries in all)."""
    df_all = load_all(xlsx)
    n_all = len(df_all)
    n_countries_all = int(df_all["Country"].nunique())
    df = df_all.dropna(subset=["UCL_2010_2013", "UCL_2014_2017"]).reset_index(drop=True)
    assert (df["Model2_Subset"] == 1).all()
    return df, n_all, n_countries_all


def load_all(xlsx):
    """All labelled reactors with their history columns (blank where no history exists)."""
    raw = pd.read_excel(xlsx, sheet_name="FINAL_DATASET", header=0)
    raw.columns = [str(c).split("\n")[0].strip() for c in raw.columns]
    df = raw[raw["Reactor_Name"].notna() &
             raw["UCL_Class"].isin(CLASS_LIST)].copy().reset_index(drop=True)
    df["Reactor_Name"] = df["Reactor_Name"].astype(str).str.strip()
    for col in ["Country", "Type", "NSSS_Supplier", "Reactor_Model"]:
        df[col] = df[col].astype(str).str.strip().str.upper()
    for col in ["Net_Capacity_MW", "Reactor_Age", "Operational_Years",
                "Construction_Duration", "First_Grid_Year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Missing values: median (or 10 operational years), as in Model 1
    df["Net_Capacity_MW"] = df["Net_Capacity_MW"].fillna(df["Net_Capacity_MW"].median())
    df["Reactor_Age"] = df["Reactor_Age"].fillna(df["Reactor_Age"].median())
    df["Operational_Years"] = df["Operational_Years"].fillna(10)
    df["Construction_Duration"] = df["Construction_Duration"].fillna(
        df["Construction_Duration"].median())
    if "Capacity_Category" not in df.columns:
        df["Capacity_Category"] = df["Net_Capacity_MW"].apply(
            lambda x: "<600 MWe" if x < 600 else ">=600 MWe")
    # Data category used by the temporal split: A = 10 or more years of operation
    df["Data_Category"] = "A"
    df.loc[df["Operational_Years"] < 10, "Data_Category"] = "B"
    df.loc[df["Operational_Years"] < 5, "Data_Category"] = "C"

    # Attach the history blocks (the label columns UCL_Value, UCL_Class and UCL_Years_Used
    # are kept only to define the target and are never model inputs)
    lg = pd.read_excel(xlsx, sheet_name="HISTORY", header=0)
    lg["Reactor_Name"] = lg["Reactor_Name"].astype(str).str.strip()
    if lg["Reactor_Name"].duplicated().any():
        raise ValueError("Duplicate reactor names in HISTORY sheet.")
    n_before = len(df)
    df = df.merge(lg[["Reactor_Name", "UCL_2010_2013", "UCL_2014_2017",
                      "UCF_2010_2013", "UCF_2014_2017", "Model2_Subset"]],
                  on="Reactor_Name", how="left")
    if len(df) != n_before:
        raise ValueError("Merge changed the number of reactors.")
    return df


# ------------------------------------------------------------------------------
# STEP 2: BUILD THE 18 FEATURES
# ------------------------------------------------------------------------------
def engineer(df, use_pris_thermal=False):
    df = df.copy()
    df["Target"] = df["UCL_Class"].map({"Low": 0, "Medium": 1, "High": 2})

    df["Type_enc"] = LabelEncoder().fit_transform(df["Type"])

    # Country: out-of-fold mean of the target (each reactor sees only other folds)
    kf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    df["Country_OOF"] = np.nan
    for tr, va in kf.split(df, df["Target"]):
        fold_mean = df.iloc[tr].groupby("Country")["Target"].mean()
        df.iloc[va, df.columns.get_loc("Country_OOF")] = \
            df.iloc[va]["Country"].map(fold_mean)
    overall = df.groupby("Country")["Target"].mean()
    df["Country_OOF"] = df["Country_OOF"].fillna(df["Country"].map(overall))

    df["CapCat_enc"] = LabelEncoder().fit_transform(df["Capacity_Category"])
    df["Age_squared"] = df["Reactor_Age"] ** 2
    df["NSSS_Group"] = df["NSSS_Supplier"].map(NSSS_GROUP_MAP).fillna("Other")
    df["NSSS_Group_enc"] = LabelEncoder().fit_transform(df["NSSS_Group"])

    if use_pris_thermal:
        df["Thermal_Efficiency"] = pd.to_numeric(
            df["Thermal_Efficiency_Pct"], errors="coerce").clip(10, 50).fillna(33.0)
    else:
        # The dataset has no thermal power column, so this is the constant 33.0
        df["Thermal_Efficiency"] = 33.0

    df["Model_Family"] = df["Reactor_Model"].apply(model_family)
    df["Model_Family_enc"] = LabelEncoder().fit_transform(df["Model_Family"])
    df["Age_Group"] = df["Reactor_Age"].apply(age_group)
    df["Op_Category"] = df["Data_Category"].map({"A": 2, "B": 1, "C": 0})

    # Five history features from the two UCL history periods (both before 2018)
    df["UCL_P1"] = df["UCL_2010_2013"]
    df["UCL_P2"] = df["UCL_2014_2017"]
    df["UCL_change_P1_P2"] = df["UCL_P2"] - df["UCL_P1"]
    df["UCL_mean_2period"] = (df["UCL_P1"] + df["UCL_P2"]) / 2
    df["UCL_trend_dir"] = df["UCL_change_P1_P2"].apply(trend_direction)

    leaked = FORBIDDEN_FEATURES.intersection(FEAT_M2)
    if leaked:
        raise RuntimeError("Target-defining column used as a feature: " + str(leaked))
    return df


# ------------------------------------------------------------------------------
# STEP 3: MODELS AND TRAINING
# ------------------------------------------------------------------------------
def make_xgb():
    return XGBClassifier(random_state=RANDOM_STATE, eval_metric="mlogloss",
                         verbosity=0, n_jobs=-1)


def make_stacking():
    return StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                          random_state=RANDOM_STATE, n_jobs=-1)),
            ("etc", ExtraTreesClassifier(n_estimators=200, class_weight="balanced",
                                         random_state=RANDOM_STATE, n_jobs=-1)),
            ("xgb", make_xgb()),
        ],
        final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        cv=5, passthrough=False, n_jobs=-1)


def make_models():
    return {
        "LR": LogisticRegression(class_weight="balanced", max_iter=2000,
                                 random_state=RANDOM_STATE),
        "RF": RandomForestClassifier(class_weight="balanced", n_estimators=300,
                                     random_state=RANDOM_STATE, n_jobs=-1),
        "ETC": ExtraTreesClassifier(class_weight="balanced", n_estimators=300,
                                    random_state=RANDOM_STATE, n_jobs=-1),
        "XGB": make_xgb(),
        "Stacking": make_stacking(),
    }


PARAM_GRIDS = {
    "LR": {"C": np.logspace(-3, 3, 20), "solver": ["lbfgs", "liblinear"]},
    "RF": {"n_estimators": [200, 300, 500], "max_depth": [None, 10, 20],
           "min_samples_split": [2, 5, 10], "max_features": ["sqrt", "log2"]},
    "ETC": {"n_estimators": [200, 300, 500], "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10], "max_features": ["sqrt", "log2"]},
    "XGB": {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1], "subsample": [0.8, 1.0]},
    "Stacking": {},   # not tuned
}


def score_model(model, X_te, y_te):
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)
    return {
        "Accuracy": round(accuracy_score(y_te, pred) * 100, 2),
        "F1": round(f1_score(y_te, pred, average="weighted", zero_division=0) * 100, 2),
        "AUC_ROC": round(roc_auc_score(y_te, proba, multi_class="ovr",
                                       average="weighted") * 100, 2),
        "MCC": round(float(matthews_corrcoef(y_te, pred)), 3),
    }, pred, proba


def train_all(X_tr, y_tr, X_te, y_te, label, hpo_iter):
    kf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results, preds, probas, fitted, params = {}, {}, {}, {}, {}
    for name, base in make_models().items():
        say("  [" + label + "] " + name + " ...")
        if PARAM_GRIDS[name]:
            search = RandomizedSearchCV(
                copy.deepcopy(base), PARAM_GRIDS[name], n_iter=hpo_iter, cv=kf,
                scoring="f1_weighted", random_state=RANDOM_STATE, n_jobs=-1)
            search.fit(X_tr, y_tr)
            model = search.best_estimator_
            params[name] = {k: (v.item() if hasattr(v, "item") else v)
                            for k, v in search.best_params_.items()}
        else:
            model = copy.deepcopy(base).fit(X_tr, y_tr)
            params[name] = {}
        res, pred, proba = score_model(model, X_te, y_te)
        cv = cross_val_score(model, X_tr, y_tr, cv=kf, scoring="accuracy")
        res["CV_Mean_balanced_train"] = round(cv.mean() * 100, 2)
        res["CV_Std_balanced_train"] = round(cv.std() * 100, 2)
        results[name], preds[name], probas[name], fitted[name] = res, pred, proba, model
        say("      accuracy %.2f  F1 %.2f  AUC %.2f" %
            (res["Accuracy"], res["F1"], res["AUC_ROC"]))
    return results, preds, probas, fitted, params


def pick_best(results):
    """Best model = highest accuracy; ties go to higher F1, then higher AUC."""
    return max(results, key=lambda k: (results[k]["Accuracy"], results[k]["F1"],
                                       results[k]["AUC_ROC"]))


def per_class_table(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    return {c: {"precision": round(float(p[i]) * 100, 1),
                "recall": round(float(r[i]) * 100, 1),
                "f1": round(float(f[i]) * 100, 1), "support": int(s[i])}
            for i, c in enumerate(CLASS_LIST)}


def mcnemar_test(correct_a, correct_b):
    """McNemar on paired right/wrong outcomes: chi-square with continuity
    correction and the exact binomial p-value."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return b, c, 0.0, 1.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    return (b, c, float(chi2), float(chi2_dist.sf(chi2, 1)),
            float(binomtest(min(b, c), n, 0.5).pvalue))


def bootstrap_ci(model, X_te, y_te, n_boot):
    acc, f1, auc = [], [], []
    for i in range(n_boot):
        idx = resample(range(len(X_te)), random_state=i)
        Xb, yb = X_te.iloc[idx], y_te.iloc[idx]
        pred = model.predict(Xb)
        acc.append(accuracy_score(yb, pred) * 100)
        f1.append(f1_score(yb, pred, average="weighted", zero_division=0) * 100)
        try:
            auc.append(roc_auc_score(yb, model.predict_proba(Xb), multi_class="ovr",
                                     average="weighted") * 100)
        except ValueError:     # a class is missing from this resample
            pass
    return {"Accuracy": acc, "F1": f1, "AUC": auc}


# ------------------------------------------------------------------------------
# FIGURES (no titles inside the figures, no grid, serif font)
# ------------------------------------------------------------------------------
def fig_compare(res1, res2, out, dpi):
    names = list(res1)
    x = np.arange(len(names))
    w = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, key, lab in zip(axes, ["Accuracy", "F1", "AUC_ROC"],
                            ["Accuracy (%)", "Weighted F1 (%)", "AUC-ROC (%)"]):
        v1 = [res1[n][key] for n in names]
        v2 = [res2[n][key] for n in names]
        ax.bar(x - w / 2, v1, w, color=C_M1, edgecolor="black", linewidth=0.4,
               label="Model 1 (13 features)")
        ax.bar(x + w / 2, v2, w, color=C_M2, edgecolor="black", linewidth=0.4,
               label="Model 2 (18 features)")
        for xi, (a, b) in enumerate(zip(v1, v2)):
            ax.text(xi - w / 2, a + 1, "%.1f" % a, ha="center", fontsize=5.5)
            ax.text(xi + w / 2, b + 1, "%.1f" % b, ha="center", fontsize=5.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25)
        ax.set_ylim(50, 108)
        ax.set_ylabel(lab)
    axes[0].legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    save_fig(fig, out, "M2_fig01_m1_vs_m2_comparison", dpi)


def fig_perclass(pc1, pc2, n1, n2, out, dpi):
    x = np.arange(3)
    w = 0.36
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    v1 = [pc1[c]["f1"] for c in CLASS_LIST]
    v2 = [pc2[c]["f1"] for c in CLASS_LIST]
    ax.bar(x - w / 2, v1, w, color=C_M1, edgecolor="black", linewidth=0.4,
           label="Model 1 (" + n1 + ")")
    ax.bar(x + w / 2, v2, w, color=C_M2, edgecolor="black", linewidth=0.4,
           label="Model 2 (" + n2 + ")")
    for xi, (a, b) in enumerate(zip(v1, v2)):
        ax.text(xi - w / 2, a + 1.2, "%.1f" % a, ha="center", fontsize=6.5)
        ax.text(xi + w / 2, b + 1.2, "%.1f" % b, ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LIST)
    ax.set_ylim(0, 118)
    ax.set_ylabel("F1-score (%)")
    ax.legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    save_fig(fig, out, "M2_fig02_perclass_f1", dpi)


def fig_confusion(preds, results, y_te, out, dpi):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    flat = axes.flatten()
    for ax, name in zip(flat, list(results)):
        cm = confusion_matrix(y_te, preds[name], labels=[0, 1, 2])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1) * 1.15)
        for i in range(3):
            for j in range(3):
                dark = cm[i, j] > cm.max() * 0.55
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if dark else "black")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_SHORT)
        ax.set_yticklabels(CLASS_SHORT)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("%s (%.1f%%)" % (name, results[name]["Accuracy"]), fontsize=8)
        ax.tick_params(top=False, right=False)
    flat[-1].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out, "M2_fig03_confusion_matrices", dpi)


def fig_importance(fi_df, name, out, dpi):
    """Horizontal bar chart of the importance shares for one model."""
    d = fi_df.sort_values("Importance")
    fig, ax = plt.subplots(figsize=(3.6, 4.2))
    colours = [C_TEMP if t else C_DESIGN for t in d["Is_Temporal"]]
    ax.barh(d["Feature"], d["Importance"] * 100, color=colours,
            edgecolor="black", linewidth=0.4)
    for i, v in enumerate(d["Importance"] * 100):
        ax.text(v + 0.3, i, "%.1f" % v, va="center", fontsize=6)
    ax.set_xlim(0, d["Importance"].max() * 100 * 1.15)
    ax.set_xlabel("Importance (%), " + name)
    ax.tick_params(axis="y", labelsize=6.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=C_TEMP, edgecolor="black", linewidth=0.4,
                             label="History (UCL) features"),
                       Patch(facecolor=C_DESIGN, edgecolor="black", linewidth=0.4,
                             label="Design features")],
              loc="lower right", fontsize=6.5)
    fig.tight_layout()
    save_fig(fig, out, "M2_fig04_feature_importance_" + name, dpi)


def fig_roc(proba, y_te, out, dpi):
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    for i, c in enumerate(CLASS_LIST):
        fpr, tpr, _ = roc_curve((y_te == i).astype(int), proba[:, i])
        ax.plot(fpr, tpr, color=C_CLASS[c], lw=1.5,
                label="%s (AUC = %.3f)" % (c, sk_auc(fpr, tpr)))
    ax.plot([0, 1], [0, 1], color="black", lw=0.7, ls="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, out, "M2_fig05_roc_curves", dpi)


def fig_bootstrap(boot, out, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for ax, (key, vals) in zip(axes, boot.items()):
        v = np.array(vals)
        lo, hi = np.percentile(v, [2.5, 97.5])
        ax.hist(v, bins=30, color=C_M2, edgecolor="white", linewidth=0.3)
        ax.axvline(v.mean(), color="black", lw=1.0)
        ax.axvline(lo, color="black", lw=0.8, ls="--")
        ax.axvline(hi, color="black", lw=0.8, ls="--")
        ax.set_xlabel(key + " (%)")
        ax.set_ylabel("Count")
        ax.text(0.03, 0.95, "mean %.1f\n95 %% CI [%.1f, %.1f]" % (v.mean(), lo, hi),
                transform=ax.transAxes, va="top", fontsize=6.5)
    fig.tight_layout()
    save_fig(fig, out, "M2_fig06_bootstrap", dpi)


def fig_temporal(random_acc, temporal, out, dpi):
    names = list(temporal)
    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    r = [random_acc[n] for n in names]
    t = [temporal[n]["Accuracy"] for n in names]
    ax.bar(x - w / 2, r, w, color=C_M2, edgecolor="black", linewidth=0.4,
           label="Random split")
    ax.bar(x + w / 2, t, w, color="#C28A1E", edgecolor="black", linewidth=0.4,
           label="Temporal split")
    for xi, (a, b) in enumerate(zip(r, t)):
        ax.text(xi - w / 2, a + 1, "%.1f" % a, ha="center", fontsize=6.5)
        ax.text(xi + w / 2, b + 1, "%.1f" % b, ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(50, 108)
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="upper right", fontsize=6.5)
    fig.tight_layout()
    save_fig(fig, out, "M2_fig07_temporal_validation", dpi)


def fig_trajectory(traj, out, dpi):
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    xs = [0, 1]
    for c in CLASS_LIST:
        ys = [traj.loc[c, "UCL_P1"], traj.loc[c, "UCL_P2"]]
        ax.plot(xs, ys, marker="o", ms=4, lw=1.6, color=C_CLASS[c], label=c)
    ax.set_xticks(xs)
    ax.set_xticklabels(["2010-2013", "2014-2017"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_xlabel("UCL averaging period")
    ax.set_ylabel("Mean UCL (%)")
    ax.legend(loc="center right")
    fig.tight_layout()
    save_fig(fig, out, "M2_fig08_ucl_trajectory", dpi)


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="UCL Model 2 pipeline")
    ap.add_argument("--data", default=None, help="path to the dataset workbook")
    ap.add_argument("--out", default="results_Model2", help="output folder")
    ap.add_argument("--dpi", type=int, default=FIG_DPI, help="figure resolution")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test with a short search (numbers will differ)")
    ap.add_argument("--pris-thermal", action="store_true",
                    help="use the Thermal_Efficiency_Pct column instead of 33.0")
    args = ap.parse_args()

    hpo_iter = 4 if args.quick else HPO_ITER
    n_boot = 50 if args.quick else N_BOOTSTRAP
    os.makedirs(args.out, exist_ok=True)
    font = set_figure_style()

    say("=" * 70)
    say("UCL MODEL 2: design features + UCL history 2010-2017")
    if args.quick:
        say("QUICK RUN: short search, numbers will NOT match the paper.")
    say("=" * 70)

    # 1. data
    xlsx = find_dataset(args.data)
    say("\n[1] Loading " + os.path.basename(xlsx))
    df, n_all, n_countries_all = load_data(xlsx)
    say("    reactors in the full dataset      : %d (%d countries)" % (n_all, n_countries_all))
    say("    reactors with both UCL history values: %d (%d countries)" %
        (len(df), df["Country"].nunique()))
    say("    classes: " + str(df["UCL_Class"].value_counts().to_dict()))

    # 2. features
    say("\n[2] Building %d features (%d design + %d history)" %
        (len(FEAT_M2), len(FEAT_M1), len(FEAT_TEMPORAL)))
    df = engineer(df, use_pris_thermal=args.pris_thermal)
    if not args.pris_thermal:
        say("    NOTE: Thermal_Efficiency is the constant 33.0 (no information).")
        say("          Use --pris-thermal to try the real column.")

    # 3. split
    say("\n[3] Splitting (80/20 stratified, seed %d)" % RANDOM_STATE)
    y = df["Target"]
    scaler = StandardScaler()
    X2 = pd.DataFrame(scaler.fit_transform(df[FEAT_M2]), columns=FEAT_M2)
    X1 = X2[FEAT_M1].copy()
    X2_tr, X2_te, y_tr, y_te = train_test_split(
        X2, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    X1_tr, X1_te = X1.loc[X2_tr.index], X1.loc[X2_te.index]
    say("    train %d | test %d" % (len(X2_tr), len(X2_te)))

    # 4. cohort split sets: older reactors train, newer reactors test. The version 5 split by
    # operational category cannot be used here because every reactor that has UCL history
    # for 2010-2013 began operating by 2011.
    mask_a = (df["First_Grid_Year"] <= COHORT_CUTOFF).values
    X2_trT, y_trT = X2[mask_a], y[mask_a]
    X2_teT, y_teT = X2[~mask_a], y[~mask_a]
    say("    cohort split: train %d (grid connection %d or earlier) | test %d (later)" %
        (len(X2_trT), COHORT_CUTOFF, len(X2_teT)))

    # 5. SMOTE on the training parts only
    say("\n[4] SMOTE on training data only (k = 3)")
    X1_sm, y1_sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=3).fit_resample(X1_tr, y_tr)
    X2_sm, y2_sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=3).fit_resample(X2_tr, y_tr)
    say("    before: " + str(y_tr.value_counts().sort_index().to_dict()))
    say("    after : " + str(pd.Series(y2_sm).value_counts().sort_index().to_dict()))

    # 6. models
    say("\n[5] Model 1 baseline on the same reactors (13 design features)")
    res1, pred1, proba1, fit1, par1 = train_all(X1_sm, y1_sm, X1_te, y_te, "M1", hpo_iter)
    best1 = pick_best(res1)
    say("\n[6] Model 2 (18 features)")
    res2, pred2, proba2, fit2, par2 = train_all(X2_sm, y2_sm, X2_te, y_te, "M2", hpo_iter)
    best2 = pick_best(res2)
    say("\n    best Model 1: %s | best Model 2: %s" % (best1, best2))

    # 7. per-class: the best Model 2 classifier against the SAME classifier in Model 1
    pc1 = per_class_table(y_te, pred1[best2])
    pc2 = per_class_table(y_te, pred2[best2])
    pc_all = {"model1": {n: per_class_table(y_te, pred1[n]) for n in pred1},
              "model2": {n: per_class_table(y_te, pred2[n]) for n in pred2}}
    say("\n[7] Per-class F1 (Model 1 %s -> Model 2 %s)" % (best2, best2))
    for c in CLASS_LIST:
        say("    %-7s %5.1f -> %5.1f  (support %d)" %
            (c, pc1[c]["f1"], pc2[c]["f1"], pc2[c]["support"]))

    # 8. McNemar (paired right/wrong on the same test reactors)
    say("\n[8] McNemar test")
    mc = {}
    for label, a, b in [("same classifier (%s)" % best2, pred1[best2], pred2[best2]),
                        ("best M1 (%s) vs best M2 (%s)" % (best1, best2),
                         pred1[best1], pred2[best2])]:
        bb, cc, chi2, p_chi, p_ex = mcnemar_test(a == y_te.values, b == y_te.values)
        mc[label] = {"b_M1_right_M2_wrong": bb, "c_M1_wrong_M2_right": cc,
                     "chi2": round(chi2, 3), "p_chi2_corrected": round(p_chi, 4),
                     "p_exact": round(p_ex, 4)}
        say("    %-34s b=%d c=%d  p(chi2)=%.4f  p(exact)=%.4f" %
            (label, bb, cc, p_chi, p_ex))

    # 9. bootstrap
    say("\n[9] Bootstrap of the test set (%d resamples), %s" % (n_boot, best2))
    boot = bootstrap_ci(fit2[best2], X2_te, y_te, n_boot)
    boot_rows = []
    for k, v in boot.items():
        v = np.array(v)
        boot_rows.append({"Metric": k, "Mean": round(v.mean(), 2),
                          "Std": round(v.std(), 2),
                          "CI_Low": round(np.percentile(v, 2.5), 1),
                          "CI_High": round(np.percentile(v, 97.5), 1)})
        say("    %-9s %.1f +/- %.1f   95%% CI [%.1f, %.1f]" %
            (k, v.mean(), v.std(), np.percentile(v, 2.5), np.percentile(v, 97.5)))
    df_boot = pd.DataFrame(boot_rows)

    # 10. temporal split (library defaults, no tuning)
    say("\n[10] Cohort split (train: first connected in %d or earlier, test: connected later)" % COHORT_CUTOFF)
    X2_trT_sm, y_trT_sm = SMOTE(random_state=RANDOM_STATE,
                                k_neighbors=3).fit_resample(X2_trT, y_trT)
    temporal = {}
    for name, mdl in [("RF", RandomForestClassifier(
                          n_estimators=300, class_weight="balanced",
                          random_state=RANDOM_STATE, n_jobs=-1)),
                      ("XGB", make_xgb()), ("Stacking", make_stacking())]:
        mdl.fit(X2_trT_sm, y_trT_sm)
        tres, _, _ = score_model(mdl, X2_teT, y_teT)
        temporal[name] = {"Accuracy": tres["Accuracy"], "F1": tres["F1"],
                          "AUC_ROC": tres["AUC_ROC"]}
        say("    %-9s random %.1f -> temporal %.1f (F1 %.1f)" %
            (name, res2[name]["Accuracy"], tres["Accuracy"], tres["F1"]))

    # 11. importance (share of the total) for the best model and for the Random Forest.
    # The two rank the features differently, so both are saved and the model is named.
    say("\n[11] Feature importance (share of the total, impurity based)")
    fi_tables, fi_share = {}, {}
    for imp_name in dict.fromkeys([best2, "RF"]):
        if not hasattr(fit2[imp_name], "feature_importances_"):
            continue
        imp = np.asarray(fit2[imp_name].feature_importances_, dtype=float)
        imp = imp / imp.sum()
        fi = pd.DataFrame({"Feature": [FEAT_LABELS[f] for f in FEAT_M2],
                           "Importance": imp,
                           "Is_Temporal": [f in FEAT_TEMPORAL for f in FEAT_M2]}
                          ).sort_values("Importance", ascending=False)
        fi_tables[imp_name] = fi
        fi_share[imp_name] = float(fi.loc[fi["Is_Temporal"], "Importance"].sum())
        say("    %s: history features together %.1f %%" % (imp_name, fi_share[imp_name] * 100))
        for _, r in fi.head(4).iterrows():
            say("        %-28s %.1f %%" % (r["Feature"], r["Importance"] * 100))

    # 12. mean history UCL by class (supports the trajectory figure)
    traj = df.groupby("UCL_Class")[["UCL_P1", "UCL_P2"]].mean().reindex(CLASS_LIST).round(1)
    say("\n[12] Mean history UCL by class")
    say(traj.to_string())

    # 13. figures
    say("\n[13] Figures (%d dpi, font: %s)" % (args.dpi, font))
    fig_compare(res1, res2, args.out, args.dpi)
    fig_perclass(pc1, pc2, best2, best2, args.out, args.dpi)
    fig_confusion(pred2, res2, y_te, args.out, args.dpi)
    for imp_name, fi in fi_tables.items():
        fig_importance(fi, imp_name, args.out, args.dpi)
    fig_roc(proba2[best2], y_te, args.out, args.dpi)
    fig_bootstrap(boot, args.out, args.dpi)
    fig_temporal({n: res2[n]["Accuracy"] for n in temporal}, temporal,
                 args.out, args.dpi)
    fig_trajectory(traj, args.out, args.dpi)

    # 14. tables and summary
    say("\n[14] Tables")
    t = lambda name: os.path.join(args.out, name)
    pd.DataFrame(res1).T.reset_index().rename(columns={"index": "Model"}).to_csv(
        t("M2_table1_model1_baseline.csv"), index=False)
    pd.DataFrame(res2).T.reset_index().rename(columns={"index": "Model"}).to_csv(
        t("M2_table2_model2_results.csv"), index=False)
    rows = []
    for c in CLASS_LIST:
        rows.append({"Class": c, "Support": pc2[c]["support"],
                     "M1_precision": pc1[c]["precision"], "M1_recall": pc1[c]["recall"],
                     "M1_F1": pc1[c]["f1"], "M2_precision": pc2[c]["precision"],
                     "M2_recall": pc2[c]["recall"], "M2_F1": pc2[c]["f1"],
                     "Delta_F1": round(pc2[c]["f1"] - pc1[c]["f1"], 1)})
    pd.DataFrame(rows).to_csv(t("M2_table3_per_class.csv"), index=False)
    df_boot.to_csv(t("M2_table4_bootstrap.csv"), index=False)
    for imp_name, fi in fi_tables.items():
        fi.to_csv(t("M2_table5_feature_importance_%s.csv" % imp_name), index=False)
    pd.DataFrame([{"Model": n, "Random_split_accuracy": res2[n]["Accuracy"],
                   "Temporal_accuracy": v["Accuracy"], "Temporal_F1": v["F1"],
                   "Temporal_AUC": v["AUC_ROC"]} for n, v in temporal.items()]
                 ).to_csv(t("M2_table6_temporal_split.csv"), index=False)
    traj.reset_index().rename(columns={"UCL_Class": "Class"}).to_csv(
        t("M2_table7_mean_ucl_history_by_class.csv"), index=False)
    rows = []
    for which in ("model1", "model2"):
        for name, tab in pc_all[which].items():
            for c in CLASS_LIST:
                rows.append({"Model_set": which, "Classifier": name, "Class": c,
                             **tab[c]})
    pd.DataFrame(rows).to_csv(t("M2_table8_per_class_all_models.csv"), index=False)

    import sklearn
    import xgboost
    import imblearn
    summary = {
        "n_reactors_full": n_all, "n_reactors_model2": int(len(df)),
        "n_countries_full": n_countries_all,
        "n_countries_model2": int(df["Country"].nunique()),
        "n_train": int(len(X2_tr)), "n_test": int(len(X2_te)),
        "thermal_efficiency": "Thermal_Efficiency_Pct column" if args.pris_thermal
        else "constant 33.0",
        "best_model_1": best1, "best_model_2": best2,
        "model1_results": res1, "model2_results": res2,
        "model1_best_params": par1, "model2_best_params": par2,
        "per_class_all_models": pc_all, "per_class_compared": {
            "classifier": best2, "model1": pc1, "model2": pc2},
        "mcnemar": mc, "temporal_split": temporal,
        "history_feature_importance_share": {k: round(v, 4) for k, v in fi_share.items()},
        "quick_run": bool(args.quick),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "imbalanced-learn": imblearn.__version__,
                     "matplotlib": matplotlib.__version__},
    }
    with open(t("run_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    say("\n" + "=" * 70)
    say("SUMMARY (test set of %d reactors)" % len(X2_te))
    for n, r in res2.items():
        star = "*" if n == best2 else " "
        say(" %s %-9s accuracy %.2f  F1 %.2f  AUC %.2f  MCC %.3f" %
            (star, n, r["Accuracy"], r["F1"], r["AUC_ROC"], r["MCC"]))
    say("Output folder: " + os.path.abspath(args.out))
    say("=" * 70)


if __name__ == "__main__":
    main()
