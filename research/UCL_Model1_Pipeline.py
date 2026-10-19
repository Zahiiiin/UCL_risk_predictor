# ==============================================================================
# UCL NUCLEAR POWER PLANT RISK CLASSIFICATION - MODEL 1
# Static design-based classifier (13 features, 407 reactors)
# ==============================================================================
#
# Paper: "Machine Learning Classification of Unplanned Capability Loss Risk in
#         Nuclear Power Plants" (target journal: Nuclear Engineering and Design)
# Data : IAEA PRIS / RDS-2, file UCL_Dataset_v5_longitudinal.xlsx
#
# WHAT THIS MODEL DOES:
#   Predicts the UCL risk class (Low / Medium / High) using ONLY reactor design
#   and identity features. No operating history (no UCF) is used, so the model
#   can be applied to a reactor before it has any operating record.
#
# FEATURES (13 in total):
#   Original (6):    Reactor type, Country (out-of-fold encoded), Net capacity,
#                    Capacity category, Reactor age, Operational years
#   Engineered (7):  Age squared, NSSS technology group, Construction duration,
#                    Thermal efficiency, Model family, Age group,
#                    Operational category
#
# RESULTS REPORTED IN THE PAPER (test set of 82 reactors, seed 42):
#   Stacking ensemble: 76.8 % accuracy, 86.9 % AUC-ROC (weighted one-vs-rest)
#   Temporal split (train on reactors with 10+ years of operation, test on
#   newer reactors): 84.1 % accuracy for the best model
#   Per-class F1 of the stacking ensemble: Low 89.8, Medium 27.3, High 72.7.
#   These numbers need the library versions pinned in requirements-research.txt.
#   scikit-learn 1.9 changes the Random Forest results by one reactor.
#
# THINGS A REVIEWER SHOULD KNOW:
#   1. Thermal_Efficiency is the constant 33.0 for every reactor, because the
#      dataset has no thermal power column. It therefore carries no information
#      (its importance of 0.000 is a property of the data, not a finding).
#      Set USE_PRIS_THERMAL = True below to use the Thermal_Efficiency_Pct column.
#   2. Scaling and the Country encoding are computed on all reactors before the
#      train/test split. The encoding is out-of-fold and trees ignore scaling,
#      but a stricter design would fit both on the training part only.
#   3. SMOTE is applied to the training part only. The cross-validation used
#      inside the random search runs on that balanced training set, so the CV
#      scores are optimistic. Test-set scores are not affected.
#
# HOW TO RUN:
#   pip install -r requirements-research.txt
#   python UCL_Model1_Pipeline.py
#   (the dataset is found automatically in ../data/ or next to this script)
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, os, copy
warnings.filterwarnings('ignore')
np.random.seed(42)

from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_val_score, RandomizedSearchCV
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import (
    RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
)
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc as sk_auc
)
from sklearn.utils import resample

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not found - Stacking will use ETC instead")

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("imbalanced-learn not found - SMOTE disabled. Run: pip install imbalanced-learn")

# -- SETTINGS ------------------------------------------------------------------
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
CV_FOLDS      = 5
HPO_ITER      = 40
N_BOOTSTRAP   = 500
USE_PRIS_THERMAL = False   # True: use Thermal_Efficiency_Pct instead of the constant 33.0
OUT           = 'results_Model1'
os.makedirs(OUT, exist_ok=True)

CLASS_LIST    = ['Low', 'Medium', 'High']
CLASS_COLORS  = {'Low': '#2B5E38', 'Medium': '#8C5E10', 'High': '#7A1C1C'}
DARK, MID     = '#1C3557', '#4A7BA7'

# -- NSSS TECHNOLOGY GROUP MAP -------------------------------------------------
# Reactors sharing the same NSSS lineage have similar design reliability
# characteristics. Grouping reduces sparsity vs raw supplier names.
NSSS_GROUP_MAP = {
    'WH':'Western_PWR',    'CE':'Western_PWR',    'B&W':'Western_PWR',
    'WH/MHI':'Western_PWR','ACECOWE':'Western_PWR','ACLF':'Western_PWR',
    'FAEA':'Western_PWR',
    'FRAM':'French_PWR',   'ORANO':'French_PWR',   'KWU':'French_PWR',
    'SIEMENS':'French_PWR','S/KWU':'French_PWR',   'GETSCO':'French_PWR',
    'GE':'GE_BWR',         'HITACHI':'GE_BWR',     'TOSHIBA':'GE_BWR',
    'ABB ATOM':'GE_BWR',   'ASEASTAL':'GE_BWR',
    'AEM':'Russian_VVER',  'JSC ASE':'Russian_VVER','IZ':'Russian_VVER',
    'MAEP':'Russian_VVER', 'AEE':'Russian_VVER',   'PAIP':'Russian_VVER',
    'PAA':'Russian_VVER',  'SKODA':'Russian_VVER', 'CZEC':'Russian_VVER',
    'CNNC':'Chinese',      'DEC':'Chinese',         'CFHI':'Chinese',
    'NPIC':'Chinese',      'SHE':'Chinese',
    'AECL':'Canadian_CANDU','OH/AECL':'Canadian_CANDU',
    'AECL/DHI':'Canadian_CANDU','AECL/DAE':'Canadian_CANDU','NPCIL':'Canadian_CANDU',
    'DHICKOPC':'Korean_Other','DHICKAEC':'Korean_Other','KEPCO':'Korean_Other',
    'MHI':'Korean_Other',  'NPC':'Korean_Other',   'NNC':'Korean_Other',
    'PPC':'Korean_Other',
}

def save_fig(fig, name):
    path = f"{OUT}/{name}.png"
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}.png")

print("=" * 65)
print("  UCL PREDICTION - MODEL 1: Static Design Classifier")
print("  13 features | n=407 reactors | Target: Low/Medium/High")
print("=" * 65)

# ==============================================================================
# STEP 1 - LOAD DATA
# ==============================================================================
print("\n[1] Loading data...")

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATHS = [
    'UCL_Dataset_v5_longitudinal.xlsx',
    os.path.join(_HERE, 'UCL_Dataset_v5_longitudinal.xlsx'),
    os.path.join(_HERE, '..', 'data', 'UCL_Dataset_v5_longitudinal.xlsx'),
    os.path.join('data', 'UCL_Dataset_v5_longitudinal.xlsx'),
]
df_raw = None
for path in DATASET_PATHS:
    try:
        df_raw = pd.read_excel(path, sheet_name='FINAL_DATASET', header=0)
        if len(df_raw) > 10:
            print(f"  Loaded: {path}")
            break
    except Exception:
        continue

if df_raw is None:
    raise FileNotFoundError(
        "Cannot find dataset. Place UCL_Dataset_v5_longitudinal.xlsx "
        "in the same folder as this script."
    )

# Flatten multi-line headers if present
df_raw.columns = [str(c).split('\n')[0].strip() for c in df_raw.columns]

# Keep valid reactors only
df = df_raw[
    df_raw['Reactor_Name'].notna() &
    df_raw['UCL_Class'].isin(['Low', 'Medium', 'High'])
].copy().reset_index(drop=True)

# Standardise text
for col in ['Country', 'Type', 'NSSS_Supplier', 'Reactor_Model']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper()

# Numeric columns - Python 3.13 compatible (no inplace=True)
for col in ['Net_Capacity_MW', 'Reactor_Age', 'Operational_Years',
            'Construction_Duration', 'UCL_Computed', 'UCF_Percent',
            'First_Grid_Year', 'UCL_Negative_Flag']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing with medians
df['Net_Capacity_MW']       = df['Net_Capacity_MW'].fillna(df['Net_Capacity_MW'].median())
df['Reactor_Age']           = df['Reactor_Age'].fillna(df['Reactor_Age'].median())
df['Operational_Years']     = df['Operational_Years'].fillna(10)
df['Construction_Duration'] = df['Construction_Duration'].fillna(
                              df['Construction_Duration'].median() if 'Construction_Duration' in df.columns else 8)
df['UCL_Negative_Flag']     = df['UCL_Negative_Flag'].fillna(0) if 'UCL_Negative_Flag' in df.columns else 0

# UCL value
ucl_col = 'UCL_Computed' if 'UCL_Computed' in df.columns else 'UCL_Value'
if ucl_col in df.columns:
    df['UCL_Value'] = pd.to_numeric(df[ucl_col], errors='coerce').fillna(0).clip(lower=0)
else:
    df['UCL_Value'] = 0.0

# Capacity category
if 'Capacity_Category' not in df.columns:
    df['Capacity_Category'] = df['Net_Capacity_MW'].apply(
        lambda x: '<600 MWe' if x < 600 else '>=600 MWe')

# Three data categories
df['Data_Category'] = 'A'
df.loc[df['Operational_Years'] < 10, 'Data_Category'] = 'B'
df.loc[df['Operational_Years'] <  5, 'Data_Category'] = 'C'

print(f"  Total reactors: {len(df)}")
print(f"  Countries:      {df['Country'].nunique()}")
print(f"  Types:          {df['Type'].nunique()}")
print(f"  UCL dist:       {df['UCL_Class'].value_counts().to_dict()}")

# ==============================================================================
# STEP 2 - FEATURE ENGINEERING (13 features)
# ==============================================================================
print("\n[2] Engineering 13 features...")

# -- Target variable ------------------------------------------------------------
df['Target'] = df['UCL_Class'].map({'Low': 0, 'Medium': 1, 'High': 2})

# -- Feature 1: Reactor Type -------------------------------------------------
le_type = LabelEncoder()
df['Type_enc'] = le_type.fit_transform(df['Type'])

# -- Feature 2: Country - Out-of-Fold (OOF) Target Encoding -----------------
# OOF prevents data leakage - each fold's training mean, not global mean
kf_oof = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
df['Country_OOF'] = np.nan
for tr_idx, val_idx in kf_oof.split(df, df['Target']):
    fold_mean = df.iloc[tr_idx].groupby('Country')['Target'].mean()
    df.iloc[val_idx, df.columns.get_loc('Country_OOF')] = \
        df.iloc[val_idx]['Country'].map(fold_mean)
global_mean = df.groupby('Country')['Target'].mean()
df['Country_OOF'] = df['Country_OOF'].fillna(df['Country'].map(global_mean))

# -- Feature 3: Net Capacity (MW) - continuous ------------------------------
# (already cleaned above)

# -- Feature 4: Capacity Category --------------------------------------------
le_cap = LabelEncoder()
df['CapCat_enc'] = le_cap.fit_transform(df['Capacity_Category'])

# -- Feature 5: Reactor Age --------------------------------------------------
# (already cleaned above)

# -- Feature 6: Operational Years --------------------------------------------
# (already cleaned above)

# -- Feature 7 (NEW): Age² - captures non-linear aging effect ---------------
df['Age_squared'] = df['Reactor_Age'] ** 2

# -- Feature 8 (NEW): NSSS Technology Group ----------------------------------
if 'NSSS_Supplier' in df.columns:
    df['NSSS_Group'] = df['NSSS_Supplier'].map(NSSS_GROUP_MAP).fillna('Other')
else:
    df['NSSS_Group'] = 'Other'
le_nsss = LabelEncoder()
df['NSSS_Group_enc'] = le_nsss.fit_transform(df['NSSS_Group'])

# -- Feature 9 (NEW): Construction Duration -----------------------------------
if 'Construction_Duration' not in df.columns:
    df['Construction_Duration'] = 8.0  # global median fallback

# -- Feature 10 (NEW): Thermal Efficiency % -----------------------------------
if USE_PRIS_THERMAL and 'Thermal_Efficiency_Pct' in df.columns:
    df['Thermal_Efficiency'] = pd.to_numeric(
        df['Thermal_Efficiency_Pct'], errors='coerce').clip(10, 50).fillna(33.0)
elif 'Thermal_Capacity_MW' in df.columns:
    df['Thermal_Efficiency'] = (
        df['Net_Capacity_MW'] / df['Thermal_Capacity_MW'] * 100
    ).clip(10, 50).fillna(33.0)
else:
    # The dataset has no thermal power column, so every reactor gets 33.0 and this
    # feature carries no information (see "THINGS A REVIEWER SHOULD KNOW" above).
    df['Thermal_Efficiency'] = 33.0

# -- Feature 11 (NEW): Model Family (simplified from Reactor_Model) -----------
if 'Reactor_Model' in df.columns:
    def model_family(m):
        m = str(m).upper()
        if 'VVER' in m or 'WWER' in m: return 'VVER'
        if 'BWR'  in m: return 'BWR'
        if 'CANDU' in m or 'PHWR' in m: return 'CANDU'
        if 'AP-1000' in m or 'AP1000' in m: return 'AP1000'
        if 'EPR'  in m: return 'EPR'
        if 'AGR'  in m or 'GCR' in m: return 'GCR'
        if 'BN'   in m or 'FBR' in m: return 'FBR'
        return 'Other'
    df['Model_Family'] = df['Reactor_Model'].apply(model_family)
else:
    df['Model_Family'] = 'Other'
le_mf = LabelEncoder()
df['Model_Family_enc'] = le_mf.fit_transform(df['Model_Family'])

# -- Feature 12 (NEW): Age Group (ordinal) ------------------------------------
# Based on nuclear aging zones:
#   0-10yr = New, 11-25yr = Young, 26-40yr = Mid-life (peak issues), 41+yr = Mature
def age_group(age):
    if age <= 10:  return 0
    if age <= 25:  return 1
    if age <= 40:  return 2
    return 3
df['Age_Group'] = df['Reactor_Age'].apply(age_group)

# -- Feature 13 (NEW): Operational Category (by data reliability) -------------
df['Op_Category'] = df['Data_Category'].map({'A': 2, 'B': 1, 'C': 0})

# -- Final feature list --------------------------------------------------------
FEATURES = [
    'Type_enc',             # 1 - Reactor technology type
    'Country_OOF',          # 2 - Country UCL tendency (OOF encoded)
    'Net_Capacity_MW',      # 3 - Plant size
    'CapCat_enc',           # 4 - <600 or >=600 MWe
    'Reactor_Age',          # 5 - Physical age (KEY predictor)
    'Operational_Years',    # 6 - Data reliability indicator
    'Age_squared',          # 7 - Non-linear aging effect
    'NSSS_Group_enc',       # 8 - NSSS technology lineage
    'Construction_Duration',# 9 - Build time (longer = more problems)
    'Thermal_Efficiency',   # 10 - Thermodynamic design quality
    'Model_Family_enc',     # 11 - Simplified design model
    'Age_Group',            # 12 - Aging zone category
    'Op_Category',          # 13 - Operational data reliability
]
FEAT_NAMES = [
    'Reactor Type', 'Country (OOF)', 'Net Capacity (MW)',
    'Capacity Category', 'Reactor Age (yr)', 'Operational Years',
    'Age² (non-linear)', 'NSSS Tech Group', 'Construction Duration (yr)',
    'Thermal Efficiency (%)', 'Model Family', 'Age Group', 'Op. Category',
]

print(f"  Features: {len(FEATURES)}")
for i, fn in enumerate(FEAT_NAMES, 1):
    print(f"    {i:2d}. {fn}")

# ==============================================================================
# STEP 3 - TRAIN / TEST SPLITS
# ==============================================================================
print("\n[3] Creating train/test splits...")

X = df[FEATURES].copy()
y = df['Target'].copy()

# Fill any remaining NaN (Python 3.13 compatible syntax)
for col in X.columns:
    if X[col].isna().any():
        X[col] = X[col].fillna(X[col].median())

scaler = StandardScaler()
Xs = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES)

# Split 1: Random 80/20
X_tr, X_te, y_tr, y_te = train_test_split(
    Xs, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

print(f"  Split 1 - Random: Train={len(X_tr)} | Test={len(X_te)}")
print(f"  Test class dist:  {dict(zip(*np.unique(y_te, return_counts=True)))}")

# Split 2: Temporal (Category A trains -> Category B+C tests)
mask_A  = df['Data_Category'].values == 'A'
mask_BC = df['Data_Category'].values != 'A'
X_trT = Xs.iloc[np.where(mask_A)[0]]
y_trT = y.iloc[np.where(mask_A)[0]]
X_teT = Xs.iloc[np.where(mask_BC)[0]]
y_teT = y.iloc[np.where(mask_BC)[0]]

print(f"  Split 2 - Temporal: Train={len(X_trT)} (Cat A) | Test={len(X_teT)} (Cat B+C)")

# ==============================================================================
# STEP 4 - SMOTE (training set only - NEVER on test set)
# ==============================================================================
if HAS_SMOTE:
    print("\n[4] Applying SMOTE to training set...")
    try:
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
        X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
        print(f"  Before: {dict(zip(*np.unique(y_tr, return_counts=True)))}")
        print(f"  After:  {dict(zip(*np.unique(y_tr_sm, return_counts=True)))}")
    except Exception as e:
        print(f"  SMOTE failed: {e} - using original training set")
        X_tr_sm, y_tr_sm = X_tr, y_tr
else:
    X_tr_sm, y_tr_sm = X_tr, y_tr
    print("\n[4] SMOTE skipped - pip install imbalanced-learn")

# ==============================================================================
# STEP 5 - DEFINE MODELS
# ==============================================================================
print("\n[5] Defining models...")

xgb_model = XGBClassifier(
    random_state=RANDOM_STATE, eval_metric='mlogloss',
    verbosity=0, n_jobs=-1
) if HAS_XGB else ExtraTreesClassifier(
    class_weight='balanced', n_estimators=200, random_state=RANDOM_STATE)

rf_base  = RandomForestClassifier(
    class_weight='balanced', n_estimators=300,
    random_state=RANDOM_STATE, n_jobs=-1)
etc_base = ExtraTreesClassifier(
    class_weight='balanced', n_estimators=300,
    random_state=RANDOM_STATE, n_jobs=-1)

# Stacking: RF + ETC + XGB as base -> LR as meta-learner
stacking = StackingClassifier(
    estimators=[
        ('rf',  copy.deepcopy(rf_base)),
        ('etc', copy.deepcopy(etc_base)),
        ('xgb', copy.deepcopy(xgb_model)),
    ],
    final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    cv=5, passthrough=False, n_jobs=-1
)

models = {
    'LR':       LogisticRegression(class_weight='balanced',
                    max_iter=2000, random_state=RANDOM_STATE),
    'RF':       copy.deepcopy(rf_base),
    'ETC':      copy.deepcopy(etc_base),
    'XGB':      copy.deepcopy(xgb_model),
    'Stacking': stacking,
}

param_grids = {
    'LR':  {'C': np.logspace(-3, 3, 20), 'solver': ['lbfgs', 'liblinear']},
    'RF':  {'n_estimators': [200, 300, 500], 'max_depth': [None, 10, 20],
             'min_samples_split': [2, 5, 10], 'max_features': ['sqrt', 'log2']},
    'ETC': {'n_estimators': [200, 300, 500], 'max_depth': [None, 10, 20],
             'min_samples_split': [2, 5, 10], 'max_features': ['sqrt', 'log2']},
    'XGB': {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7],
             'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.8, 1.0]},
    'Stacking': {},
}

# ==============================================================================
# STEP 6 - HYPERPARAMETER OPTIMISATION + TRAINING
# ==============================================================================
print(f"\n[6] HPO + Training ({HPO_ITER} iterations, F1-weighted scoring)...")

best_models = {}
results_s1  = {}
preds_s1    = {}
probas_s1   = {}

kf5 = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

for name, mdl in models.items():
    print(f"  Training {name}...", end=' ', flush=True)

    if param_grids[name]:
        rs = RandomizedSearchCV(
            copy.deepcopy(mdl), param_grids[name],
            n_iter=HPO_ITER,
            cv=kf5,
            scoring='f1_weighted',
            random_state=RANDOM_STATE,
            n_jobs=-1, verbose=0
        )
        rs.fit(X_tr_sm, y_tr_sm)
        fitted = rs.best_estimator_
    else:
        fitted = copy.deepcopy(mdl)
        fitted.fit(X_tr_sm, y_tr_sm)

    best_models[name] = fitted

    # Evaluate on held-out test set
    yp   = fitted.predict(X_te)
    ypr  = fitted.predict_proba(X_te)
    cvs  = cross_val_score(fitted, X_tr_sm, y_tr_sm, cv=kf5, scoring='accuracy')

    acc = accuracy_score(y_te, yp) * 100
    f1  = f1_score(y_te, yp, average='weighted', zero_division=0) * 100
    try:
        auc_val = roc_auc_score(y_te, ypr, multi_class='ovr', average='weighted') * 100
    except Exception:
        auc_val = 0.0

    results_s1[name] = {
        'Accuracy': round(acc, 2), 'F1': round(f1, 2),
        'AUC_ROC': round(auc_val, 2),
        'CV_Mean': round(cvs.mean() * 100, 2),
        'CV_Std':  round(cvs.std()  * 100, 2),
    }
    preds_s1[name]  = yp
    probas_s1[name] = ypr

    print(f"Acc={acc:.1f}%  F1={f1:.1f}%  AUC={auc_val:.1f}%  "
          f"CV={cvs.mean()*100:.1f}%+/-{cvs.std()*100:.1f}%")

df_s1 = pd.DataFrame(results_s1).T.reset_index().rename(columns={'index': 'Model'})
best_name = df_s1.loc[df_s1['Accuracy'].idxmax(), 'Model']
best_model = best_models[best_name]

print(f"\n  * BEST MODEL: {best_name} - "
      f"Acc={results_s1[best_name]['Accuracy']:.1f}%  "
      f"AUC={results_s1[best_name]['AUC_ROC']:.1f}%")

# ==============================================================================
# STEP 7 - TEMPORAL SPLIT VALIDATION
# ==============================================================================
print("\n[7] Temporal split validation (Cat A -> Cat B+C)...")

if HAS_SMOTE:
    try:
        smote_t = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
        X_trT_sm, y_trT_sm = smote_t.fit_resample(X_trT, y_trT)
    except Exception:
        X_trT_sm, y_trT_sm = X_trT, y_trT
else:
    X_trT_sm, y_trT_sm = X_trT, y_trT

results_temp, preds_temp = {}, {}
for name, mdl in models.items():
    m = copy.deepcopy(mdl)
    m.fit(X_trT_sm, y_trT_sm)
    yp_t = m.predict(X_teT)
    acc_t = accuracy_score(y_teT, yp_t) * 100
    results_temp[name] = round(acc_t, 2)
    preds_temp[name]   = yp_t
    delta = acc_t - results_s1[name]['Accuracy']
    print(f"  {name:10s}: {results_s1[name]['Accuracy']:.1f}% -> {acc_t:.1f}%  "
          f"({delta:+.1f}pp) {'up' if delta > 0 else 'down'}")

# ==============================================================================
# STEP 8 - PER-CLASS PERFORMANCE (best model)
# ==============================================================================
print(f"\n[8] Per-class performance ({best_name})...")
report = classification_report(
    y_te, preds_s1[best_name],
    target_names=CLASS_LIST, output_dict=True
)
perclass = []
for cls in CLASS_LIST:
    r = report[cls]
    perclass.append({
        'Class': cls, 'Support': int(r['support']),
        'Precision': round(r['precision'] * 100, 1),
        'Recall':    round(r['recall']    * 100, 1),
        'F1':        round(r['f1-score']  * 100, 1),
    })
    flag = "(!) WEAK" if cls == 'Medium' else "OK"
    print(f"  {flag} {cls:8s}: P={r['precision']*100:.1f}%  "
          f"R={r['recall']*100:.1f}%  F1={r['f1-score']*100:.1f}%  "
          f"(n={int(r['support'])})")

# ==============================================================================
# STEP 9 - BOOTSTRAP UNCERTAINTY (locked to Split 1 best model)
# ==============================================================================
print(f"\n[9] Bootstrap uncertainty ({N_BOOTSTRAP} iterations)...")

boot_acc, boot_f1, boot_auc = [], [], []
for i in range(N_BOOTSTRAP):
    idx  = resample(range(len(X_te)), random_state=i)
    Xb   = X_te.iloc[idx]; yb = y_te.iloc[idx]
    yp_b = best_model.predict(Xb)
    ypr_b= best_model.predict_proba(Xb)
    boot_acc.append(accuracy_score(yb, yp_b) * 100)
    boot_f1.append(f1_score(yb, yp_b, average='weighted', zero_division=0) * 100)
    try:
        boot_auc.append(
            roc_auc_score(yb, ypr_b, multi_class='ovr', average='weighted') * 100)
    except Exception:
        pass

df_boot = pd.DataFrame([
    {'Metric': m, 'Mean': round(np.mean(v), 2), 'Std': round(np.std(v), 2),
     'CI_Low': round(np.percentile(v, 2.5), 1),
     'CI_High': round(np.percentile(v, 97.5), 1)}
    for m, v in [('Accuracy', boot_acc), ('F1', boot_f1), ('AUC', boot_auc)]
])
for _, row in df_boot.iterrows():
    flag = "OK" if row['Std'] <= 5 else "(!)"
    print(f"  {row['Metric']:10s}: {row['Mean']:.1f}%+/-{row['Std']:.1f}%  "
          f"95%CI:[{row['CI_Low']:.1f}%, {row['CI_High']:.1f}%] {flag}")

# ==============================================================================
# STEP 10 - FIGURES
# ==============================================================================
print("\n[10] Generating figures...")

# Fig 1 - Model Comparison Bar Chart
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Model 1 - Design Classifier: All Models Comparison',
             fontsize=13, fontweight='bold')
nms  = list(results_s1.keys())
xpos = np.arange(len(nms))
w    = 0.6
colors_bar = [DARK, MID, '#4A90D9', '#2B5E38', '#7A1C1C']

for ax, metric in zip(axes, ['Accuracy', 'F1', 'AUC_ROC']):
    vals = [results_s1[n][metric] for n in nms]
    bars = ax.bar(xpos, vals, w, color=colors_bar[:len(nms)], alpha=0.88, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', fontsize=8.5, fontweight='bold')
    ax.set_xticks(xpos); ax.set_xticklabels(nms, rotation=20, fontsize=9)
    ax.set_ylim(0, 105); ax.set_ylabel('Score (%)'); ax.set_title(metric)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
save_fig(fig, 'M1_fig01_model_comparison')

# Fig 2 - Confusion Matrices
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle('Model 1 - Confusion Matrices (All Models)', fontsize=13, fontweight='bold')
from sklearn.metrics import confusion_matrix as cm_fn
import seaborn as sns
for ax, name in zip(axes.flatten(), list(models.keys())):
    cm = cm_fn(y_te, preds_s1[name])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'Med', 'High'],
                yticklabels=['Low', 'Med', 'High'],
                linewidths=0.5, ax=ax, cbar=False)
    ax.set_title(f"{name} - {results_s1[name]['Accuracy']:.1f}%", fontsize=10)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
axes.flatten()[-1].set_visible(False)
save_fig(fig, 'M1_fig02_confusion_matrices')

# Fig 3 - Per-Class Performance
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle(f'Model 1 - Per-Class Performance ({best_name})', fontsize=13, fontweight='bold')
xc = np.arange(3); wc = 0.25
ax.bar(xc - wc, [r['Precision'] for r in perclass], wc, label='Precision', color=DARK, alpha=0.9)
ax.bar(xc,      [r['Recall']    for r in perclass], wc, label='Recall',    color=MID,  alpha=0.9)
ax.bar(xc + wc, [r['F1']        for r in perclass], wc, label='F1-Score',  color='#4A90D9', alpha=0.9)
ax.set_xticks(xc); ax.set_xticklabels(CLASS_LIST)
ax.set_ylim(0, 115); ax.set_ylabel('Score (%)')
ax.axhline(50, color='grey', ls='--', lw=0.8, alpha=0.5)
ax.legend(fontsize=9)
ax.annotate('Medium class weak\n(14.3% of data)',
            xy=(1, perclass[1]['F1']), xytext=(1.4, perclass[1]['F1'] + 20),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=8.5, color='red')
save_fig(fig, 'M1_fig03_per_class_performance')

# Fig 4 - Feature Importance
fig, ax = plt.subplots(figsize=(11, 6))
fig.suptitle(f'Model 1 - Feature Importance ({best_name})', fontsize=13, fontweight='bold')
try:
    if hasattr(best_models.get(best_name, best_model), 'feature_importances_'):
        fi = best_model.feature_importances_
    else:
        # For Stacking, use internal RF importance
        fi_mdl = best_models.get('RF', best_models.get('ETC'))
        fi = fi_mdl.feature_importances_
    idx = np.argsort(fi)
    ax.barh([FEAT_NAMES[i] for i in idx], fi[idx], color=DARK, alpha=0.85, edgecolor='white')
    for i, val in enumerate(fi[idx]):
        ax.text(val + 0.001, i, f'{val*100:.1f}%', va='center', fontsize=8.5)
    ax.set_xlabel('Feature Importance Score')
    ax.axvline(np.mean(fi), color='red', ls='--', lw=1, alpha=0.6, label='Mean')
    ax.legend(fontsize=8)
except Exception as e:
    ax.text(0.5, 0.5, f'Feature importance\nnot available:\n{e}',
            ha='center', va='center', transform=ax.transAxes)
save_fig(fig, 'M1_fig04_feature_importance')

# Fig 5 - ROC Curves (best model)
fig, ax = plt.subplots(figsize=(7, 6))
fig.suptitle(f'Model 1 - ROC Curves ({best_name})', fontsize=13, fontweight='bold')
for i, (cls, col) in enumerate(CLASS_COLORS.items()):
    fpr, tpr, _ = roc_curve((y_te == i).astype(int), probas_s1[best_name][:, i])
    ax.plot(fpr, tpr, color=col, lw=2.5, label=f'{cls} (AUC={sk_auc(fpr,tpr):.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
save_fig(fig, 'M1_fig05_roc_curves')

# Fig 6 - Bootstrap Uncertainty
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f'Model 1 - Bootstrap Uncertainty ({N_BOOTSTRAP} iterations)',
             fontsize=13, fontweight='bold')
for ax, (metric, vals) in zip(axes, [('Accuracy', boot_acc), ('F1', boot_f1), ('AUC', boot_auc)]):
    vals = np.array(vals)
    ax.hist(vals, bins=35, color=DARK, edgecolor='white', alpha=0.82)
    ci_l, ci_h = np.percentile(vals, [2.5, 97.5])
    ax.axvline(np.mean(vals), color='red',    lw=2,   label=f'Mean={np.mean(vals):.1f}%')
    ax.axvline(ci_l,          color='orange', lw=1.5, ls='--')
    ax.axvline(ci_h,          color='orange', lw=1.5, ls='--',
               label=f'95%CI [{ci_l:.1f}, {ci_h:.1f}]')
    ax.set_xlabel(f'{metric} (%)'); ax.set_title(f'{metric} | Std={np.std(vals):.1f}%')
    ax.legend(fontsize=7.5)
save_fig(fig, 'M1_fig06_bootstrap_uncertainty')

# Fig 7 - Temporal vs Random Split
fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle('Model 1 - Random Split vs Temporal Split Accuracy',
             fontsize=13, fontweight='bold')
xpos = np.arange(len(nms)); w = 0.35
rand_accs = [results_s1[n]['Accuracy'] for n in nms]
temp_accs = [results_temp[n] for n in nms]
ax.bar(xpos - w/2, rand_accs, w, label='Split 1 - Random 80/20', color=DARK, alpha=0.87)
ax.bar(xpos + w/2, temp_accs, w, label='Split 2 - Temporal (Cat A->B+C)', color='#2B5E38', alpha=0.87)
for xi, (r, t) in enumerate(zip(rand_accs, temp_accs)):
    ax.text(xi - w/2, r + 0.5, f'{r:.1f}', ha='center', fontsize=8)
    ax.text(xi + w/2, t + 0.5, f'{t:.1f}', ha='center', fontsize=8)
ax.set_xticks(xpos); ax.set_xticklabels(nms)
ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 105)
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.25, linestyle='--')
ax.text(0.5, 0.92,
        f'Temporal: Train={len(X_trT)} (Cat A) | Test={len(X_teT)} (Cat B+C)',
        transform=ax.transAxes, ha='center', fontsize=9, style='italic', color='grey')
save_fig(fig, 'M1_fig07_temporal_vs_random')

# ==============================================================================
# STEP 11 - SAVE TABLES
# ==============================================================================
print("\n[11] Saving tables...")
df_s1.to_csv(f'{OUT}/M1_table1_split1_results.csv', index=False)
pd.DataFrame(perclass).to_csv(f'{OUT}/M1_table2_per_class.csv', index=False)
df_boot.to_csv(f'{OUT}/M1_table3_bootstrap.csv', index=False)
pd.DataFrame([{'Model': n, 'Temporal_Accuracy': v}
              for n, v in results_temp.items()]
    ).to_csv(f'{OUT}/M1_table4_temporal.csv', index=False)

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
best_row = df_s1[df_s1['Model'] == best_name].iloc[0]
print("\n" + "=" * 65)
print("  MODEL 1 - FINAL RESULTS SUMMARY")
print("=" * 65)
print(f"\n  Dataset:    {len(df)} reactors | {df['Country'].nunique()} countries")
print(f"  Features:   {len(FEATURES)} (6 original + 7 engineered)")
print(f"\n  SPLIT 1 - Random 80/20:")
for _, row in df_s1.iterrows():
    star = "*" if row['Model'] == best_name else " "
    print(f"    {star} {row['Model']:12s}: Acc={row['Accuracy']:.1f}%  "
          f"F1={row['F1']:.1f}%  AUC={row['AUC_ROC']:.1f}%  "
          f"CV={row['CV_Mean']:.1f}%+/-{row['CV_Std']:.1f}%")

print(f"\n  TEMPORAL SPLIT (Cat A->B+C):")
for n, v in results_temp.items():
    delta = v - results_s1[n]['Accuracy']
    print(f"    {n:12s}: {results_s1[n]['Accuracy']:.1f}% -> {v:.1f}%  ({delta:+.1f}pp)")

print(f"\n  PER-CLASS ({best_name}):")
for r in perclass:
    flag = "(!) WEAK (Medium class, see Model 2)" \
           if r['Class'] == 'Medium' else "OK"
    print(f"    {flag} {r['Class']:8s}: F1={r['F1']:.1f}%  (n={r['Support']})")

print(f"\n  BOOTSTRAP 95% CI ({best_name}, n={N_BOOTSTRAP}):")
for _, row in df_boot.iterrows():
    print(f"    {row['Metric']:10s}: {row['Mean']:.1f}%+/-{row['Std']:.1f}%  "
          f"[{row['CI_Low']:.1f}%, {row['CI_High']:.1f}%]")

print(f"\n  Output: {OUT}/")
print(f"  Figures: 7 PNG  |  Tables: 4 CSV")
print("=" * 65)
print("\n  -> Run UCL_Model2_Pipeline.py next to see what adding the two UCF")
print("    history features does for the Medium class.")
