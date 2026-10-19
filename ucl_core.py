"""
ucl_core.py
Shared feature-engineering code for both UCL models.

Everything in here is copied (logic unchanged) from the verified research
pipelines UCL_Model1_Pipeline.py and UCL_Model2_Pipeline.py, so the website
builds its inputs exactly the way the paper's models were trained.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42
CLASS_LIST = ["Low", "Medium", "High"]

NSSS_GROUP_MAP = {
    'WH': 'Western_PWR', 'CE': 'Western_PWR', 'B&W': 'Western_PWR',
    'WH/MHI': 'Western_PWR', 'ACECOWE': 'Western_PWR', 'ACLF': 'Western_PWR',
    'FAEA': 'Western_PWR',
    'FRAM': 'French_PWR', 'ORANO': 'French_PWR', 'KWU': 'French_PWR',
    'SIEMENS': 'French_PWR', 'S/KWU': 'French_PWR', 'GETSCO': 'French_PWR',
    'GE': 'GE_BWR', 'HITACHI': 'GE_BWR', 'TOSHIBA': 'GE_BWR',
    'ABB ATOM': 'GE_BWR', 'ASEASTAL': 'GE_BWR',
    'AEM': 'Russian_VVER', 'JSC ASE': 'Russian_VVER', 'IZ': 'Russian_VVER',
    'MAEP': 'Russian_VVER', 'AEE': 'Russian_VVER', 'PAIP': 'Russian_VVER',
    'PAA': 'Russian_VVER', 'SKODA': 'Russian_VVER', 'CZEC': 'Russian_VVER',
    'CNNC': 'Chinese', 'DEC': 'Chinese', 'CFHI': 'Chinese',
    'NPIC': 'Chinese', 'SHE': 'Chinese',
    'AECL': 'Canadian_CANDU', 'OH/AECL': 'Canadian_CANDU',
    'AECL/DHI': 'Canadian_CANDU', 'AECL/DAE': 'Canadian_CANDU', 'NPCIL': 'Canadian_CANDU',
    'DHICKOPC': 'Korean_Other', 'DHICKAEC': 'Korean_Other', 'KEPCO': 'Korean_Other',
    'MHI': 'Korean_Other', 'NPC': 'Korean_Other', 'NNC': 'Korean_Other',
    'PPC': 'Korean_Other',
}

FEAT_M1 = [
    'Type_enc', 'Country_OOF', 'Net_Capacity_MW', 'CapCat_enc',
    'Reactor_Age', 'Operational_Years', 'Age_squared', 'NSSS_Group_enc',
    'Construction_Duration', 'Thermal_Efficiency', 'Model_Family_enc',
    'Age_Group', 'Op_Category',
]
FEAT_TEMPORAL = ['UCF_P1', 'UCF_P2', 'UCF_change_P1_P2',
                 'UCF_mean_2period', 'UCF_trend_dir']
FEAT_M2 = FEAT_M1 + FEAT_TEMPORAL


def model_family(m, phwr_is_candu=True):
    """Same rules as the pipelines (Model 1 version treats PHWR as CANDU)."""
    m = str(m).upper()
    if 'VVER' in m or 'WWER' in m:
        return 'VVER'
    if 'BWR' in m:
        return 'BWR'
    if 'CANDU' in m or (phwr_is_candu and 'PHWR' in m):
        return 'CANDU'
    if 'AP-1000' in m or 'AP1000' in m:
        return 'AP1000'
    if 'EPR' in m:
        return 'EPR'
    if 'AGR' in m or 'GCR' in m:
        return 'GCR'
    if 'BN' in m or 'FBR' in m:
        return 'FBR'
    return 'Other'


def age_group(age):
    if age <= 10:
        return 0
    if age <= 25:
        return 1
    if age <= 40:
        return 2
    return 3


def trend_direction(change):
    if change > 2:
        return 1
    if change < -2:
        return -1
    return 0


def load_final(xlsx_path):
    """Load and clean the FINAL_DATASET sheet exactly like the pipelines do."""
    df_raw = pd.read_excel(xlsx_path, sheet_name='FINAL_DATASET', header=0)
    df_raw.columns = [str(c).split('\n')[0].strip() for c in df_raw.columns]
    df = df_raw[df_raw['Reactor_Name'].notna() &
                df_raw['UCL_Class'].isin(CLASS_LIST)].copy().reset_index(drop=True)
    df['Reactor_Name'] = df['Reactor_Name'].astype(str).str.strip()
    for col in ['Country', 'Type', 'NSSS_Supplier', 'Reactor_Model']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
    for col in ['Net_Capacity_MW', 'Reactor_Age', 'Operational_Years',
                'Construction_Duration', 'First_Grid_Year']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Net_Capacity_MW'] = df['Net_Capacity_MW'].fillna(df['Net_Capacity_MW'].median())
    df['Reactor_Age'] = df['Reactor_Age'].fillna(df['Reactor_Age'].median())
    df['Operational_Years'] = df['Operational_Years'].fillna(10)
    df['Construction_Duration'] = df['Construction_Duration'].fillna(
        df['Construction_Duration'].median())
    if 'Capacity_Category' not in df.columns:
        df['Capacity_Category'] = df['Net_Capacity_MW'].apply(
            lambda x: '<600 MWe' if x < 600 else '>=600 MWe')
    df['Data_Category'] = 'A'
    df.loc[df['Operational_Years'] < 10, 'Data_Category'] = 'B'
    df.loc[df['Operational_Years'] < 5, 'Data_Category'] = 'C'
    return df


def add_longitudinal(df, xlsx_path):
    """Attach UCF 2013-2017 and 2010-2020 (never 2014-2024: that is the target window)."""
    lg = pd.read_excel(xlsx_path, sheet_name='UCL_LONGITUDAL', header=0)
    lg['Reactor_Name'] = lg['Reactor_Name'].astype(str).str.strip()
    out = df.merge(lg[['Reactor_Name', 'UCF_2013_2017', 'UCF_2010_2020']],
                   on='Reactor_Name', how='left')
    return out


def engineer(df, thermal='constant', phwr_is_candu=True, temporal=False):
    """
    Build the model features in place on a copy of df.
    Returns (df, info) where info holds everything needed to encode a new reactor.
    thermal='constant' reproduces the pipelines (no Thermal_Capacity_MW column
    in FINAL_DATASET, so the feature is the constant 33.0).
    """
    df = df.copy().reset_index(drop=True)
    df['Target'] = df['UCL_Class'].map({'Low': 0, 'Medium': 1, 'High': 2})

    le_type = LabelEncoder()
    df['Type_enc'] = le_type.fit_transform(df['Type'])

    kf_oof = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    df['Country_OOF'] = np.nan
    for tr_idx, val_idx in kf_oof.split(df, df['Target']):
        fold_mean = df.iloc[tr_idx].groupby('Country')['Target'].mean()
        df.iloc[val_idx, df.columns.get_loc('Country_OOF')] = \
            df.iloc[val_idx]['Country'].map(fold_mean)
    global_mean = df.groupby('Country')['Target'].mean()
    df['Country_OOF'] = df['Country_OOF'].fillna(df['Country'].map(global_mean))

    le_cap = LabelEncoder()
    df['CapCat_enc'] = le_cap.fit_transform(df['Capacity_Category'])
    df['Age_squared'] = df['Reactor_Age'] ** 2

    df['NSSS_Group'] = df['NSSS_Supplier'].map(NSSS_GROUP_MAP).fillna('Other')
    le_nsss = LabelEncoder()
    df['NSSS_Group_enc'] = le_nsss.fit_transform(df['NSSS_Group'])

    if thermal == 'constant':
        df['Thermal_Efficiency'] = 33.0
    else:
        df['Thermal_Efficiency'] = pd.to_numeric(
            df['Thermal_Efficiency_Pct'], errors='coerce').clip(10, 50).fillna(33.0)

    df['Model_Family'] = df['Reactor_Model'].apply(
        lambda m: model_family(m, phwr_is_candu))
    le_mf = LabelEncoder()
    df['Model_Family_enc'] = le_mf.fit_transform(df['Model_Family'])

    df['Age_Group'] = df['Reactor_Age'].apply(age_group)
    df['Op_Category'] = df['Data_Category'].map({'A': 2, 'B': 1, 'C': 0})

    feats = list(FEAT_M1)
    if temporal:
        df['UCF_P1'] = df['UCF_2013_2017']
        df['UCF_P2'] = df['UCF_2010_2020']
        df['UCF_change_P1_P2'] = df['UCF_P2'] - df['UCF_P1']
        df['UCF_mean_2period'] = (df['UCF_P1'] + df['UCF_P2']) / 2
        df['UCF_trend_dir'] = df['UCF_change_P1_P2'].apply(trend_direction)
        feats = list(FEAT_M2)

    info = {
        'features': feats,
        'type_classes': list(le_type.classes_),
        'capcat_classes': list(le_cap.classes_),
        'nsss_classes': list(le_nsss.classes_),
        'family_classes': list(le_mf.classes_),
        'country_mean': {k: float(v) for k, v in global_mean.items()},
        'overall_country_mean': float(df['Target'].mean()),
    }
    return df, info


def design_row(inp, info, temporal=False):
    """
    Turn a plain-language form input into the model's feature dictionary.
    inp keys: type, country, net_capacity_mw, age_years, nsss_group,
              construction_years, model_family, (ucf_p1, ucf_p2 for Model 2)
    """
    age = float(inp['age_years'])
    cap = float(inp['net_capacity_mw'])
    op_years = min(10.0, age)
    if op_years >= 10:
        op_cat = 2
    elif op_years >= 5:
        op_cat = 1
    else:
        op_cat = 0
    country = inp['country'].upper()
    row = {
        'Type_enc': info['type_classes'].index(inp['type']),
        'Country_OOF': info['country_mean'].get(country, info['overall_country_mean']),
        'Net_Capacity_MW': cap,
        'CapCat_enc': info['capcat_classes'].index('<600 MWe' if cap < 600 else '>=600 MWe'),
        'Reactor_Age': age,
        'Operational_Years': op_years,
        'Age_squared': age ** 2,
        'NSSS_Group_enc': info['nsss_classes'].index(inp['nsss_group']),
        'Construction_Duration': float(inp['construction_years']),
        'Thermal_Efficiency': 33.0,
        'Model_Family_enc': info['family_classes'].index(inp['model_family']),
        'Age_Group': age_group(age),
        'Op_Category': op_cat,
    }
    if temporal:
        p1, p2 = float(inp['ucf_p1']), float(inp['ucf_p2'])
        ch = p2 - p1
        row.update({'UCF_P1': p1, 'UCF_P2': p2, 'UCF_change_P1_P2': ch,
                    'UCF_mean_2period': (p1 + p2) / 2,
                    'UCF_trend_dir': trend_direction(ch)})
    return row
