# ================================================================
# UCL RISK PREDICTOR — Nuclear Power Plant Dashboard
# ================================================================
# INSTALL:
#   pip install dash dash-bootstrap-components plotly pandas
#              numpy scikit-learn xgboost imbalanced-learn openpyxl
#
# RUN:
#   1. Put this file AND UCL_Dataset_v5_longitudinal.xlsx
#      in the same folder (IAEA_PRIS folder)
#   2. Open cmd in that folder
#   3. python ucl_predictor_app.py
#   4. Open browser → http://127.0.0.1:8050
# ================================================================

import pandas as pd
import numpy as np
import warnings, os, sys
warnings.filterwarnings('ignore')

# ── Train models on startup ──────────────────────────────────────
print("Loading pre-trained models...")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load pre-trained models from pickle file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, 'models.pkl')

if not os.path.exists(MODEL_FILE):
    # Fallback: try to train on the fly
    print("models.pkl not found, training from scratch...")
    MODEL_FILE = None

if MODEL_FILE:
    with open(MODEL_FILE, 'rb') as f:
        _M = pickle.load(f)
    rf_m1 = _M['rf_m1']; rf_m2 = _M['rf_m2']
    sc1 = _M['sc1']; sc2 = _M['sc2']
    FEAT_M1 = _M['FEAT_M1']; FEAT_M2 = _M['FEAT_M2']
    le_type = _M['le_type']; le_nsss = _M['le_nsss']
    le_cap = _M['le_cap']; le_mfam = _M['le_mfam']
    le_agegrp = _M['le_agegrp']; le_opcat = _M['le_opcat']
    country_mean = _M['country_mean']
    COUNTRIES = _M['countries_list']; TYPES = _M['types_list']
    NSSS_GROUPS = _M['nsss_grp_list']; MDL_FAMS = _M['mdl_fam_list']
    CAP_CATS = _M['cap_cats']; AGE_GRPS = _M['age_grps']; OP_CATS = _M['op_cats']
    df = None  # Not needed when using pre-trained models

print("Models ready. Starting dashboard...")

# ── Dropdown lists — loaded from pickle, not from df ─────────────
COUNTRIES   = M['countries_list']
TYPES       = M['types_list']
NSSS_GROUPS = M['nsss_grp_list']
MDL_FAMS    = M['mdl_fam_list']
CAP_CATS    = M['cap_cats']
AGE_GRPS    = M['age_grps']
OP_CATS     = M['op_cats']

# ── Dash App ─────────────────────────────────────────────────────
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.SLATE],
                suppress_callback_exceptions=True)
app.title = "UCL Risk Predictor — Nuclear Power Plant"
server = app.server  # Required for Render deployment

RISK_COLORS = {'Low':'#00C853','Medium':'#FFD600','High':'#FF1744'}
RISK_ICONS  = {'Low':'🟢','Medium':'🟡','High':'🔴'}
RISK_DESC   = {
    'Low':    'Reactor demonstrates excellent reliability. Unplanned losses below 2% indicate minimal unexpected outages and strong operational management.',
    'Medium': 'Reactor shows moderate unplanned losses (2–6%). Operational attention and proactive maintenance recommended.',
    'High':   'Reactor exhibits significant unplanned capability losses above 6%. Operational review and targeted maintenance intervention advised.'
}

# ── LAYOUT ───────────────────────────────────────────────────────
app.layout = dbc.Container(fluid=True, style={'backgroundColor':'#0d1117','minHeight':'100vh'}, children=[

    # Header
    html.Div([
        html.Div([
            html.Span("☢", style={'fontSize':'2rem','marginRight':'12px'}),
            html.Span("UCL Risk Predictor", style={
                'fontSize':'1.8rem','fontWeight':'700','color':'#E8EDF2',
                'fontFamily':'Georgia, serif','letterSpacing':'0.5px'}),
        ], style={'display':'flex','alignItems':'center'}),
        html.P("Nuclear Power Plant · Unplanned Capability Loss Classification · IAEA RDS-2 Data",
               style={'color':'#8B949E','margin':'4px 0 0 44px','fontSize':'0.85rem',
                      'fontFamily':'monospace','letterSpacing':'1px'}),
    ], style={'background':'linear-gradient(90deg,#161b22,#0d1117)',
              'borderBottom':'1px solid #30363d','padding':'20px 32px','marginBottom':'24px'}),

    dbc.Row([

        # ── LEFT PANEL: INPUTS ───────────────────────────────────
        dbc.Col(width=4, children=[

            # Model selector
            html.Div([
                html.Label("PREDICTION MODE", style={'color':'#58A6FF','fontSize':'0.7rem',
                           'fontWeight':'700','letterSpacing':'2px','fontFamily':'monospace'}),
                dcc.RadioItems(
                    id='model-mode',
                    options=[
                        {'label': html.Span([
                            html.Span("Model 1", style={'fontWeight':'600','color':'#E8EDF2'}),
                            html.Span(" — Design features only (79.3% acc)",
                                      style={'color':'#8B949E','fontSize':'0.8rem'})
                        ]), 'value': 'M1'},
                        {'label': html.Span([
                            html.Span("Model 2", style={'fontWeight':'600','color':'#3FB950'}),
                            html.Span(" — Design + UCF history (94.6% acc)",
                                      style={'color':'#8B949E','fontSize':'0.8rem'})
                        ]), 'value': 'M2'},
                    ],
                    value='M2',
                    style={'color':'#E8EDF2','marginTop':'8px'},
                    labelStyle={'display':'block','padding':'6px 0','cursor':'pointer'},
                ),
            ], style={'background':'#161b22','border':'1px solid #30363d',
                      'borderRadius':'8px','padding':'16px','marginBottom':'16px'}),

            # Design Features card
            html.Div([
                html.Label("REACTOR DESIGN FEATURES",
                           style={'color':'#58A6FF','fontSize':'0.7rem',
                                  'fontWeight':'700','letterSpacing':'2px','fontFamily':'monospace',
                                  'marginBottom':'12px','display':'block'}),
                # Country
                html.Label("Country", style={'color':'#8B949E','fontSize':'0.8rem'}),
                dcc.Dropdown(id='inp-country', options=[{'label':c,'value':c} for c in COUNTRIES],
                             value='USA', clearable=False,
                             style={'backgroundColor':'#21262d','color':'#E8EDF2','border':'1px solid #30363d'}),
                html.Br(),
                # Type
                dbc.Row([
                    dbc.Col([
                        html.Label("Reactor Type", style={'color':'#8B949E','fontSize':'0.8rem'}),
                        dcc.Dropdown(id='inp-type', options=[{'label':t,'value':t} for t in TYPES],
                                     value='PWR', clearable=False,
                                     style={'backgroundColor':'#21262d','border':'1px solid #30363d'}),
                    ], width=6),
                    dbc.Col([
                        html.Label("NSSS Tech Group", style={'color':'#8B949E','fontSize':'0.8rem'}),
                        dcc.Dropdown(id='inp-nsss', options=[{'label':n,'value':n} for n in NSSS_GROUPS],
                                     value='Western_PWR', clearable=False,
                                     style={'backgroundColor':'#21262d','border':'1px solid #30363d'}),
                    ], width=6),
                ], className='mb-2'),
                # Capacity
                html.Label("Net Capacity (MWe)", style={'color':'#8B949E','fontSize':'0.8rem'}),
                dcc.Slider(id='inp-capacity', min=100, max=1700, step=50, value=1000,
                           marks={100:'100',500:'500',1000:'1000',1500:'1500',1700:'1700'},
                           tooltip={'always_visible':True}),
                # Age
                html.Label("Reactor Age (years)", style={'color':'#8B949E','fontSize':'0.8rem','marginTop':'8px','display':'block'}),
                dcc.Slider(id='inp-age', min=0, max=60, step=1, value=30,
                           marks={0:'0',15:'15',30:'30',45:'45',60:'60'},
                           tooltip={'always_visible':True}),
                # Construction Duration
                html.Label("Construction Duration (years)", style={'color':'#8B949E','fontSize':'0.8rem','marginTop':'8px','display':'block'}),
                dcc.Slider(id='inp-constdur', min=2, max=40, step=1, value=8,
                           marks={2:'2',10:'10',20:'20',30:'30',40:'40'},
                           tooltip={'always_visible':True}),
                # Thermal Efficiency
                html.Label("Thermal Efficiency (%)", style={'color':'#8B949E','fontSize':'0.8rem','marginTop':'8px','display':'block'}),
                dcc.Slider(id='inp-thermal', min=22, max=40, step=0.5, value=33,
                           marks={22:'22%',28:'28%',33:'33%',38:'38%',40:'40%'},
                           tooltip={'always_visible':True}),
                # Capacity category
                dbc.Row([
                    dbc.Col([
                        html.Label("Capacity Category", style={'color':'#8B949E','fontSize':'0.8rem'}),
                        dcc.Dropdown(id='inp-capcat',
                                     options=[{'label':'< 600 MWe','value':'small'},
                                              {'label':'≥ 600 MWe','value':'large'}],
                                     value='large', clearable=False,
                                     style={'backgroundColor':'#21262d','border':'1px solid #30363d'}),
                    ], width=6),
                    dbc.Col([
                        html.Label("Reactor Model Family", style={'color':'#8B949E','fontSize':'0.8rem'}),
                        dcc.Dropdown(id='inp-mdlfam',
                                     options=[{'label':m,'value':m} for m in MDL_FAMS],
                                     value='WH_PWR' if 'WH_PWR' in MDL_FAMS else MDL_FAMS[0],
                                     clearable=False,
                                     style={'backgroundColor':'#21262d','border':'1px solid #30363d'}),
                    ], width=6),
                ], className='mt-2'),
            ], style={'background':'#161b22','border':'1px solid #30363d',
                      'borderRadius':'8px','padding':'16px','marginBottom':'16px'}),

            # UCF History card
            html.Div(id='ucf-panel', children=[
                html.Label("UCF HISTORY (MODEL 2 ONLY)",
                           style={'color':'#3FB950','fontSize':'0.7rem',
                                  'fontWeight':'700','letterSpacing':'2px','fontFamily':'monospace',
                                  'marginBottom':'12px','display':'block'}),
                html.Label("UCF 2013–2017 (%)", style={'color':'#8B949E','fontSize':'0.8rem'}),
                dcc.Slider(id='inp-ucf1', min=0, max=100, step=0.5, value=88,
                           marks={0:'0',25:'25',50:'50',75:'75',100:'100'},
                           tooltip={'always_visible':True}),
                html.Label("UCF 2010–2020 (%)", style={'color':'#8B949E','fontSize':'0.8rem',
                                                         'marginTop':'8px','display':'block'}),
                dcc.Slider(id='inp-ucf2', min=0, max=100, step=0.5, value=87,
                           marks={0:'0',25:'25',50:'50',75:'75',100:'100'},
                           tooltip={'always_visible':True}),
                html.P("UCF 2014–2024 is set automatically from the design features.",
                       style={'color':'#6E7681','fontSize':'0.75rem','marginTop':'8px'}),
            ], style={'background':'#161b22','border':'1px solid #238636',
                      'borderRadius':'8px','padding':'16px','marginBottom':'16px'}),

            # Predict button
            html.Button("PREDICT UCL RISK",
                        id='btn-predict', n_clicks=0,
                        style={'width':'100%','padding':'14px','fontSize':'0.95rem',
                               'fontWeight':'700','letterSpacing':'2px','fontFamily':'monospace',
                               'background':'linear-gradient(135deg,#1f6feb,#388bfd)',
                               'border':'none','borderRadius':'8px','color':'white',
                               'cursor':'pointer','transition':'all 0.2s'}),
        ]),

        # ── RIGHT PANEL: RESULTS ─────────────────────────────────
        dbc.Col(width=8, children=[
            html.Div(id='result-panel', children=[
                # Default placeholder
                html.Div([
                    html.Div("⚛", style={'fontSize':'4rem','textAlign':'center',
                                          'opacity':'0.15','marginBottom':'16px'}),
                    html.P("Configure reactor parameters and click PREDICT",
                           style={'color':'#6E7681','textAlign':'center',
                                  'fontFamily':'monospace','letterSpacing':'1px'}),
                ], style={'display':'flex','flexDirection':'column','justifyContent':'center',
                          'alignItems':'center','height':'500px'}),
            ]),
        ]),
    ], className='g-3 px-3'),

    # Footer
    html.Div([
        html.P("Based on IAEA RDS-2 data (2018, 2021, 2025) · 407 operational reactors · 32 countries · "
               "Nuclear Engineering and Design Research Project",
               style={'color':'#6E7681','textAlign':'center','fontSize':'0.75rem',
                      'fontFamily':'monospace','margin':'0'})
    ], style={'borderTop':'1px solid #21262d','padding':'16px','marginTop':'32px'}),
])


# ── CALLBACKS ────────────────────────────────────────────────────

@app.callback(Output('ucf-panel','style'), Input('model-mode','value'))
def toggle_ucf_panel(mode):
    base = {'background':'#161b22','borderRadius':'8px','padding':'16px','marginBottom':'16px'}
    if mode == 'M2':
        return {**base, 'border':'1px solid #238636'}
    return {**base, 'border':'1px solid #30363d', 'opacity':'0.4', 'pointerEvents':'none'}


@app.callback(
    Output('result-panel','children'),
    Input('btn-predict','n_clicks'),
    [State('model-mode','value'),
     State('inp-country','value'), State('inp-type','value'),
     State('inp-nsss','value'),    State('inp-capacity','value'),
     State('inp-age','value'),     State('inp-constdur','value'),
     State('inp-thermal','value'), State('inp-capcat','value'),
     State('inp-mdlfam','value'),  State('inp-ucf1','value'),
     State('inp-ucf2','value')],
    prevent_initial_call=True
)
def predict(n_clicks, mode, country, rtype, nsss_grp, capacity, age,
            constdur, thermal, capcat, mdlfam, ucf1, ucf2):
    if not n_clicks:
        return []

    # Encode inputs
    def safe_encode(le, val, default=0):
        try:
            return int(le.transform([val])[0])
        except:
            return default

    # Derived fields
    age_sq     = age ** 2
    op_years   = min(10, age)
    cap_cat_val = 'small' if capacity < 600 else 'large'
    if capcat:
        cap_cat_val = capcat

    age_grp = ('New' if age < 10 else 'Mature' if age < 25
                else 'Mid_Life' if age < 40 else 'Old')
    op_cat  = 'A' if op_years >= 10 else ('B' if op_years >= 5 else 'C')

    country_ucl = country_mean.get(country, float(np.mean(list(country_mean.values()))))
    type_enc    = safe_encode(le_type, rtype)
    nsss_enc    = safe_encode(le_nsss, nsss_grp)
    cap_enc     = safe_encode(le_cap, cap_cat_val)
    mfam_enc    = safe_encode(le_mfam, mdlfam)
    agegrp_enc  = safe_encode(le_agegrp, age_grp)
    opcat_enc   = safe_encode(le_opcat, op_cat)

    row_m1 = {
        'Type_enc': type_enc, 'Country_enc': country_ucl,
        'Net_Capacity_MW': capacity, 'Cap_enc': cap_enc,
        'Reactor_Age': age, 'Reactor_Age_Sq': age_sq,
        'Operational_Years': op_years, 'NSSS_enc': nsss_enc,
        'Construction_Duration': constdur, 'Thermal_Efficiency_Pct': thermal,
        'MdlFam_enc': mfam_enc, 'AgeGrp_enc': agegrp_enc, 'OpCat_enc': opcat_enc,
    }

    if mode == 'M1':
        X_input = pd.DataFrame([row_m1])[FEAT_M1].fillna(0)
        X_scaled = sc1.transform(X_input)
        proba   = rf_m1.predict_proba(X_scaled)[0]
        fi      = rf_m1.feature_importances_
        feat_names = FEAT_M1
        model_label = "Model 1 — Design Features"
        acc_label   = "79.3% test accuracy"
    else:
        # Estimate UCF3 from UCF1 & UCF2 trend
        ucf3_est  = ucf2 + (ucf2 - ucf1) * 0.5
        ucf3_est  = max(0, min(100, ucf3_est))
        change_13 = ucf3_est - ucf1
        change_23 = ucf3_est - ucf2
        mean_3    = (ucf1 + ucf2 + ucf3_est) / 3
        trend_d   = 2 if ucf3_est > ucf1+2 else (0 if ucf3_est < ucf1-2 else 1)
        row_m2 = {**row_m1,
                  'UCF_2013_2017': ucf1, 'UCF_2010_2020': ucf2,
                  'UCF_change_P1_P3': change_13, 'UCF_change_P2_P3': change_23,
                  'UCF_mean_3period': mean_3, 'UCF_trend_dir': trend_d}
        X_input = pd.DataFrame([row_m2])[FEAT_M2].fillna(0)
        X_scaled = sc2.transform(X_input)
        proba   = rf_m2.predict_proba(X_scaled)[0]
        fi      = rf_m2.feature_importances_
        feat_names = FEAT_M2
        model_label = "Model 2 — Design + UCF History"
        acc_label   = "94.6% test accuracy"

    pred_idx   = int(np.argmax(proba))
    pred_label = ['Low','Medium','High'][pred_idx]
    risk_color = RISK_COLORS[pred_label]

    # Probability chart
    prob_fig = go.Figure()
    bar_colors = [RISK_COLORS['Low'], RISK_COLORS['Medium'], RISK_COLORS['High']]
    prob_fig.add_trace(go.Bar(
        x=['Low UCL\n(<2%)', 'Medium UCL\n(2-6%)', 'High UCL\n(>6%)'],
        y=[p*100 for p in proba],
        marker_color=bar_colors,
        text=[f"{p*100:.1f}%" for p in proba],
        textposition='outside', textfont=dict(size=14, color='white'),
    ))
    prob_fig.update_layout(
        plot_bgcolor='#161b22', paper_bgcolor='#161b22',
        font=dict(color='#E8EDF2'), yaxis=dict(range=[0,120], title='Probability (%)',
                                                gridcolor='#21262d', color='#8B949E'),
        xaxis=dict(color='#8B949E'), showlegend=False,
        margin=dict(t=10, b=10, l=40, r=20), height=240,
    )

    # Feature importance chart (top 8)
    fi_pairs  = sorted(zip(feat_names, fi), key=lambda x: -x[1])[:8]
    fi_labels = [f[0].replace('_enc','').replace('_',' ').title() for f,_ in fi_pairs]
    fi_vals   = [v for _,v in fi_pairs]
    temporal  = {'UCF 2013 2017','UCF 2010 2020','UCF Change P1 P3',
                 'UCF Change P2 P3','UCF Mean 3Period','UCF Trend Dir'}
    fi_colors = ['#3FB950' if l in temporal else '#388BFD' for l in fi_labels]

    fi_fig = go.Figure(go.Bar(
        y=fi_labels[::-1], x=fi_vals[::-1], orientation='h',
        marker_color=fi_colors[::-1], opacity=0.9,
        text=[f"{v:.3f}" for v in fi_vals[::-1]],
        textposition='outside', textfont=dict(color='#8B949E', size=11),
    ))
    fi_fig.update_layout(
        plot_bgcolor='#161b22', paper_bgcolor='#161b22',
        font=dict(color='#E8EDF2'), xaxis=dict(title='Importance', gridcolor='#21262d',
                                                color='#8B949E'),
        yaxis=dict(color='#E8EDF2'), showlegend=False,
        margin=dict(t=10, b=10, l=10, r=60), height=280,
    )

    # Build result panel
    result = html.Div([

        # Risk badge
        html.Div([
            html.Div([
                html.Span(RISK_ICONS[pred_label],
                          style={'fontSize':'2.5rem','marginRight':'12px'}),
                html.Div([
                    html.Div(f"UCL RISK: {pred_label.upper()}",
                             style={'fontSize':'1.5rem','fontWeight':'700',
                                    'color':risk_color,'fontFamily':'Georgia,serif',
                                    'letterSpacing':'1px'}),
                    html.Div(f"Confidence: {proba[pred_idx]*100:.1f}%  ·  {model_label}  ·  {acc_label}",
                             style={'color':'#8B949E','fontSize':'0.8rem',
                                    'fontFamily':'monospace','marginTop':'3px'}),
                ]),
            ], style={'display':'flex','alignItems':'center'}),
            html.P(RISK_DESC[pred_label],
                   style={'color':'#C9D1D9','fontSize':'0.85rem','marginTop':'10px',
                          'lineHeight':'1.6','borderLeft':f'3px solid {risk_color}',
                          'paddingLeft':'12px'}),
        ], style={'background':'#161b22','border':f'1px solid {risk_color}',
                  'borderRadius':'10px','padding':'20px','marginBottom':'16px'}),

        dbc.Row([
            # Probability chart
            dbc.Col([
                html.Label("PREDICTION PROBABILITIES",
                           style={'color':'#58A6FF','fontSize':'0.7rem','fontWeight':'700',
                                  'letterSpacing':'2px','fontFamily':'monospace',
                                  'marginBottom':'8px','display':'block'}),
                dcc.Graph(figure=prob_fig, config={'displayModeBar':False}),
            ], width=6),
            # Feature importance
            dbc.Col([
                html.Label("TOP FEATURE CONTRIBUTIONS",
                           style={'color':'#58A6FF','fontSize':'0.7rem','fontWeight':'700',
                                  'letterSpacing':'2px','fontFamily':'monospace',
                                  'marginBottom':'8px','display':'block'}),
                dcc.Graph(figure=fi_fig, config={'displayModeBar':False}),
                html.P("🟢 Green = temporal feature  🔵 Blue = design feature",
                       style={'color':'#6E7681','fontSize':'0.72rem',
                              'fontFamily':'monospace','marginTop':'4px'}),
            ], width=6),
        ], className='g-3'),

        # Reactor summary
        html.Div([
            html.Label("REACTOR SUMMARY",
                       style={'color':'#58A6FF','fontSize':'0.7rem','fontWeight':'700',
                              'letterSpacing':'2px','fontFamily':'monospace',
                              'marginBottom':'12px','display':'block'}),
            dbc.Row([
                dbc.Col([html.Div([
                    html.Span(f"{country}", style={'color':'#E8EDF2','fontWeight':'600'}),
                    html.Span(" — Country", style={'color':'#6E7681','fontSize':'0.8rem'}),
                ]), html.Div([
                    html.Span(f"{rtype}", style={'color':'#E8EDF2','fontWeight':'600'}),
                    html.Span(" — Reactor Type", style={'color':'#6E7681','fontSize':'0.8rem'}),
                ], style={'marginTop':'6px'})], width=4),
                dbc.Col([html.Div([
                    html.Span(f"{capacity} MWe", style={'color':'#E8EDF2','fontWeight':'600'}),
                    html.Span(" — Net Capacity", style={'color':'#6E7681','fontSize':'0.8rem'}),
                ]), html.Div([
                    html.Span(f"{age} years", style={'color':'#E8EDF2','fontWeight':'600'}),
                    html.Span(" — Reactor Age", style={'color':'#6E7681','fontSize':'0.8rem'}),
                ], style={'marginTop':'6px'})], width=4),
                dbc.Col([html.Div([
                    html.Span(f"{constdur} years", style={'color':'#E8EDF2','fontWeight':'600'}),
                    html.Span(" — Construction Duration", style={'color':'#6E7681','fontSize':'0.8rem'}),
                ]), html.Div([
                    html.Span(f"{thermal:.1f}%", style={'color':'#E8EDF2','fontWeight':'600'}),
                    html.Span(" — Thermal Efficiency", style={'color':'#6E7681','fontSize':'0.8rem'}),
                ], style={'marginTop':'6px'})], width=4),
            ]),
        ], style={'background':'#161b22','border':'1px solid #30363d',
                  'borderRadius':'8px','padding':'16px','marginTop':'12px'}),
    ])

    return result


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  UCL Risk Predictor — Nuclear Power Plant")
    print("  Local:  http://127.0.0.1:8050")
    print("  Network: use your IP address")
    print("="*55 + "\n")
    app.run(debug=False, port=int(os.environ.get('PORT', 8050)), host='0.0.0.0')
