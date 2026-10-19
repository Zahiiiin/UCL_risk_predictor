# UCL Risk Predictor

Code, data and web app for the paper

**Machine Learning Classification of Unplanned Capability Loss Risk in Nuclear Power Plants: A Longitudinal Study Using IAEA RDS-2 Data**
(submitted to *Nuclear Engineering and Design*)

Live web app: <https://ucl-risk-predictor.onrender.com>

The paper sorts nuclear reactors into three risk classes for **unplanned capability loss (UCL)**:
Low (below 2 %), Medium (2 to 6 %) and High (above 6 %). UCL is computed as 100 - UCF - PCL from IAEA
data. **Model 1** uses design features only (407 reactors). **Model 2** adds two average unit
capability factors (UCF) from IAEA reports, for 2013 to 2017 and 2010 to 2020 (370 reactors).

## What is in this repository

| Folder or file | What it is |
|---|---|
| `data/` | The dataset workbook and a data dictionary. |
| `research/` | The analysis scripts behind the paper's tables and figures, and the reference outputs. |
| `app.py`, `ucl_core.py`, `static/`, `artifacts/` | The web app: server, feature code, page, and the two trained models. |
| `train_models.py` | Rebuilds the two web-app models from the dataset. |
| `export_models_for_browser.py` | Converts the models so the page can run them inside the browser. |
| `tests/`, `tools/` | Automatic checks (see "The web app" below). |

## Reproduce the paper's results

Install Python 3.13, then from the repository root:

```
pip install -r requirements-research.txt
cd research
python UCL_Model1_Pipeline.py
python UCL_Model2_Pipeline.py
```

Both scripts find the dataset in `data/` by themselves, use the fixed seed 42, and write tables
and figures to `results_Model1/` and `results_Model2/`. A full run takes roughly 10 to 40 minutes on a
normal laptop. `python UCL_Model2_Pipeline.py --quick` is a two-minute smoke test whose numbers
differ slightly. The tables from the reference run are in `research/expected_results/`.

`python UCL_Model2_History_Ablation.py` runs one extra check that is not in the paper's main tables:
it shows how much each of the two UCF periods adds (see "Known limitations", item 6).

### Reference results

**Model 1** (407 reactors, test set of 82 reactors). Output: `results_Model1/`.

| Classifier | Accuracy (%) | Weighted F1 (%) | AUC-ROC (%) | CV mean (%) | Temporal split accuracy (%) |
|---|---|---|---|---|---|
| LR | 71.95 | 72.09 | 79.93 | 67.03 | 82.54 |
| RF | 69.51 | 70.26 | 85.71 | 85.77 | 84.13 |
| ETC | 75.61 | 74.97 | 89.11 | 87.57 | 82.54 |
| XGB | 71.95 | 71.29 | 84.91 | 85.23 | 74.60 |
| Stacking | 76.83 | 75.65 | 86.90 | 86.49 | 84.13 |

Per-class results of the stacking ensemble (paper Table 3):

| Class | Support | Precision (%) | Recall (%) | F1 (%) |
|---|---|---|---|---|
| Low | 46 | 84.6 | 95.7 | 89.8 |
| Medium | 12 | 30.0 | 25.0 | 27.3 |
| High | 24 | 80.0 | 66.7 | 72.7 |

Bootstrap 95 % intervals of the stacking ensemble (500 resamples): accuracy
[67.1, 85.4], AUC-ROC
[79.6, 95.1].

**Model 2** (370 reactors, test set of 74 reactors). Output: `results_Model2/`.
Model 1 is evaluated here on the same 370 reactors and the same split, so the two columns can be compared.

| Classifier | Model 1 accuracy (%) | Model 2 accuracy (%) | Model 2 weighted F1 (%) | Model 2 AUC-ROC (%) | Model 2 MCC |
|---|---|---|---|---|---|
| LR | 72.97 | 81.08 | 82.69 | 95.79 | 0.701 |
| RF | 79.73 | 89.19 | 89.24 | 98.25 | 0.818 |
| ETC | 78.38 | 89.19 | 89.18 | 98.37 | 0.822 |
| XGB | 79.73 | 93.24 | 92.83 | 96.35 | 0.887 |
| Stacking | 78.38 | 90.54 | 90.72 | 98.78 | 0.841 |

Per-class F1 of the best classifier (XGBoost), Model 1 against Model 2 on the same test reactors:

| Class | Support | Model 1 XGBoost F1 (%) | Model 2 XGBoost F1 (%) |
|---|---|---|---|
| Low | 40 | 90.0 | 97.5 |
| Medium | 11 | 47.6 | 73.7 |
| High | 23 | 76.6 | 93.9 |

McNemar test, Model 1 XGBoost against Model 2 XGBoost on the 74 test reactors: Model 1 was right and
Model 2 wrong for 1 test reactor, the reverse held for 11 test reactors,
p = 0.0094 (chi-square with continuity correction; exact binomial p = 0.0063).

Bootstrap 95 % intervals of Model 2 XGBoost (500 resamples): accuracy
[86.5, 98.0], AUC-ROC
[92.1, 99.4].

Temporal split (train on reactors with 10 or more years of operation, test on the 27 newer reactors; library-default settings, paper Table 7):

| Classifier | Random split accuracy (%) | Temporal split accuracy (%) | Temporal F1 (%) |
|---|---|---|---|
| RF | 89.19 | 85.19 | 81.98 |
| XGB | 93.24 | 81.48 | 80.00 |
| Stacking | 90.54 | 85.19 | 79.98 |

Feature importance (share of the total). The two models rank the features differently, so both are saved.
History features together: Random Forest 60.6 %, XGBoost 54.1 %.
UCF 2010-2020: Random Forest 26.4 %, XGBoost 14.9 %.
UCF mean (2 periods): Random Forest 19.6 %, XGBoost 34.0 %.
Country: Random Forest 10.8 %, XGBoost 4.6 %.

**History-period check** (`UCL_Model2_History_Ablation.py`, not part of the paper's main tables).
Correlation of the two history periods with the UCF 2014-2024 window that defines the label: UCF 2013-2017 r = 0.60,
UCF 2010-2020 r = 0.93. Tuned XGBoost on five feature sets:

| Feature set | Test accuracy (%) | 5 x 5-fold CV accuracy (%) |
|---|---|---|
| Design only (13) | 79.73 | 74.4 +/- 4.6 |
| Design + UCF 2013-2017 (14) | 78.38 | 75.5 +/- 3.9 |
| Design + UCF 2010-2020 (14) | 91.89 | 88.0 +/- 3.4 |
| Design + both periods, no derived (15) | 90.54 | 88.0 +/- 3.4 |
| Full Model 2 (18) | 93.24 | 88.0 +/- 3.2 |

Almost all of Model 2's gain over Model 1 comes from UCF 2010-2020. Adding UCF 2013-2017 alone changes little, and the three
derived trend features add nothing in cross-validation. The repeated cross-validation is the steadier comparison, because one
test reactor moves the 74-reactor test accuracy by 1.35 points.


With the library versions pinned in `requirements-research.txt`, these values equal the paper's results for
Model 1 (Tables 2 and 3, the bootstrap intervals and the temporal split) and Model 2 (Tables 5 to 7 and the McNemar test).
A newer scikit-learn (1.9) changes the Random Forest results by one test reactor, so please keep the pinned version.
One test reactor is worth 1.2 points of accuracy for Model 1 and 1.35 points for Model 2.
The tables above show two decimals; the paper rounds to one.

## The web app

The web app runs both models. You describe a reactor (type, country, size, age, build time, supplier,
design family and, for Model 2, the two UCF averages) and it returns the class, the three class
scores, and what pushed the result compared with a typical reactor. The page has a light and a dark
mode. The models run inside the visitor's browser, so answers appear instantly once the page has
loaded. If the browser cannot load them, the page asks the server instead.

To run it on your own computer:

```
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:5000>. To run the checks, use `python -m unittest -v` (server tests) and
`node tools/check_local_engine.mjs` (compares the in-browser models with the Python models on 604
random inputs; the largest difference in a class probability is 0.0001).

The site is deployed on Render from `render.yaml`. The start command is
`gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`. On Render's free plan the
server sleeps when nobody is using it, so the first visit after a pause can take up to a minute.

The web-app models are rebuilt from the dataset with `python train_models.py data/UCL_Dataset_v5_longitudinal.xlsx`
(needs `requirements-train.txt`). They use the same split and settings as the research scripts.

The tool is a research demonstration. It must not be used for operating, safety or licensing decisions.

## Data and code availability

> The compiled dataset (407 reactors, derived from IAEA RDS-2 and PRIS data), all analysis code, the
> trained models and the source of the web application are available at [repository link]
> (archived at [DOI]). The underlying figures are published by the International Atomic Energy
> Agency (RDS-2, 2018, 2021 and 2025 editions; PRIS). The web application is available at
> https://ucl-risk-predictor.onrender.com.

## Known limitations of the analysis

These are listed so that readers can judge the results for themselves.

1. **Thermal efficiency carries no information.** The workbook sheet that the scripts read has no
   thermal power column, so the feature `Thermal_Efficiency` is the constant 33.0 for every reactor.
   Its zero importance is a property of the data. The real values are in the `Thermal_Efficiency_Pct`
   column; `UCL_Model2_Pipeline.py --pris-thermal` (and `USE_PRIS_THERMAL = True` in the Model 1 script)
   switch to them.
2. **Scaling and the Country encoding use all reactors.** Both are computed before the train and test
   split. The Country encoding is out-of-fold, and tree models ignore scaling, but a stricter design
   would fit both on the training reactors only.
3. **Cross-validation inside the search uses balanced data.** SMOTE is applied to the training reactors
   only, but the search's cross-validation then runs on that balanced set, so those scores (and the
   chosen hyperparameters) are optimistic. The reported test scores are not affected.
4. **Small test sets.** One test reactor is worth 1.2 points (Model 1, 82 reactors) or 1.35 points
   (Model 2, 74 reactors). The bootstrap intervals show how wide the uncertainty is.
5. **Settings that differ between experiments.** The stacking ensemble uses fixed, untuned base learners.
   The temporal-split experiments use library-default hyperparameters, so they are not directly
   comparable with the tuned random-split results. Hyperparameter search is scored with weighted F1.
6. **The two history periods overlap the label window.** UCF 2013 to 2017 lies inside UCF 2010 to 2020,
   and both overlap the 2014 to 2024 window that defines the label (2014 to 2017 and 2014 to 2020).
   UCF 2014 to 2024 itself is never an input. Model 2 therefore uses recent performance data and is not a
   forecast from strictly earlier data. `UCL_Model2_History_Ablation.py` measures how much each period adds.

## Licence, data terms and citation

The code is released under the MIT licence (`LICENSE`). The dataset is derived from IAEA publications
and the IAEA PRIS database, which remain the property of the IAEA; see `data/DATA_DICTIONARY.md`.
Please cite the paper (see `CITATION.cff`) and the IAEA sources.
