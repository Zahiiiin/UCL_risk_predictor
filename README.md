# UCL Risk Predictor

Code, data and web app for the paper

**Machine Learning Classification of Unplanned Capability Loss Risk in Nuclear Power Plants: Evidence from IAEA RDS-2 Data**
(manuscript in preparation)

Live web app: <https://ucl-risk-predictor.onrender.com>

The paper sorts nuclear reactors into three risk classes for **unplanned capability loss (UCL)**:
Low (below 2 %), Medium (2 to 6 %) and High (above 6 %). UCL is computed as 100 - UCF - PCL from IAEA
data. **Model 1** uses design features only (407 reactors). **Model 2** adds two average unit
capability factors (UCF) from IAEA reports, for 2013 to 2017 and 2010 to 2020 (368 reactors).

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

`python UCL_Model2_History_Ablation.py` runs the history-period check that is reported as Table 8 of the paper (its output file is called `M2_table9_history_ablation.csv`, because the script numbers its own tables):
it shows how much each of the two UCF periods adds (see "Known limitations", item 6).

### Reference results

**Model 1** (407 reactors, test set of 82 reactors). Output: `results_Model1/`.

| Classifier | Accuracy (%) | Weighted F1 (%) | AUC-ROC (%) | CV mean (%) | Temporal split accuracy (%) |
|---|---|---|---|---|---|
| LR | 71.95 | 72.09 | 79.62 | 67.39 | 81.97 |
| RF | 69.51 | 70.29 | 85.57 | 85.59 | 83.61 |
| ETC | 74.39 | 73.54 | 88.99 | 87.75 | 80.33 |
| XGB | 70.73 | 70.24 | 84.67 | 85.05 | 72.13 |
| Stacking | 75.61 | 74.88 | 86.45 | 86.85 | 80.33 |

Per-class results of the stacking ensemble (paper Table 3):

| Class | Support | Precision (%) | Recall (%) | F1 (%) |
|---|---|---|---|---|
| Low | 46 | 86.0 | 93.5 | 89.6 |
| Medium | 12 | 27.3 | 25.0 | 26.1 |
| High | 24 | 76.2 | 66.7 | 71.1 |

Bootstrap 95 % intervals of the stacking ensemble (500 resamples): accuracy
[65.9, 84.1], AUC-ROC
[78.9, 94.8].

**Model 2** (368 reactors, test set of 74 reactors). Output: `results_Model2/`.
Model 1 is evaluated here on the same 368 reactors and the same split, so the two columns can be compared.

| Classifier | Model 1 accuracy (%) | Model 2 accuracy (%) | Model 2 weighted F1 (%) | Model 2 AUC-ROC (%) | Model 2 MCC |
|---|---|---|---|---|---|
| LR | 74.32 | 86.49 | 86.78 | 94.98 | 0.775 |
| RF | 77.03 | 89.19 | 88.73 | 98.02 | 0.817 |
| ETC | 78.38 | 91.89 | 91.77 | 98.61 | 0.862 |
| XGB | 75.68 | 87.84 | 87.07 | 98.22 | 0.797 |
| Stacking | 75.68 | 91.89 | 91.76 | 98.79 | 0.861 |

Per-class F1 of the best classifier (Extra Trees), Model 1 against Model 2 on the same test reactors:

| Class | Support | Model 1 Extra Trees F1 (%) | Model 2 Extra Trees F1 (%) |
|---|---|---|---|
| Low | 40 | 87.8 | 95.0 |
| Medium | 11 | 50.0 | 76.2 |
| High | 23 | 73.9 | 93.6 |

McNemar test, Model 1 Extra Trees against Model 2 Extra Trees on the 74 test reactors: Model 1 was right and
Model 2 wrong for 0 test reactors, the reverse held for 10 test reactors,
p = 0.0044 (chi-square with continuity correction; exact binomial p = 0.0020).

Bootstrap 95 % intervals of Model 2 Extra Trees (500 resamples): accuracy
[85.1, 97.3], AUC-ROC
[96.8, 99.8].

Temporal split (train on reactors with 10 or more years of operation, test on the 23 newer reactors; library-default settings, paper Table 7):

| Classifier | Random split accuracy (%) | Temporal split accuracy (%) | Temporal F1 (%) |
|---|---|---|---|
| RF | 89.19 | 82.61 | 78.87 |
| XGB | 87.84 | 73.91 | 73.04 |
| Stacking | 91.89 | 78.26 | 74.37 |

Feature importance (share of the total). The two models rank the features differently, so both are saved.
History features together: Random Forest 61.2 %, Extra Trees 47.3 %.
UCF 2010-2020: Random Forest 26.1 %, Extra Trees 21.0 %.
UCF mean (2 periods): Random Forest 17.7 %, Extra Trees 11.7 %.
Country: Random Forest 10.8 %, Extra Trees 13.6 %.

**History-period check** (`UCL_Model2_History_Ablation.py`; this is Table 8 of the paper, saved as `M2_table9_history_ablation.csv`).
Correlation of the two history periods with the UCF 2014-2024 window that defines the label: UCF 2013-2017 r = 0.60,
UCF 2010-2020 r = 0.93. Tuned XGBoost on five feature sets:

| Feature set | Test accuracy (%) | 5 x 5-fold CV accuracy (%) |
|---|---|---|
| Design only (13) | 75.68 | 73.9 +/- 3.3 |
| Design + UCF 2013-2017 (14) | 79.73 | 75.2 +/- 3.6 |
| Design + UCF 2010-2020 (14) | 87.84 | 88.9 +/- 2.9 |
| Design + both periods, no derived (15) | 89.19 | 88.1 +/- 3.6 |
| Full Model 2 (18) | 87.84 | 87.5 +/- 3.5 |

Almost all of Model 2's gain over Model 1 comes from UCF 2010-2020 (cross-validated accuracy 73.9 % without it, 88.9 % with it).
Adding UCF 2013-2017 alone adds little in cross-validation (+1.3 points), and the three derived trend features add nothing
(88.1 % without them, 87.5 % with them). The repeated cross-validation is the steadier comparison, because one
test reactor moves the 74-reactor test accuracy by 1.35 points.


With the library versions pinned in `requirements-research.txt`, these values equal the paper's results for
Model 1 (Tables 2 and 3, the bootstrap intervals and the temporal split) and Model 2 (Tables 5 to 8 and the McNemar test).
Repeating the run on the same versions gives the same numbers. A check on the first version of the dataset showed that a newer
scikit-learn (1.9) changes the Random Forest results by one test reactor, so please keep the pinned version.
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

The compiled dataset (407 reactors, derived from IAEA RDS-2 and PRIS data), all analysis code, the
trained models and the source of the web application are in this repository
(<https://github.com/Zahiiiin/UCL_risk_predictor>). Before release the dataset was checked value by value against the
printed IAEA tables; 24 reactors had values corrected, and the list is in `data/DATA_DICTIONARY.md`. The underlying figures are published by the
International Atomic Energy Agency (RDS-2, 2018, 2021, 2024 and 2025 editions; PRIS). The web
application is at <https://ucl-risk-predictor.onrender.com>.

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
   (Model 2, 74 reactors). The bootstrap intervals show how wide the uncertainty is. The effect is real: when the
   24 corrected reactors were put into the dataset, the XGBoost Model 2 test accuracy moved by four reactors
   (from 93.24 % to 87.84 %), while the cross-validated accuracies moved by about half a point.
5. **Settings that differ between experiments.** The stacking ensemble uses fixed, untuned base learners.
   The temporal-split experiments use library-default hyperparameters, so they are not directly
   comparable with the tuned random-split results. Hyperparameter search is scored with weighted F1.
6. **The two history periods overlap the label window.** UCF 2013 to 2017 lies inside UCF 2010 to 2020,
   and both overlap the 2014 to 2024 window that defines the label (2014 to 2017 and 2014 to 2020).
   UCF 2014 to 2024 itself is never an input. Model 2 therefore uses recent performance data and is not a
   forecast from strictly earlier data. `UCL_Model2_History_Ablation.py` measures how much each period adds.
7. **The labels are a proxy.** PCL is one average value per reactor type and size class (see
   `data/DATA_DICTIONARY.md`), not a measured value for each reactor. For 191 of the 407 reactors UCL is set
   to 0 (190 where 100 - UCF - PCL is negative and one where it is exactly 0), and all of them are Low.
   Because PCL is a constant for each type, all 43 BWR reactors are Low and all 8 GCR reactors are High.
8. **Two PCL sources leave out some reactors.** Table 18 of the 2021 edition states that it contains no
   outage data for French reactor units. The PWR values taken from that edition are therefore based on PWRs
   other than the French units, although French reactors are in the dataset. Table 18 of the 2024 edition
   likewise has no 2022 and 2023 outage data for Ukrainian units.

## Licence, data terms and citation

The code is released under the MIT licence (`LICENSE`). The dataset is derived from IAEA publications
and the IAEA PRIS database, which remain the property of the IAEA; see `data/DATA_DICTIONARY.md`.
Please cite the paper (see `CITATION.cff`) and the IAEA sources.
