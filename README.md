# UCL Risk Predictor

Code, data and web app for the paper

**Machine Learning Prediction of Unplanned Capability Loss in Nuclear Power Plants: reactor classes and one-year-ahead forecasts** (manuscript in preparation)

Live web app: <https://ucl-risk-predictor.onrender.com>

The paper asks one question: how much of a reactor's **unplanned capability loss (UCL)** can public IAEA data predict beyond simple rules?
It answers in two parts that use the same data and the same standard of evidence.

**Part I** sorts nuclear reactors into three risk classes: Low (below 2 %), Medium (2 to 6 %) and High (above 6 %). The label is the mean of the annual UCL
that the IAEA PRIS Data Browser publishes for each reactor, 2018 to 2024. **Model 1** uses design features only
(387 reactors). **Model 2** adds the reactor's own mean UCL in 2010 to 2013 and in 2014 to 2017, both before the label window (324 reactors).

**Part II** forecasts the UCL of each reactor one year ahead. It uses 17,548 reactor-years from 603 reactors in 35 countries (reactors that have since
been shut down are included for the years they ran) and a rolling test over the target years 2010 to 2025, in which every forecast uses only data
available at the end of the previous year. The models are compared with simple rules such as repeating last year's value or taking the median of the last five usable years.

**Main result, in one paragraph.** The country of a reactor carries most of the usable signal. No design-only model is
clearly better than simply guessing the most common class of the reactor's country. Past UCL adds a modest gain
(Model 2 is about 2 points above the country guess, which is not statistically clear with 324 reactors), and the Medium class
is hard for every model. All Part I numbers below come from 25 test folds in which scaling, the country encoding and SMOTE are fitted on
the training reactors only.

For Part II, a random forest that sees the reactor's own recent UCL and country-level context lowers the mean absolute error by about 21 % against
repeating last year's value and by about 8 % against the median of the last five usable years (3.60 against 4.56 and 3.92 UCL points). A longer median
is a stronger simple rule: the best fixed window (15 years, found by looking at the test results) has an MAE of 3.74, and a median whose window is chosen
on earlier years only has 3.78, so against these the gain is about 4 to 5 %. The class of the forecast is right for 65.9 % of the reactor-years, against
65.1 % (five-year median), 64.5 % (15-year median) and 64.9 % (selected window): about one point, and the intervals reach zero except against the 15-year
median. A pre-specified tuning check (settings searched on training data only) leaves the main conclusions unchanged. The details, the two small things
that did change, and the limits are below and in the paper.

## What is in this repository

| Folder or file | What it is |
|---|---|
| `data/` | The dataset workbook (version 6), a data dictionary, the raw PRIS download and the audit report. |
| `research/` | The analysis scripts behind the paper's tables and figures (Part I, Part II and the tuning check), the reactor-year table of Part II, and the reference outputs. `research/README.md` explains every script and the run order. |
| `app.py`, `ucl_core.py`, `static/`, `artifacts/` | The web app: server, feature code, page, and the two trained models. |
| `train_models.py` | Rebuilds the two web-app models from the dataset. |
| `export_models_for_browser.py` | Converts the models so the page can run them inside the browser. |
| `tools/` | `build_dataset_v6.py` builds the dataset from the raw PRIS download, `audit_dataset.py` checks it independently, the other files check the web app. |
| `tests/` | Automatic checks of the web app. |

## Reproduce the paper's results

Install Python 3.13, then from the repository root:

```
pip install -r requirements-research.txt
cd research
python UCL_Benchmark_CV.py
python UCL_Model1_Pipeline.py
python UCL_Model2_Pipeline.py
python UCL_Model2_History_Ablation.py
```

The scripts find the dataset in `data/` by themselves and use the fixed seed 42. **`UCL_Benchmark_CV.py` is the main evidence** (about 15 minutes):
it scores every model and every baseline on 25 test folds. The two pipeline scripts give the single 80/20 split results, bootstrap intervals, the
cohort split, figures and feature importance (10 to 40 minutes each). The tables from the reference runs are in `research/expected_results/`.
`python UCL_Benchmark_CV.py --tests-only` recomputes the summary and the paired tests from the saved fold scores in seconds.

Part II (forecasts) and the tuning check have their own scripts. They are listed with the run order in `research/README.md`. The reactor-year table is
`research/results_Forecast/forecast_panel.csv`; `python research/UCL_Forecast_Panel.py` rebuilds it identically from the raw PRIS download, and
`python research/UCL_Independent_Check.py` recomputes the main Part II numbers and all 27 input columns of the input sets A to D (for 600 sampled rows) from the raw table by a different code path. The two tuned runs take most of the total time (many hours on a laptop).

To rebuild the dataset from the raw PRIS download and check it again:

```
python tools/build_dataset_v6.py --design data/UCL_Dataset_v5_longitudinal.xlsx --pris data/source/PRIS_annual_values_raw.csv --out data
python tools/audit_dataset.py --xlsx data/UCL_Dataset_v6_PRIS.xlsx --csv data/UCL_Dataset_flat.csv --pris data/source/PRIS_annual_values_raw.csv --report data/AUDIT_REPORT.txt
```

### Reference results: cross-validation benchmark

5 x repeated stratified 5-fold cross-validation (25 test folds). Numbers are means in percent; the standard deviation of the accuracy over the folds is in brackets.
No hyperparameter search is used, so no fold is used to pick settings.

| Setting | Accuracy (SD) | Balanced accuracy | Macro F1 | Medium F1 | AUC macro |
|---|---|---|---|---|---|
| Baseline: always the most common class (387) | 56.3 (0.5) | 33.3 | 24.0 | 0.0 |  |
| Baseline: majority class of the same country (387) | 69.0 (4.3) | 59.5 | 59.0 | 26.7 |  |
| **Model 1**, design only, Logistic Regression (387) | 66.0 (6.3) | 62.4 | 61.4 | 39.7 | 79.1 |
| Model 1, design only, Random Forest (387) | 66.0 (4.5) | 58.4 | 58.8 | 34.1 | 79.9 |
| Model 1, design only, Extra Trees (387) | 65.6 (4.5) | 58.3 | 58.4 | 32.8 | 79.3 |
| Model 1, design only, XGBoost (387) | 64.8 (4.6) | 58.1 | 58.2 | 34.7 | 78.3 |
| Model 1, design only, Stacking (387) | 65.8 (4.0) | 56.0 | 56.1 | 26.3 | 77.4 |

| Setting (324 reactors) | Accuracy (SD) | Balanced accuracy | Macro F1 | Medium F1 | AUC macro |
|---|---|---|---|---|---|
| Baseline: majority class of the same country | 69.3 (3.8) | 62.3 | 61.3 | 29.6 |  |
| Baseline: class of the reactor's own past UCL | 63.9 (4.1) | 60.4 | 59.4 | 40.7 |  |
| Design only (13), Random Forest | 68.0 (4.7) | 62.2 | 62.6 | 38.5 | 79.3 |
| Design without Country (12), Random Forest | 63.2 (4.5) | 57.7 | 58.0 | 33.6 | 76.3 |
| History only (5), Logistic Regression | 66.7 (5.3) | 59.6 | 59.0 | 35.4 | 77.7 |
| **Model 2**, design + history (18), Random Forest | 71.0 (5.5) | 65.2 | 65.5 | 45.9 | 82.5 |
| Model 2, Extra Trees | 71.6 (5.3) | 64.8 | 64.9 | 42.4 | 82.2 |
| Model 2, Logistic Regression | 67.5 (5.3) | 63.4 | 62.6 | 38.1 | 79.3 |
| Model 2, XGBoost | 65.3 (5.8) | 59.4 | 59.3 | 35.4 | 78.8 |
| Model 2, Stacking | 70.4 (5.1) | 62.0 | 62.2 | 36.6 | 81.7 |

Paired comparisons on the same folds, accuracy difference in points with the Nadeau-Bengio corrected resampled t-test (p-values are not adjusted for the many comparisons):

| Comparison | Logistic Regression | Random Forest |
|---|---|---|
| Model 1 (387) minus country guess | -3.0 (p = 0.40) | -2.9 (p = 0.26) |
| Model 2 (324) minus country guess | -1.9 (p = 0.49) | +1.7 (p = 0.52) |
| Model 2 (324) minus own-past-UCL guess | +3.6 (p = 0.32) | +7.2 (p = 0.04) |
| Model 2 minus Model 1 (same 324 reactors) | +1.0 (p = 0.60) | +3.0 (p = 0.20) |
| Design with minus without Country (324) | +6.6 (p = 0.04) | +4.8 (p = 0.02) |

History-period check (the same folds; Random Forest):

| Feature set (Random Forest, 324 reactors) | Accuracy (SD) | Macro F1 | Medium F1 |
|---|---|---|---|
| Design only (13) | 68.0 (4.7) | 62.6 | 38.5 |
| Design + UCL 2010-2013 (14) | 69.4 (5.3) | 63.5 | 39.6 |
| Design + UCL 2014-2017 (14) | 69.8 (4.6) | 64.0 | 41.1 |
| Design + both periods, no derived features (15) | 71.6 (5.0) | 65.7 | 45.3 |
| Full Model 2 (18) | 71.0 (5.5) | 65.5 | 45.9 |
| Model 2 without Country (17) | 69.1 (5.7) | 63.1 | 42.3 |

How to read this: both history periods together add about 3.6 points for the Random Forest (p = 0.16), the three derived trend features add nothing, and Model 2 loses only 2.0 points when the country is
removed (p = 0.26), compared with 4.8 points for the design-only model, so a reactor's past UCL partly stands in for its country.
Correlation of the label with the history blocks: UCL 2010-2013 r = 0.54, UCL 2014-2017 r = 0.40.

### Reference results: Part II, one-year-ahead forecasts

Rolling-origin test, target years 2010 to 2025 pooled (6,196 scored reactor-years, 505 reactors). Each forecast for year Y uses only data up to the end of Y minus 1; the learners are fitted
on the latest 15 (or 10, or all) earlier target years. MAE is the mean absolute error in UCL points (lower is better). Class accuracy is the share of reactor-years in which the forecast falls in the correct
Low, Medium or High class, 15-year window. Fixed (untuned) settings. Input set C is the reactor's own UCL history and its planned and external losses, plus the world and country context at the origin year (the sets A to E are defined in the paper).

| Method | MAE, latest 15 years | MAE, latest 10 years | MAE, all earlier years | Class accuracy (%), 15 years |
|---|---|---|---|---|
| Repeat last usable UCL (persistence) | 4.56 | 4.56 | 4.56 | 61.1 |
| Mean of the last 5 usable years | 4.43 | 4.43 | 4.43 | 60.1 |
| **Median of the last 5 usable years** | 3.92 | 3.92 | 3.92 | 65.1 |
| Mean of the country, last 3 years | 4.57 | 4.57 | 4.57 | 54.3 |
| Mean of the whole fleet | 5.41 | 5.41 | 5.41 | 19.4 |
| Ridge regression, set C | 3.74 | 3.73 | 3.69 | 65.9 |
| **Random forest, set C** | 3.60 | 3.61 | 3.57 | 65.9 |
| XGBoost, set C | 3.63 | 3.66 | 3.80 | 65.9 |

Random forest (set C, 15 years) minus persistence: MAE -0.97 (95 % interval -1.12 to -0.79), class accuracy +4.8 points (+3.6 to +5.9).
Random forest minus the median rule: MAE -0.32 (-0.45 to -0.21), class accuracy +0.7 points (-0.2 to +1.7), so the class accuracy cannot be told from that of the median rule.
The intervals come from 1,000 resamples of whole reactors. The forecasts are typical-year forecasts: on average they lie about 1.8 points below the realised UCL (`research/expected_results/forecast/EXTRA_CHECKS.json`).
The full tables, the by-year results, the new-site and new-country tests and the checks without 2020, 2022 and 2025 are in `research/expected_results/forecast/`.

### Reference results: tuning check

The main analysis uses fixed, sensible settings. To show that the conclusions do not depend on them, the settings were also searched (8 random candidates per model, with the fixed setting as candidate 0)
using training data only. The ranges, the number of candidates and the decision rule were written down and committed **before** any tuned model was run (`research/TUNING_PRESPECIFICATION.md` and its git history).
The ranges are the author's own choices. Results are in `research/expected_results/tuning/`.

| | Part I (class, 25 outer folds, nested validation) | Part II (forecast, rolling origin, three windows) |
|---|---|---|
| Change from tuning | accuracy -2.0 to +5.9 points (21 of 25 combinations up); 1 of 25 differences significant | MAE changes of at most 0.054 points in 11 of 12 combinations; the twelfth (XGBoost, set C, all earlier years) improves by 0.178 because its fixed setting was the weakest |
| Tuned random forest, set C, against the median rule | not applicable | MAE 3.59, 3.60 and 3.55 in the three windows, against 3.92; every interval excludes zero; class accuracy still not different |
| Main claims | hold with fixed and with tuned settings | hold with fixed and with tuned settings |
| What changed | tree ensembles on history-only inputs gain 3 to 6 points and come close to logistic regression; Model 2 over Model 1 for extra trees is no longer significant | the lower class accuracy of XGBoost with all earlier years (fixed settings) does not appear when tuned |

Tuning covered the repeated cross-validation of Part I and the rolling origin of Part II for the random forest and XGBoost with input sets A and C. The leave-one-country-out analysis, the grouped tests, ridge regression and the other input sets were run with fixed settings only.

### Reference results: single 80/20 split (pipeline scripts)

These come from one split of the reactors, so they move by one to one and a half points per reactor and should not be read as the model's accuracy.
They are kept because they also give the bootstrap intervals, the cohort split and the figures. Hyperparameters are tuned in these scripts.
The columns `CV_Mean_balanced_train` in the pipeline tables are cross-validated on SMOTE-balanced training data and are optimistic. Do not quote them; use the benchmark above.

**Model 1** (387 reactors, test set of 78). Output: `results_Model1/`. The cohort split trains on reactors with 10 or more years of operation and tests on the newer reactors.

| Classifier | Accuracy (%) | Weighted F1 (%) | AUC-ROC (%) | Cohort split accuracy (%) |
|---|---|---|---|---|
| LR | 71.79 | 71.56 | 79.47 | 66.67 |
| RF | 74.36 | 73.09 | 82.75 | 68.89 |
| ETC | 71.79 | 70.28 | 82.25 | 71.11 |
| XGB | 75.64 | 74.74 | 83.53 | 71.11 |
| Stacking | 69.23 | 66.83 | 83.21 | 68.89 |

Bootstrap 95 % intervals of the best classifier are in `M1_table3_bootstrap.csv`. Per-class results of the first classifier are in `M1_table2_per_class.csv`.

**Model 2** (324 reactors, test set of 65). Output: `results_Model2/`. Model 1 is evaluated on the same reactors and the same split.

| Classifier | Model 1 accuracy (%) | Model 2 accuracy (%) | Model 2 weighted F1 (%) | Model 2 AUC-ROC (%) |
|---|---|---|---|---|
| LR | 67.69 | 69.23 | 68.86 | 81.03 |
| RF | 63.08 | 69.23 | 67.50 | 83.25 |
| ETC | 63.08 | 66.15 | 63.33 | 83.01 |
| XGB | 70.77 | 69.23 | 67.39 | 83.59 |
| Stacking | 64.62 | 67.69 | 65.55 | 84.26 |

The McNemar test between Model 1 and Model 2 is not significant on this split (p = 1.00). On the cohort split (train on reactors connected to the grid in 1990 or earlier, test on newer ones) the Random Forest reaches
59.7 %, the XGBoost 48.6 % and the Stacking 55.6 %: models trained on older reactors do not carry over well to newer ones.
Random Forest importance: history features together 42.7 %, Country 21.3 %.

With the library versions pinned in `requirements-research.txt` the scripts reproduce these values. Newer scikit-learn versions can change Random Forest results by one reactor.

## The web app

The web app runs both models. You describe a reactor (type, country, size, age, build time, supplier,
design family and, for Model 2, the two UCL averages) and it returns the class, the three class
scores, and what pushed the result compared with a typical reactor. The page shows the cross-validated accuracy of the model
(66 % for Model 1, 71 % for Model 2) next to the accuracy of guessing from the country alone
(69 % and 69 %). Model 1 is a Logistic Regression and Model 2 a Random Forest, chosen
by the highest macro F1 in the benchmark; both are trained on all reactors of their group. The page has a light and a dark
mode. The models run inside the visitor's browser, so answers appear instantly once the page has
loaded. If the browser cannot load them, the page asks the server instead.

To run it on your own computer:

```
pip install -r requirements.txt
python app.py
```

Then open <http://127.0.0.1:5000>. To run the checks, use `python -m unittest -v` (server tests) and
`node tools/check_local_engine.mjs` (compares the in-browser models with the Python models on 604
random inputs).

The site is deployed on Render from `render.yaml`. The start command is
`gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:$PORT`. On Render's free plan the
server sleeps when nobody is using it, so the first visit after a pause can take up to a minute.

The web-app models are rebuilt from the dataset with `python train_models.py data/UCL_Dataset_v6_PRIS.xlsx`
(needs `requirements-train.txt`).

The tool is a research demonstration. It must not be used for operating, safety or licensing decisions.

## Data and code availability

The compiled dataset (387 reactors in 31 countries), the raw PRIS download it is built from, the reactor-year table of Part II (17,548 rows), the build and audit scripts, all analysis code, the stored results of every table and figure, the
trained models and the source of the web application are in this repository
(<https://github.com/Zahiiiin/UCL_risk_predictor>). The design columns come from IAEA RDS-2 and PRIS reactor details; the label and the history
come from the annual performance values of the IAEA PRIS Data Browser (downloaded October 2026). The independent audit (`data/AUDIT_REPORT.txt`) recomputes the label
and the history from the raw download and finds no difference. The underlying figures are published by the
International Atomic Energy Agency. The web application is at <https://ucl-risk-predictor.onrender.com>.

## Known limitations of the analysis

These are listed so that readers can judge the results for themselves.

1. **Small samples.** There are 387 reactors (324 for Model 2), in 31 countries. In the benchmark the standard deviation of the fold accuracy is about 5 points, and most differences between models are not statistically clear. The paired tests are corrected for overlapping training sets but not for the number of comparisons.
2. **The country dominates.** Guessing the most common class of the reactor's country is as accurate as any design-only model. A country effect can stand for operator practice, regulation, grid conditions and fleet age; the data cannot separate them.
3. **Population.** In Part I, only reactors listed as operational at the end of 2024 are included, and a reactor needs at least 3 usable years in 2018 to 2024. Reactors shut down earlier are missing, so the Part I data describe the current fleet and cannot be generalised to all reactors that ever operated. Part II also includes reactors that have been shut down, for the years they ran, but only for years in which PRIS publishes a UCL value (not a full shutdown, not the first grid year).
4. **The label is an average of annual values.** A reactor with one bad year and a reactor with a steady medium loss can fall in the same class. The class limits (2 and 6 %) were chosen by the authors; three reactors lie within 0.05 of a limit.
5. **The Medium class is hard.** The best F1 of the Medium class in the benchmark is below 50 %. Predictions of Medium should be treated as a weak signal.
6. **Thermal efficiency carries no information.** The scripts read a workbook sheet without a thermal power column, so `Thermal_Efficiency` is the constant 33.0 for every reactor. The real values are in the `Thermal_Efficiency_Pct` column; `UCL_Model2_Pipeline.py --pris-thermal` (and `USE_PRIS_THERMAL = True` in the Model 1 script) switch to them.
7. **The pipeline scripts are less strict than the benchmark.** In `UCL_Model1_Pipeline.py` and `UCL_Model2_Pipeline.py` scaling and the country encoding are computed on all reactors before the split, the hyperparameter search is scored on SMOTE-balanced data, and the single test sets are small. The benchmark fits scaling, country encoding and SMOTE inside each training fold and uses fixed settings; the separate tuning check described above searches settings inside the training parts only.
8. **The cohort split is not a forecast.** All reactors of Model 2 were connected to the grid by 2011, so a split by years of operation is not possible; the pipeline splits by grid-connection year (1990) instead. This tests transfer to newer reactors, not prediction of future years.
9. **Not an operating or safety tool.** The web app is a research demonstration. UCL measures lost energy, not safety.
10. **Part II forecasts are typical-year forecasts.** The learners are fitted to log(1 + UCL), so on average their forecasts lie about 1.8 points below the realised UCL. They are not suited to estimating the expected total energy loss of a reactor.
11. **Part II gains are small.** The random forest lowers the error against the five-year median by about 0.3 UCL points and against the best longer median by about 0.14 to 0.19 points (4 to 5 %); the class of its forecast is right about one point more often, a difference whose interval reaches zero for most of these rules. Twelve combinations of learner and input set were compared without adjustment, and the one with the lowest error is quoted, so its advantage is somewhat optimistic.
12. **Post hoc checks.** The check without the 15 Ukrainian reactors (whose labels rest on four years; without them the gain of Model 2 over the persistence rule is 5.0 points, p = 0.080), the analysis by recent level, the checks without 2020, 2022 and 2025, the choice of the 15-year window as the primary window and the comparison with medians over longer windows (`research/UCL_Forecast_LongMedian.py`, added after a referee-style review) were made after the first results were seen. They are reported as checks, not confirmed findings.
13. **Tuning is limited.** Eight random candidates per model, in ranges chosen by the author, with small and noisy inner validation parts. A larger search could find better settings.
14. **2025 values.** The values for 2025 come from the October 2026 download. It is not confirmed that the IAEA treats them as final.

## Licence, data terms and citation

The code is released under the MIT licence (`LICENSE`). The dataset is derived from IAEA publications
and the IAEA PRIS database, which remain the property of the IAEA; see `data/DATA_DICTIONARY.md`.
Please cite the paper (see `CITATION.cff`) and the IAEA sources.
