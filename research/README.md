# Research scripts

These scripts produce every table, number and figure of the paper. Nothing in the paper is typed by hand: each number is read from a
result file in `expected_results/`, which is produced by one of the scripts below. The dataset
(`../data/UCL_Dataset_v6_PRIS.xlsx`) is found by the scripts themselves.

## What each script does

**Part I: which class does a reactor fall in?**

| Script | What it does |
|---|---|
| `UCL_Benchmark_CV.py` | **Main evidence of Part I.** 5 x 5-fold repeated cross-validation (25 test folds) of five classifiers on nine input sets, against rules that need no learning (most common class, most common class of the same country, class of the reactor's own past UCL). Scaling, country encoding and SMOTE are fitted inside each training fold. Paired tests with the Nadeau-Bengio correction. About 15 minutes. |
| `UCL_Leave_Country_Out.py` | Each reactor is predicted by a model trained on other countries only. |
| `UCL_Permutation_Importance.py` | Which inputs the forest really uses, measured on reactors it has not seen. |
| `UCL_Label_Stability.py` | How often the class from 2018 to 2021 equals the class from 2022 to 2024 (no model). |
| `UCL_Threshold_Sensitivity.py` | Repeats the benchmark with other class limits than 2 % and 6 %. |
| `UCL_Ukraine_Check.py` | Post hoc check: repeats the benchmark without the 15 Ukrainian reactors, whose labels rest on four years. |
| `UCL_Model1_Pipeline.py`, `UCL_Model2_Pipeline.py`, `UCL_Model2_History_Ablation.py` | The older single-split analyses of Model 1 and Model 2 (13 design inputs; plus five inputs from past UCL). They also produce the models that the web application uses. |

**Part II: what will a reactor's UCL be next year?**

| Script | What it does |
|---|---|
| `UCL_Forecast_Panel.py` | Builds the reactor-year table (one row per reactor and year, inputs known at the end of the year, target = UCL of the next year). Contains a leakage check that recomputes 400 rows from scratch. |
| `UCL_Forecast_Benchmark.py` | The rolling-origin test for the target years 2010 to 2025: simple rules, ridge regression, random forest and XGBoost, bootstrap intervals that resample whole reactors. Options for the training window (`--window 15`, `--window 10`), for places the model has not seen (`--grouped site` or `country`) and for the 387 reactors of Part I (`--subset 387`). |
| `UCL_Forecast_Report.py` | Writes `expected_results/forecast/FORECAST_RESULTS.md` from the stored forecasts. |
| `UCL_Forecast_Extra.py` | Extra checks that need no refitting: input sets of one model, other row selections, bias, and errors by the reactor's recent level. |
| `UCL_Forecast_LongMedian.py` | Post hoc check added after the main results: the median of the last 5, 8, 10, 12, 15 and 20 years and of all earlier years as the simple rule, one rule whose window is chosen on the three target years before each test year, and the comparison of the learners with each (bootstrap over reactors and over the 16 test years). Output in `expected_results/forecast/long_median/`. |
| `UCL_Independent_Check.py` | Recomputes the targets, the persistence and median forecasts and the headline scores from the raw PRIS table with a different code path, and compares them with the stored results. |

**Tuning check (robustness of both parts)**

| Script | What it does |
|---|---|
| `TUNING_PRESPECIFICATION.md` and `UCL_Tuning_Spec.py` | The search ranges, the number of candidates (8), the selection scores and the decision rule, **committed before any tuned model was run** (see the git history). |
| `UCL_Benchmark_Tuned.py` | Part I with nested validation: the settings of each model are searched only inside each outer training part. |
| `UCL_Forecast_Tuned.py` | Part II with settings searched for every test year from earlier years only (`--window 15`, `10` or `0`). |

**Figures**

| Script | What it does |
|---|---|
| `UCL_Paper_Figures.py` | The Part I figures (label, PRIS against RDS-2, benchmark, country, importance, stability). |
| `UCL_Paper_Figures_V6.py` | The new figures of the combined paper: overview, Part II forecasts and the tuning check. It trains nothing. |

## How to run

Install Python 3.13, then install the libraries once from the repository root:

```
pip install -r requirements-research.txt
```

Then, from the repository root, in this order:

```
python3 research/UCL_Benchmark_CV.py
python3 research/UCL_Leave_Country_Out.py
python3 research/UCL_Permutation_Importance.py
python3 research/UCL_Label_Stability.py
python3 research/UCL_Threshold_Sensitivity.py
python3 research/UCL_Ukraine_Check.py
python3 research/UCL_Forecast_Panel.py
python3 research/UCL_Forecast_Benchmark.py --window 15 --tag main_window15
python3 research/UCL_Forecast_Benchmark.py --window 10 --tag main_window10
python3 research/UCL_Forecast_Benchmark.py --tag main_expanding
python3 research/UCL_Forecast_Benchmark.py --grouped country --window 15 --sets A C --tag grouped-country_window15
python3 research/UCL_Forecast_Benchmark.py --grouped site --window 15 --sets A C --tag grouped-site_window15
python3 research/UCL_Forecast_Benchmark.py --subset 387 --window 15 --sets D E --tag subset387_window15
python3 research/UCL_Forecast_Report.py
python3 research/UCL_Forecast_Extra.py
python3 research/UCL_Independent_Check.py
python3 research/UCL_Benchmark_Tuned.py          # several hours
python3 research/UCL_Forecast_Tuned.py --window 15
python3 research/UCL_Forecast_Tuned.py --window 10
python3 research/UCL_Forecast_Tuned.py --window 0
python3 research/UCL_Paper_Figures.py
python3 research/UCL_Paper_Figures_V6.py
```

The full set takes many hours on a laptop; the two tuned runs take most of it. Each script can be run on its own once the earlier
ones that it needs have been run (the Part II scripts need the panel; the figure scripts need the result files).

## Keep the pinned library versions

`requirements-research.txt` pins every library. This matters. With scikit-learn 1.8.0 (the pinned version) the scripts reproduce the reference
tables in `expected_results/` exactly. A newer scikit-learn (1.9) changes the Random Forest rows by about one test reactor on a single
split. All scripts use the fixed seed 42, so repeating a run on the same versions gives the same numbers.

## What you should see

`expected_results/` holds the files of the reference runs, one folder per analysis (`benchmark/`, `forecast/`, `tuning/`, ...).
Compare your own output file by file. The most important numbers are printed in the repository README. The limits of the analysis
are listed there and in the paper (Section "Limitations").
