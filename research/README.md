# Research scripts

These scripts produce the tables and numbers reported in the paper. They find the dataset
(`../data/UCL_Dataset_v6_PRIS.xlsx`) by themselves.

| Script | What it does |
|---|---|
| `UCL_Benchmark_CV.py` | **The main evidence.** 5 x repeated 5-fold cross-validation (25 test folds) of five classifiers on nine feature sets, against baselines that need no learning (most common class, most common class of the same country, class of the reactor's own past UCL). Scaling, country encoding and SMOTE are fitted inside each training fold. Paired tests with the Nadeau-Bengio correction. About 15 minutes. |
| `UCL_Model1_Pipeline.py` | Model 1: 13 design features, 387 reactors. Five classifiers, cohort split, bootstrap, figures. Single 80/20 split with tuning. |
| `UCL_Model2_Pipeline.py` | Model 2: 18 features (design plus the mean UCL of 2010-2013 and 2014-2017), 324 reactors. Five classifiers, Model 1 baseline on the same reactors, McNemar test, bootstrap, cohort split, feature importance, figures. Single 80/20 split with tuning. |
| `UCL_Model2_History_Ablation.py` | A single-split version of the history-period check (output files `M2_table9_history_ablation.csv` and `M2_table10_history_correlation.csv`). The cross-validated version is in the benchmark (feature sets B, F, G, H, D, I). It imports the Model 2 script, so keep the two files together. |

A full run of each script takes roughly 10 to 40 minutes, depending on the computer.

## How to run

Install Python 3.13, then install the libraries once from the repository root:

```
pip install -r requirements-research.txt
```

Then, from inside this folder:

```
python UCL_Benchmark_CV.py
python UCL_Model1_Pipeline.py
python UCL_Model2_Pipeline.py
python UCL_Model2_History_Ablation.py
```

Results appear in `results_Benchmark/`, `results_Model1/` and `results_Model2/` (tables as CSV, figures as PNG, and for
Model 2 a `run_summary.json` that also records the library versions that were used).
Use `python UCL_Model2_Pipeline.py --quick` for a two-minute smoke test. A quick run uses a very
short search, so its numbers differ from the paper.

## Keep the pinned library versions

`requirements-research.txt` pins every library. This matters. With scikit-learn 1.8.0 (the pinned
version) the scripts reproduce the reference tables in `expected_results/` exactly. A newer scikit-learn
(1.9) changes the Random Forest rows by about one test reactor, which is 1.3 to 1.5 points of accuracy on a single split.
Both scripts use the fixed seed 42, so repeating a run on the same versions gives the same numbers.

## What you should see

`expected_results/` holds the tables from the reference runs, in `benchmark/`, `model1/` and `model2/`.
Compare your own output file by file. `expected_results/model2/run_summary.json` lists the
library versions of the reference run, so you can check yours against them.
The most important numbers are printed in the repository README. The limits of the analysis are
listed there too, under "Known limitations".
