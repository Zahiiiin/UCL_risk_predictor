# Research scripts

These scripts produce the tables and numbers reported in the paper. They find the dataset
(`../data/UCL_Dataset_v5_longitudinal.xlsx`) by themselves.

| Script | What it does |
|---|---|
| `UCL_Model1_Pipeline.py` | Model 1: 13 design features, 407 reactors. Five classifiers, temporal split, bootstrap, figures. |
| `UCL_Model2_Pipeline.py` | Model 2: 18 features (design plus two past UCF periods), 370 reactors. Five classifiers, Model 1 baseline on the same reactors, McNemar test, bootstrap, temporal split, feature importance, figures. |
| `UCL_Model2_History_Ablation.py` | An extra check, not in the paper's main tables. Shows how much each of the two UCF periods adds to Model 2. It imports the Model 2 script, so keep the two files together. |

A full run of each script takes roughly 10 to 40 minutes, depending on the computer.

## How to run

Install Python 3.13, then install the libraries once from the repository root:

```
pip install -r requirements-research.txt
```

Then, from inside this folder:

```
python UCL_Model1_Pipeline.py
python UCL_Model2_Pipeline.py
python UCL_Model2_History_Ablation.py
```

Results appear in `results_Model1/` and `results_Model2/` (tables as CSV, figures as PNG, and for
Model 2 a `run_summary.json` that also records the library versions that were used).
Use `python UCL_Model2_Pipeline.py --quick` for a two-minute smoke test. A quick run uses a very
short search, so its numbers differ from the paper.

## Keep the pinned library versions

`requirements-research.txt` pins every library. This matters. With scikit-learn 1.8.0 (the pinned
version) the scripts reproduce the paper's Model 1 and Model 2 tables exactly. A newer scikit-learn
(1.9) changes the Random Forest rows by one test reactor, which is 1.2 to 1.35 points of accuracy.
Both scripts use the fixed seed 42, so repeating a run on the same versions gives the same numbers.

## What you should see

`expected_results/` holds the tables from the reference run, in `model1/` and `model2/`.
Compare your own output file by file. `expected_results/model2/run_summary.json` lists the
library versions of the reference run, so you can check yours against them.
The most important numbers are printed in the repository README. The limits of the analysis are
listed there too, under "Known limitations".
