# Tuning check: what was decided BEFORE running it

This note was committed before any tuned model was run. The git history shows the order. The exact numbers and lists are in `UCL_Tuning_Spec.py` and `expected_results/tuning/TUNING_SPEC.json`.

## Why do this at all

The models in the paper use fixed, documented settings ("untuned"). A reviewer can fairly ask: would better settings change the conclusions? So we run a second analysis in which the settings are searched, and we say in advance how we will read the outcome.

## The one rule that keeps it honest

Settings are searched only on the training data of each fold. The test fold never takes part in the choice. This is called nested validation. If the settings were chosen on the same data that scores the final result, the score would be too good.

## What is searched

Each model gets 8 candidate settings. Candidate 0 is always the untuned setting. Candidates 1 to 7 are random draws from fixed ranges, made with a fixed seed so anyone gets the same list. So tuning can leave the default only if another candidate scores higher inside the training data.

Part I (class across reactors): logistic regression (regularisation C), random forest and extra trees (depth, minimum leaf size, features per split), XGBoost (depth, learning rate, number of trees). Stacking is not searched by itself. It is built in each fold from the tuned forest, extra trees and XGBoost, with the same fixed final logistic regression as before.

Part II (next-year forecast): random forest and XGBoost only. Ridge stays at its fixed setting because it has a single knob and is not the headline.

## How a winner is picked

Part I: three inner folds inside the outer training part, and the winner is the candidate with the highest mean accuracy. The country encoding, scaling and SMOTE are redone inside every inner fold.

Part II: for each test year Y, the three target years just before Y are used as the inner test, and the winner is the candidate with the lowest mean absolute error. Then the winner is refitted on all years before Y and predicts year Y.

## How the result will be read

1. The headline numbers of the paper stay the untuned ones.
2. The tuned runs are reported in full, whatever they show.
3. Only one tuned run per design. If the result is disappointing, the ranges are not changed and the search is not repeated. A re-run is allowed only to fix a coding error, and then the first version is described too.
4. A tuned model counts as different from its untuned version only if the corrected resampled t-test (Part I) gives p below 0.05, or the 95 percent bootstrap interval of the error difference (Part II) excludes zero. Otherwise the text says "no distinguishable change".
5. A claim stays in the main text only if it holds in both the untuned and the tuned analysis. If it holds in only one, it is written with that caveat.

## What is my own assumption

The ranges, the number of candidates (8) and the inner scores are my own choices, not values from a standard. The paper must say so. Random search with 8 candidates is a limited search, and a larger search could find better settings. The inner scores are noisy because the inner parts are small.

## Run log (added after the runs; the text above was committed before any tuned run)

- Part I (UCL_Benchmark_Tuned.py), Part II 15-year window and Part II 10-year window: one run each; results stored in `expected_results/tuning/`. Short code tests on a few folds were made first; their results were not used.
- Part II with all earlier years (`UCL_Forecast_Tuned.py --window 0`): the first attempt was stopped by the computer's time limit for background jobs on 29 October 2026 at about 23:18 (local time), after 5 of the 16 test years (2010 to 2014). The script writes its result files only at the end, so no result file existed, and only the progress lines (rows and seconds per year) were read. The run was started again at about 23:19 with the same command, the same seed and unchanged code and ranges. The stored result is from the second run. This restart was not caused by a coding error or by a result, and nothing in the design was changed.
