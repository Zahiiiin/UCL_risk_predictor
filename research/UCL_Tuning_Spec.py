"""
UCL_Tuning_Spec.py
==============================================================================
PRE-SPECIFICATION of the hyperparameter-tuning robustness check.

This file was written and committed BEFORE any tuned model was run, so that the
search ranges, the number of candidates, the score used to pick a winner and the
rule for reading the outcome cannot be adjusted after seeing results.
The git history of this file is the evidence of that order.

WHAT IS FIXED HERE
    1. The search ranges of every model (my own choices, not taken from a standard).
    2. How the candidate list is drawn (seeded, so anybody gets the same list).
    3. The score that picks the winner inside the training data.
    4. How the outcome will be read (the decision rule).

The two scripts that use this file are
    UCL_Benchmark_Tuned.py   (Part I, class prediction across reactors)
    UCL_Forecast_Tuned.py    (Part II, next-year UCL forecasting)

Run   python3 UCL_Tuning_Spec.py   to write expected_results/tuning/TUNING_SPEC.json
==============================================================================
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
N_CANDIDATES = 8          # candidate number 0 is always the untuned (default) setting; 1 to 7 are random draws

# ------------------------------------------------------------------------------------------------
# PART I: classification of the UCL class. Tuned inside every outer training fold (nested validation).
# ------------------------------------------------------------------------------------------------
PART1_INNER_FOLDS = 3                       # StratifiedKFold(3, shuffle=True, random_state=42) inside the outer training part
PART1_SELECT_SCORE = "accuracy"             # mean accuracy over the 3 inner validation folds; ties go to the lower candidate number
PART1_SETS = "ABCDE"                        # feature sets of UCL_Benchmark_CV.py that are tuned
PART1_MODELS = ["LR", "RF", "ETC", "XGB", "Stacking"]

PART1_DEFAULT = {
    "LR": {"C": 1.0},
    "RF": {"max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
    "ETC": {"max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
    "XGB": {"max_depth": 6, "learning_rate": 0.3, "n_estimators": 100},
}
PART1_SPACE = {
    "LR": {"C": "log-uniform from 0.001 to 100"},
    "RF": {"max_depth": [3, 5, 8, 12, None], "min_samples_leaf": [1, 2, 3, 5, 8],
           "max_features": ["sqrt", 0.3, 0.5, 0.8]},
    "ETC": {"max_depth": [3, 5, 8, 12, None], "min_samples_leaf": [1, 2, 3, 5, 8],
            "max_features": ["sqrt", 0.3, 0.5, 0.8]},
    "XGB": {"max_depth": [2, 3, 4, 6], "learning_rate": [0.03, 0.05, 0.1, 0.2, 0.3],
            "n_estimators": [100, 200, 300]},
}
PART1_STACKING = ("Stacking is NOT searched on its own. In every outer fold it is built from the tuned RF, ETC and XGB "
                  "chosen in that fold (RF and ETC with 200 trees as in the untuned stack), with the same fixed final "
                  "logistic regression (C = 1, 2000 iterations) and the same internal 5-fold stacking as the untuned stack.")

# ------------------------------------------------------------------------------------------------
# PART II: next-year UCL forecasting. Tuned for every test year with data from before that year only.
# ------------------------------------------------------------------------------------------------
PART2_INNER_YEARS = 3                       # inner test = the 3 target years just before the test year Y (Y-3, Y-2, Y-1)
PART2_SELECT_SCORE = "MAE"                  # mean absolute error in UCL percentage points on those 3 years (original scale); ties to lower number
PART2_SETS = ["A", "C"]                     # feature sets of UCL_Forecast_Benchmark.py
PART2_MODELS = ["RF", "XGB"]                # Ridge stays at its fixed setting (alpha = 10): it has one knob and is not the headline
PART2_WINDOWS = [15, 10, 0]                 # 15 is the primary tuned run; 10 and 0 (all earlier years) are secondary

PART2_DEFAULT = {
    "RF": {"min_samples_leaf": 5, "max_features": 0.5, "max_depth": None},
    "XGB": {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300, "min_child_weight": 5},
}
PART2_SPACE = {
    "RF": {"min_samples_leaf": [1, 3, 5, 10, 20], "max_features": [0.3, 0.5, 0.7, 1.0], "max_depth": [6, 10, None]},
    "XGB": {"max_depth": [2, 3, 4, 6], "learning_rate": [0.02, 0.05, 0.1], "n_estimators": [100, 200, 300],
            "min_child_weight": [1, 5, 10]},
}

# ------------------------------------------------------------------------------------------------
# HOW THE OUTCOME WILL BE READ (decision rule, fixed now)
# ------------------------------------------------------------------------------------------------
DECISION_RULE = [
    "The headline numbers of the paper stay the untuned ones (fixed, documented settings).",
    "The tuned runs are reported in full for every set, model and window that was run, whatever they show. Nothing is dropped.",
    "Only one tuned run per design is made. If the result is disappointing, the ranges are NOT changed and the search is NOT repeated. "
    "A re-run is allowed only to fix a coding error, and then the first version is described too.",
    "A tuned model is said to differ from its untuned version only if the corrected resampled t-test on the 25 outer folds "
    "(Part I, accuracy) has p < 0.05, or the 95 percent bootstrap interval of the MAE difference (Part II) excludes zero. "
    "Otherwise the text says 'no distinguishable change'.",
    "A claim stays in the main text only if it holds in BOTH the untuned and the tuned analysis. If it holds in only one, "
    "the main text states it with that caveat.",
    "Because of the previous point, no tuned number replaces an untuned headline number, even if it is higher.",
]
ASSUMPTIONS = [
    "The ranges, the number of candidates (8) and the inner scores are my own choices, not values taken from a standard. They must be "
    "described as such in the paper.",
    "Random search with a small number of candidates is a limited search. A larger search could find better settings.",
    "Only 3 inner folds (Part I) and 3 inner years (Part II) are used, so the inner score is noisy; the default is always one of "
    "the 8 candidates, so tuning can leave the default only if another candidate scores higher.",
]


# ------------------------------------------------------------------------------------------------
def _draw(space, default, seed, log_c=False):
    """Return N_CANDIDATES settings: the default first, then random distinct draws from the space."""
    rng = np.random.default_rng(seed)
    out = [dict(default)]
    seen = {json.dumps(default, sort_keys=True)}
    tries = 0
    while len(out) < N_CANDIDATES and tries < 10000:
        tries += 1
        if log_c:
            cand = {"C": float(np.round(10 ** rng.uniform(-3, 2), 4))}
        else:
            cand = {}
            for k, vals in space.items():
                cand[k] = vals[int(rng.integers(len(vals)))]
        key = json.dumps(cand, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


_SEEDS1 = {"LR": 1, "RF": 2, "ETC": 3, "XGB": 4}
_SEEDS2 = {"RF": 5, "XGB": 6}


def part1_candidates(model):
    return _draw(PART1_SPACE[model], PART1_DEFAULT[model], SEED + _SEEDS1[model], log_c=(model == "LR"))


def part2_candidates(model):
    return _draw(PART2_SPACE[model], PART2_DEFAULT[model], SEED + _SEEDS2[model])


def spec_dict():
    return {
        "seed": SEED, "n_candidates": N_CANDIDATES,
        "part1": {"inner_folds": PART1_INNER_FOLDS, "select_score": PART1_SELECT_SCORE, "sets": PART1_SETS,
                  "space": PART1_SPACE, "stacking": PART1_STACKING,
                  "candidates": {m: part1_candidates(m) for m in PART1_DEFAULT}},
        "part2": {"inner_years": PART2_INNER_YEARS, "select_score": PART2_SELECT_SCORE, "sets": PART2_SETS,
                  "models": PART2_MODELS, "windows": PART2_WINDOWS, "space": PART2_SPACE,
                  "candidates": {m: part2_candidates(m) for m in PART2_DEFAULT}},
        "decision_rule": DECISION_RULE, "assumptions": ASSUMPTIONS,
    }


if __name__ == "__main__":
    out = os.path.join(HERE, "expected_results", "tuning")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "TUNING_SPEC.json"), "w") as fh:
        json.dump(spec_dict(), fh, indent=1)
    print("wrote", os.path.join(out, "TUNING_SPEC.json"))
    for part, fn, names in (("PART I", part1_candidates, PART1_DEFAULT), ("PART II", part2_candidates, PART2_DEFAULT)):
        for m in names:
            print(part, m)
            for i, c in enumerate(fn(m)):
                print("   ", i, c)
