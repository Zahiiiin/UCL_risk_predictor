"""
UCL_Ukraine_Check.py
==============================================================================
POST-HOC SENSITIVITY CHECK (not part of the pre-specified tuning check; reported as post hoc in the paper).

WHY
    PRIS has no UCL values for the 15 Ukrainian reactors in 2022 to 2024 (the RDS-2 2025 edition states that the
    2022-2024 operating data of Ukrainian reactors had not been submitted when it was printed). Their labels
    therefore rest on 2018-2021 only (4 usable years), while most other reactors use 7 years. This script repeats
    the main benchmark (sets A to E, the same five classifiers, the same 5 x 5 folds, the same seed, the same
    in-fold steps) after removing the Ukrainian reactors, to see whether any conclusion depends on them.

HOW TO RUN (from the repository root)
    python3 research/UCL_Ukraine_Check.py
    Output: research/expected_results/ukraine_check/  (BM_summary.csv, BM_paired_tests.csv, BM_fold_scores.csv, check.json)

It re-uses the functions of UCL_Benchmark_CV.py unchanged; only the list of reactors differs.
==============================================================================
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import UCL_Benchmark_CV as B          # noqa: E402
import UCL_Model2_Pipeline as M       # noqa: E402

OUT = os.path.join(HERE, "expected_results", "ukraine_check")


def main():
    os.makedirs(OUT, exist_ok=True)
    xlsx = M.find_dataset(None)
    all_df = B.build_static_features(M.load_all(xlsx))
    n0 = len(all_df)
    ukr = all_df["Country"].str.upper().str.contains("UKRAINE")
    all_df = all_df[~ukr].reset_index(drop=True)
    sub = all_df[all_df["Model2_Subset"] == 1].reset_index(drop=True)
    print(f"reactors before: {n0}; Ukrainian removed: {int(ukr.sum())}; all: {len(all_df)}; Model 2 subset: {len(sub)}", flush=True)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=B.RS)
    rows, preds = [], {}
    plan = [
        (all_df, M.FEAT_M1, "A. Design (13), all reactors without Ukraine", ["majority", "country"], "all"),
        (sub, M.FEAT_M1, "B. Design (13), Model 2 reactors without Ukraine", ["majority", "country", "persistence"], "sub"),
        (sub, B.DESIGN_NO_COUNTRY, "C. Design without country (12), Model 2 reactors without Ukraine", [], "sub"),
        (sub, M.FEAT_M2, "D. Model 2: design + history (18), without Ukraine", [], "sub"),
        (sub, M.FEAT_TEMPORAL, "E. History only (5), without Ukraine", [], "sub"),
    ]
    for df, feats, label, bl, tag in plan:
        print("running:", label, flush=True)
        B.run_set(df, feats, label, True, bl, cv, tag, rows, preds)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "BM_fold_scores.csv"), index=False)
    B.summarise_and_test(res, OUT)
    json.dump({"reactors_before": int(n0), "ukrainian_removed": int(ukr.sum()), "reactors_after": int(len(all_df)),
               "model2_subset_after": int(len(sub)), "note": "post hoc sensitivity check; same pipeline as UCL_Benchmark_CV.py"},
              open(os.path.join(OUT, "check.json"), "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
