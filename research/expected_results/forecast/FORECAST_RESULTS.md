# Option B results: forecasting next-year UCL of each reactor

All numbers are read from the result files in this folder by `research/UCL_Forecast_Report.py`. Scores are pooled over target years 2010 to 2025 unless a table says otherwise. MAE is the mean absolute error in percentage points of UCL (smaller is better). Class accuracy is the share of reactor-years whose predicted Low/Medium/High class (limits 2 and 6 percent) equals the true class of that year.

**Panel.** 17,548 reactor-year rows, 603 reactors, 35 countries, target years 1972 to 2025. Leakage check: 400 random rows recomputed from scratch, 0 mismatches.

## Table 1. Rolling-origin forecasts, three training windows

Feature sets: A = own UCL history; C = A + own planned/external losses + world and country context. Window = how many past target years the model may learn from. None of the windows was tuned on the test years.

| Method | MAE, all earlier years | MAE, latest 15 years | MAE, latest 10 years | Class acc. % (15 yr) | AUC high (15 yr) |
|---|---|---|---|---|---|
| Persistence (last usable year) | 4.56 | 4.56 | 4.56 | 61.1 | 0.774 |
| Mean of last 5 years | 4.43 | 4.43 | 4.43 | 60.1 | 0.809 |
| Median of last 5 years | 3.92 | 3.92 | 3.92 | 65.1 | 0.803 |
| Country average (3 years) | 4.57 | 4.57 | 4.57 | 54.3 | 0.800 |
| Ridge, feature set A | 3.78 | 3.78 | 3.76 | 65.4 | 0.808 |
| Ridge, feature set C | 3.69 | 3.74 | 3.73 | 65.9 | 0.823 |
| RF, feature set A | 3.65 | 3.67 | 3.67 | 64.5 | 0.814 |
| RF, feature set C | 3.57 | 3.60 | 3.61 | 65.9 | 0.829 |
| XGB, feature set A | 3.62 | 3.68 | 3.70 | 65.2 | 0.816 |
| XGB, feature set C | 3.80 | 3.63 | 3.66 | 65.9 | 0.830 |

## Table 2. Differences with 95% intervals (reactors resampled 1000 times)

Negative change in MAE = better than the comparison. Rows of one reactor are not independent, so whole reactors are resampled. No correction for the number of comparisons was applied.

| Window | Model | Compared with | Change in MAE (95% CI) | Change in class acc., points (95% CI) |
|---|---|---|---|---|
| All earlier years | Ridge, feature set C | median of last 5 | -0.23 (-0.36 to -0.11) | +0.1 (-1.0 to +1.2) |
| All earlier years | RF, feature set C | median of last 5 | -0.35 (-0.48 to -0.23) | +0.7 (-0.3 to +1.7) |
| All earlier years | XGB, feature set C | median of last 5 | -0.11 (-0.25 to +0.01) | -5.6 (-7.0 to -4.4) |
| All earlier years | Ridge, feature set C | persistence | -0.88 (-1.03 to -0.72) | +4.2 (+2.9 to +5.4) |
| All earlier years | RF, feature set C | persistence | -1.00 (-1.15 to -0.83) | +4.8 (+3.6 to +5.9) |
| All earlier years | XGB, feature set C | persistence | -0.76 (-0.91 to -0.59) | -1.6 (-3.0 to -0.3) |
| Latest 15 years | Ridge, feature set C | median of last 5 | -0.18 (-0.29 to -0.08) | +0.8 (-0.3 to +1.9) |
| Latest 15 years | RF, feature set C | median of last 5 | -0.32 (-0.45 to -0.21) | +0.7 (-0.2 to +1.7) |
| Latest 15 years | XGB, feature set C | median of last 5 | -0.28 (-0.40 to -0.18) | +0.7 (-0.3 to +1.8) |
| Latest 15 years | Ridge, feature set C | persistence | -0.83 (-0.98 to -0.67) | +4.8 (+3.6 to +6.0) |
| Latest 15 years | RF, feature set C | persistence | -0.97 (-1.12 to -0.79) | +4.8 (+3.6 to +5.9) |
| Latest 15 years | XGB, feature set C | persistence | -0.93 (-1.09 to -0.75) | +4.8 (+3.6 to +6.0) |
| Latest 10 years | Ridge, feature set C | median of last 5 | -0.19 (-0.31 to -0.08) | +1.0 (-0.1 to +2.2) |
| Latest 10 years | RF, feature set C | median of last 5 | -0.31 (-0.43 to -0.20) | +0.8 (-0.2 to +1.8) |
| Latest 10 years | XGB, feature set C | median of last 5 | -0.25 (-0.37 to -0.15) | +0.5 (-0.5 to +1.5) |
| Latest 10 years | Ridge, feature set C | persistence | -0.84 (-0.99 to -0.69) | +5.1 (+3.8 to +6.3) |
| Latest 10 years | RF, feature set C | persistence | -0.95 (-1.11 to -0.78) | +4.8 (+3.6 to +6.0) |
| Latest 10 years | XGB, feature set C | persistence | -0.90 (-1.06 to -0.73) | +4.6 (+3.4 to +5.7) |

## Table 3. What the models look at (MAE, 15-year window)

A = own UCL history; B = A + own planned/external losses; C = B + world and country context; D = C + age and size.

| Model | Set A | Set B | Set C | Set D |
|---|---|---|---|---|
| Ridge | 3.78 | 3.78 | 3.74 | 3.75 |
| RF | 3.67 | 3.64 | 3.60 | 3.59 |
| XGB | 3.68 | 3.65 | 3.63 | 3.62 |

## Table 4. Early and late half of the test period (15-year window)

| Target years | Method | MAE | Class acc. % |
|---|---|---|---|
| 2010-2017 | Persistence (last usable year) | 4.62 | 58.6 |
| 2010-2017 | Median of last 5 years | 4.06 | 64.1 |
| 2010-2017 | RF, feature set C | 3.83 | 63.7 |
| 2018-2025 | Persistence (last usable year) | 4.51 | 63.6 |
| 2018-2025 | Median of last 5 years | 3.78 | 66.2 |
| 2018-2025 | RF, feature set C | 3.36 | 68.1 |

## Table 5. Do the models work for places they have not seen? (MAE, 15-year window)

Reference rules on the same rows: persistence 4.56, median of last 5 3.92. Sites are guessed from unit names (heuristic, not an IAEA code). Six folds; every training row of the held-out group is removed.

| Test | RF, set A | RF, set C | XGB, set A | XGB, set C | Ridge, set A | Ridge, set C |
|---|---|---|---|---|---|---|
| Same countries and sites allowed in training | 3.67 | 3.60 | 3.68 | 3.63 | 3.78 | 3.74 |
| New site (no unit of the site in training) | 3.70 | 3.63 | 3.73 | 3.67 | 3.77 | 3.73 |
| New country (no unit of the country in training) | 3.75 | 3.78 | 3.76 | 3.82 | 3.84 | 4.05 |

| New-country model | Change in MAE vs median of last 5 (95% CI) | Change in class acc., points (95% CI) |
|---|---|---|
| RF, feature set A | -0.17 (-0.30 to -0.06) | -1.9 (-2.9 to -0.9) |
| XGB, feature set A | -0.15 (-0.28 to -0.04) | -1.6 (-2.6 to -0.5) |
| Ridge, feature set A | -0.08 (-0.19 to +0.02) | -0.6 (-1.8 to +0.5) |
| RF, feature set C | -0.14 (-0.26 to -0.02) | -3.6 (-4.8 to -2.3) |
| Ridge, feature set C | +0.13 (+0.00 to +0.25) | -14.2 (-16.8 to -11.4) |

## Table 6. In how many of the 16 test years does the model have the lower MAE?

| Window | Model | Years better than median of last 5 | Years better than persistence |
|---|---|---|---|
| All earlier years | RF, feature set C | 16 of 16 | 16 of 16 |
| All earlier years | XGB, feature set C | 11 of 16 | 16 of 16 |
| All earlier years | Ridge, feature set C | 16 of 16 | 16 of 16 |
| Latest 15 years | RF, feature set C | 16 of 16 | 16 of 16 |
| Latest 15 years | XGB, feature set C | 16 of 16 | 16 of 16 |
| Latest 15 years | Ridge, feature set C | 11 of 16 | 16 of 16 |
| Latest 10 years | RF, feature set C | 16 of 16 | 16 of 16 |
| Latest 10 years | XGB, feature set C | 15 of 16 | 16 of 16 |
| Latest 10 years | Ridge, feature set C | 9 of 16 | 16 of 16 |

## Table 7. Do design facts help? (the 387 reactors of the paper, 15-year window)

Reference rules on these rows: persistence 4.17, median of last 5 3.40.

| Model | MAE without design facts (set D) | MAE with reactor type and construction time (set E) |
|---|---|---|
| Ridge | 3.28 | 3.30 |
| RF | 3.10 | 3.10 |
| XGB | 3.14 | 3.15 |
